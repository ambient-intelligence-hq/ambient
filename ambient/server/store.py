"""PostgreSQL-backed persistence for sessions, events, and managed-agents metadata.

Async via an asyncpg connection pool so multiple uvicorn workers / replicas share
one durable store. Payloads are kept as JSONB; the store stays schema-light and
serializes whole records, mirroring the original SQLite design.

Tables:
  sessions(id, status, payload_json, messages, ...)  -- `messages` is the runner's
      conversation context, persisted so any worker can rehydrate a session.
  events(seq BIGSERIAL, ...)                          -- append-only event log.
  agents / environments / files                       -- managed-agents metadata
      that used to live in in-memory app.state dicts.
"""
from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from typing import Any, Optional

import asyncpg

from ambient.config import settings


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def _new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:16]}"


SCHEMA = """
CREATE TABLE IF NOT EXISTS sessions (
    id TEXT PRIMARY KEY,
    status TEXT NOT NULL,
    payload_json JSONB NOT NULL,
    messages JSONB NOT NULL DEFAULT '[]'::jsonb,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS events (
    seq BIGSERIAL PRIMARY KEY,
    event_id TEXT NOT NULL UNIQUE,
    session_id TEXT NOT NULL,
    type TEXT NOT NULL,
    payload_json JSONB NOT NULL,
    ts TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS events_session_seq_idx ON events(session_id, seq);

CREATE TABLE IF NOT EXISTS agents (
    id TEXT PRIMARY KEY,
    payload_json JSONB NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS environments (
    id TEXT PRIMARY KEY,
    payload_json JSONB NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS files (
    id TEXT PRIMARY KEY,
    payload_json JSONB NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
"""


class Store:
    def __init__(self, dsn: Optional[str] = None) -> None:
        self.dsn = dsn or settings.database_url
        self._pool: Optional[asyncpg.Pool] = None

    async def connect(self) -> None:
        self._pool = await asyncpg.create_pool(self.dsn, min_size=1, max_size=10)
        async with self._pool.acquire() as conn:
            await conn.execute(SCHEMA)

    async def close(self) -> None:
        if self._pool is not None:
            await self._pool.close()
            self._pool = None

    # --- sessions ---------------------------------------------------------

    async def create_session(self, payload: dict[str, Any]) -> dict[str, Any]:
        session_id = payload.get("session_id") or _new_id("ses")
        now = _now()
        record = dict(payload)
        record.setdefault("session_id", session_id)
        record.setdefault("created_at", now)
        record["updated_at"] = now
        record.setdefault("status", "created")

        await self._pool.execute(
            "INSERT INTO sessions(id, status, payload_json, created_at, updated_at) "
            "VALUES ($1, $2, $3, $4, $5)",
            session_id, record["status"], json.dumps(record), record["created_at"], record["updated_at"],
        )
        return record

    async def get_session(self, session_id: str) -> Optional[dict[str, Any]]:
        row = await self._pool.fetchrow(
            "SELECT payload_json FROM sessions WHERE id = $1", session_id
        )
        return json.loads(row["payload_json"]) if row else None

    async def update_session(self, session_id: str, mutator) -> dict[str, Any]:
        async with self._pool.acquire() as conn:
            async with conn.transaction():
                row = await conn.fetchrow(
                    "SELECT payload_json FROM sessions WHERE id = $1 FOR UPDATE",
                    session_id,
                )
                if row is None:
                    raise KeyError(session_id)
                record = json.loads(row["payload_json"])
                record = mutator(record)
                record["updated_at"] = _now()
                await conn.execute(
                    "UPDATE sessions SET status = $1, payload_json = $2, updated_at = $3 WHERE id = $4",
                    record["status"], json.dumps(record), record["updated_at"], session_id,
                )
                return record

    async def list_sessions(self, limit: int = 50) -> list[dict[str, Any]]:
        rows = await self._pool.fetch(
            "SELECT payload_json FROM sessions ORDER BY created_at DESC LIMIT $1", limit
        )
        return [json.loads(r["payload_json"]) for r in rows]

    async def delete_session(self, session_id: str) -> bool:
        result = await self._pool.execute("DELETE FROM sessions WHERE id = $1", session_id)
        # asyncpg returns e.g. "DELETE 1"
        return result.rsplit(" ", 1)[-1] != "0"

    # --- runner messages (rehydration state) ------------------------------

    async def get_messages(self, session_id: str) -> list[dict[str, Any]]:
        row = await self._pool.fetchrow(
            "SELECT messages FROM sessions WHERE id = $1", session_id
        )
        if row is None or row["messages"] is None:
            return []
        return json.loads(row["messages"])

    async def save_messages(self, session_id: str, messages: list[dict[str, Any]]) -> None:
        await self._pool.execute(
            "UPDATE sessions SET messages = $1, updated_at = $2 WHERE id = $3",
            json.dumps(messages), _now(), session_id,
        )

    # --- events -----------------------------------------------------------

    async def append_event(
        self,
        session_id: str,
        type: str,
        payload: dict[str, Any],
    ) -> dict[str, Any]:
        event_id = _new_id("evt")
        ts = _now()
        seq = await self._pool.fetchval(
            "INSERT INTO events(event_id, session_id, type, payload_json, ts) "
            "VALUES ($1, $2, $3, $4, $5) RETURNING seq",
            event_id, session_id, type, json.dumps(payload), ts,
        )
        return {
            "event_id": event_id,
            "seq": seq,
            "session_id": session_id,
            "type": type,
            "payload": payload,
            "ts": ts,
        }

    async def list_events(
        self,
        session_id: str,
        after_seq: int = 0,
        limit: int = 1000,
    ) -> list[dict[str, Any]]:
        rows = await self._pool.fetch(
            "SELECT event_id, seq, type, payload_json, ts FROM events "
            "WHERE session_id = $1 AND seq > $2 ORDER BY seq ASC LIMIT $3",
            session_id, after_seq, limit,
        )
        return [
            {
                "event_id": r["event_id"],
                "seq": r["seq"],
                "type": r["type"],
                "payload": json.loads(r["payload_json"]),
                "ts": r["ts"],
            }
            for r in rows
        ]

    # --- managed-agents metadata (agents / environments / files) ----------

    async def _put_record(self, table: str, record: dict[str, Any]) -> dict[str, Any]:
        await self._pool.execute(
            f"INSERT INTO {table}(id, payload_json, created_at, updated_at) "
            "VALUES ($1, $2, $3, $4) "
            "ON CONFLICT (id) DO UPDATE SET payload_json = EXCLUDED.payload_json, "
            "updated_at = EXCLUDED.updated_at",
            record["id"], json.dumps(record),
            record.get("created_at") or _now(), record.get("updated_at") or _now(),
        )
        return record

    async def _get_record(self, table: str, record_id: str) -> Optional[dict[str, Any]]:
        row = await self._pool.fetchrow(
            f"SELECT payload_json FROM {table} WHERE id = $1", record_id
        )
        return json.loads(row["payload_json"]) if row else None

    async def _list_records(self, table: str, limit: int = 1000) -> list[dict[str, Any]]:
        rows = await self._pool.fetch(
            f"SELECT payload_json FROM {table} ORDER BY created_at DESC LIMIT $1", limit
        )
        return [json.loads(r["payload_json"]) for r in rows]

    async def put_agent(self, record: dict[str, Any]) -> dict[str, Any]:
        return await self._put_record("agents", record)

    async def get_agent(self, agent_id: str) -> Optional[dict[str, Any]]:
        return await self._get_record("agents", agent_id)

    async def list_agents(self, limit: int = 1000) -> list[dict[str, Any]]:
        return await self._list_records("agents", limit)

    async def put_environment(self, record: dict[str, Any]) -> dict[str, Any]:
        return await self._put_record("environments", record)

    async def get_environment(self, environment_id: str) -> Optional[dict[str, Any]]:
        return await self._get_record("environments", environment_id)

    async def put_file(self, record: dict[str, Any]) -> dict[str, Any]:
        # File records key on the video id (== Files API file id).
        rec = dict(record)
        rec.setdefault("id", record.get("video_id"))
        rec.setdefault("created_at", record.get("created_at") or _now())
        rec.setdefault("updated_at", _now())
        return await self._put_record("files", rec)

    async def get_file(self, file_id: str | None) -> Optional[dict[str, Any]]:
        return await self._get_record("files", file_id)

    async def update_file(self, file_id: str, mutator) -> dict[str, Any]:
        """Read-modify-write a file record under a row lock.

        `put_file` overwrites the whole payload, so concurrent status updates
        (upload handler, ingest consumer, sweeper) must serialize through here.
        Mirrors `update_session`. Raises KeyError if the file is gone.
        """
        async with self._pool.acquire() as conn:
            async with conn.transaction():
                row = await conn.fetchrow(
                    "SELECT payload_json FROM files WHERE id = $1 FOR UPDATE", file_id
                )
                if row is None:
                    raise KeyError(file_id)
                record = json.loads(row["payload_json"])
                record = mutator(record)
                record["updated_at"] = _now()
                await conn.execute(
                    "UPDATE files SET payload_json = $1, updated_at = $2 WHERE id = $3",
                    json.dumps(record), record["updated_at"], file_id,
                )
                return record

    async def list_stale_ingest_files(
        self, cutoff: str, limit: int = 100
    ) -> list[dict[str, Any]]:
        """Files stuck in pending/processing past `cutoff` (for the sweeper).

        `updated_at` is an ISO-8601 UTC string (always 'Z', fixed width via
        `_now()`), so lexical comparison is chronological.
        """
        rows = await self._pool.fetch(
            "SELECT payload_json FROM files "
            "WHERE payload_json->>'description_status' IN ('pending', 'processing') "
            "AND updated_at < $1 ORDER BY updated_at ASC LIMIT $2",
            cutoff, limit,
        )
        return [json.loads(r["payload_json"]) for r in rows]

    async def list_files(self, limit: int = 1000) -> list[dict[str, Any]]:
        return await self._list_records("files", limit)

    async def delete_file(self, file_id: str) -> Optional[dict[str, Any]]:
        async with self._pool.acquire() as conn:
            async with conn.transaction():
                row = await conn.fetchrow(
                    "SELECT payload_json FROM files WHERE id = $1 FOR UPDATE", file_id
                )
                if row is None:
                    return None
                await conn.execute("DELETE FROM files WHERE id = $1", file_id)
                return json.loads(row["payload_json"])

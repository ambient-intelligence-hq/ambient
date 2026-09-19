"""In-memory Store and Broker for driving a SessionRunner without infra.

The production server keeps durable state in Postgres (`server/store.py`) and the
per-session run lease + SSE fan-out in Redis (`server/broker.py`). A single-user,
single-process CLI run needs neither: one session, one run, one consumer, no
rehydration, no cross-worker fan-out. `LiteStore` and `LiteBroker` implement just
the method surface `SessionRunner` touches, backed by plain dicts and an
asyncio.Queue, so `ambient.cli` can run the exact same agent loop the server uses
with no external services.

Surface required by SessionRunner (see server/runner.py):
  Store  -> get_session, update_session, append_event, save_messages,
            get_file, update_file
  Broker -> publish, renew, release
Plus create_session / put_file used here to seed the run.
"""
from __future__ import annotations

import asyncio
import uuid
from datetime import datetime, timezone
from typing import Any, AsyncIterator, Callable, Optional


def _now() -> str:
    return (
        datetime.now(timezone.utc)
        .isoformat(timespec="milliseconds")
        .replace("+00:00", "Z")
    )


def _new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


class LiteStore:
    """Dict-backed stand-in for `server.store.Store` (no Postgres)."""

    def __init__(self) -> None:
        self._sessions: dict[str, dict[str, Any]] = {}
        self._messages: dict[str, list[dict[str, Any]]] = {}
        self._events: dict[str, list[dict[str, Any]]] = {}
        self._files: dict[str, dict[str, Any]] = {}
        self._seq = 0

    # --- sessions ---------------------------------------------------------
    async def create_session(self, payload: dict[str, Any]) -> dict[str, Any]:
        record = dict(payload)
        sid = record.setdefault("session_id", _new_id("ses"))
        record.setdefault("created_at", _now())
        record["updated_at"] = _now()
        record.setdefault("status", "created")
        self._sessions[sid] = record
        return record

    async def get_session(self, session_id: str) -> Optional[dict[str, Any]]:
        return self._sessions.get(session_id)

    async def update_session(self, session_id: str, mutator: Callable) -> dict[str, Any]:
        record = self._sessions.get(session_id)
        if record is None:
            raise KeyError(session_id)
        record = mutator(record)
        record["updated_at"] = _now()
        self._sessions[session_id] = record
        return record

    # --- runner messages --------------------------------------------------
    async def save_messages(self, session_id: str, messages: list[dict[str, Any]]) -> None:
        self._messages[session_id] = list(messages)

    async def get_messages(self, session_id: str) -> list[dict[str, Any]]:
        return list(self._messages.get(session_id, []))

    # --- events -----------------------------------------------------------
    async def append_event(self, session_id: str, type: str, payload: dict[str, Any]) -> dict[str, Any]:
        self._seq += 1
        ev = {
            "event_id": _new_id("evt"),
            "seq": self._seq,
            "session_id": session_id,
            "type": type,
            "payload": payload,
            "ts": _now(),
        }
        self._events.setdefault(session_id, []).append(ev)
        return ev

    # --- files ------------------------------------------------------------
    async def put_file(self, record: dict[str, Any]) -> dict[str, Any]:
        self._files[record["video_id"]] = dict(record)
        return record

    async def get_file(self, file_id: str | None) -> Optional[dict[str, Any]]:
        if file_id is None:
            return None
        return self._files.get(file_id)

    async def update_file(self, file_id: str, mutator: Callable) -> dict[str, Any]:
        record = self._files.get(file_id)
        if record is None:
            raise KeyError(file_id)
        record = mutator(record)
        self._files[file_id] = record
        return record


class LiteBroker:
    """asyncio.Queue-backed stand-in for `server.broker.Broker` (no Redis).

    `publish` feeds a single local consumer (the CLI's event renderer) instead of
    a Redis pub/sub fan-out. The run-ownership lease is a no-op — there is exactly
    one run in one process.
    """

    def __init__(self) -> None:
        self._queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue()

    async def publish(self, session_id: str, event: dict[str, Any]) -> None:
        await self._queue.put(event)

    async def renew(self, session_id: str) -> None:
        return None

    async def release(self, session_id: str) -> None:
        return None

    async def claim(self, session_id: str) -> bool:
        return True

    async def events(self) -> AsyncIterator[dict[str, Any]]:
        """Yield published events until an explicit `{"__close__": True}` sentinel.

        The caller publishes the sentinel *after* the run task has fully finished,
        so everything the run emitted — including `run.completed` and the trailing
        `session.status_changed` from the runner's finally — is drained first and
        the queue is left empty for the next turn. Stopping on `run.completed`
        instead would race that trailing event and strand it for the next turn.
        """
        while True:
            ev = await self._queue.get()
            if ev.get("__close__"):
                return
            yield ev

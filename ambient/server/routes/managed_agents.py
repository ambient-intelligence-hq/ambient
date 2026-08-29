"""Anthropic Managed Agents protocol, backed by the Ambient video-agent engine.

This router makes the server speak the same HTTP surface the Anthropic Python
SDK (`client.beta.agents` / `client.beta.sessions`) talks to, so the SDK can
drive our video agent directly:

    agent = client.beta.agents.create(model=..., name=..., tools=[...])
    env   = client.beta.environments.create(name=...)
    sess  = client.beta.sessions.create(
                agent=agent.id, environment_id=env.id,
                metadata={"video_id": "..."})        # video lives here
    client.beta.sessions.events.send(sess.id, events=[{...user.message...}])
    for ev in client.beta.sessions.events.stream(sess.id):
        ...

Mapping to the engine:
  - Agent      -> per-session model (and optional system override). Persisted in
                  the store (agents table).
  - Environment-> a stub; the local sandbox needs no container config. Persisted
                  in the store (environments table).
  - Session    -> a SessionRunner (boots the sandbox for the video).
  - events.send-> runner.submit_user_message (starts a run).
  - events.stream -> the runner's event stream, reshaped into Managed Agents
                  session events.

The SDK constructs responses leniently (missing fields -> None), so responses
return only the fields a client realistically reads, kept JSON-serializable.

Note on auth: the SDK sends `x-api-key`; `require_api_key` also accepts bearer.
The `?beta=true` query param the SDK appends is ignored by FastAPI routing.
"""
from __future__ import annotations

import asyncio
import json
import os
from typing import Any, Optional

from fastapi import APIRouter, Body, Depends, Query, Request
from sse_starlette.sse import EventSourceResponse
from starlette.datastructures import UploadFile as StarletteUploadFile

from ambient.config import settings
from ambient.server.auth import require_api_key
from ambient.server.errors import bad_request, conflict, not_found
from ambient.server.runner import SandboxNotReady, SessionRunner, _empty_usage
from ambient.server.sandbox import SandboxLimits
from ambient.server.store import _new_id, _now
from ambient.server.video_store import content_video_id, create_youtube_video, store_video
import logging

log = logging.getLogger(__name__)

router = APIRouter(prefix="/v1", dependencies=[Depends(require_api_key)])

# Stable IDs for the auto-seeded default agent/environment. Sessions fall back to
# these when the client omits `agent` / `environment_id`, so callers can skip the
# agents.create / environments.create steps entirely.
DEFAULT_AGENT_ID = "ambient_v1"
DEFAULT_FAST_AGENT_ID = "ambient_fast"
DEFAULT_ENV_ID = "default"


async def seed_defaults(store) -> None:
    """Idempotently ensure a default agent + environment exist in the store.

    Called once at server startup. Uses get-then-put so a later edit to either
    record (via the API) is never clobbered on restart.
    """
    if await store.get_agent(DEFAULT_AGENT_ID) is None:
        now = _now()
        await store.put_agent({
            "id": DEFAULT_AGENT_ID,
            "version": 1,
            "name": settings.default_agent_name,
            "description": "Default video-analysis agent — agent mode (tool loop).",
            "model": settings.agent_model,
            "system": settings.default_agent_system,
            "tools": [{"type": "video_agent_20260825"}],
            "skills": [],
            "mcp_servers": [],
            "metadata": {},
            "created_at": now,
            "updated_at": now,
        })
    if await store.get_agent(DEFAULT_FAST_AGENT_ID) is None:
        now = _now()
        await store.put_agent({
            "id": DEFAULT_FAST_AGENT_ID,
            "version": 1,
            "name": "Video Analyst (Fast)",
            "description": "Fast mode — single dense-frame vision pass, no tools.",
            # Fast mode is a single call on the vision model directly.
            "model": settings.llm_model,
            "system": settings.default_agent_system,
            "tools": [{"type": "video_fast_20260825"}],
            "skills": [],
            "mcp_servers": [],
            "metadata": {},
            "created_at": now,
            "updated_at": now,
        })
    if await store.get_environment(DEFAULT_ENV_ID) is None:
        now = _now()
        await store.put_environment({
            "id": DEFAULT_ENV_ID,
            "name": "default",
            "metadata": {},
            "created_at": now,
            "updated_at": now,
        })


# --------------------------------------------------------------------------
# Agents
# --------------------------------------------------------------------------

def _model_id(model: Any) -> str:
    # `model` may be a plain string or a {"id": "...", ...} model_config object.
    if isinstance(model, dict):
        return model.get("id") or settings.agent_model
    return model or settings.agent_model


def _agent_resource(rec: dict) -> dict:
    return {
        "type": "agent",
        "id": rec["id"],
        "version": rec["version"],
        "name": rec.get("name"),
        "description": rec.get("description"),
        "model": {"id": _model_id(rec.get("model")), "speed": None},
        "system": rec.get("system"),
        "tools": rec.get("tools") or [],
        "skills": rec.get("skills") or [],
        "mcp_servers": rec.get("mcp_servers") or [],
        "metadata": rec.get("metadata") or {},
        "multiagent": None,
        "created_at": rec["created_at"],
        "updated_at": rec["updated_at"],
    }


@router.post("/agents")
async def create_agent(request: Request, body: dict[str, Any] = Body(...)) -> dict:
    if not body.get("model"):
        raise bad_request("agent.model is required")
    now = _now()
    rec = {
        "id": _new_id("agt"),
        "version": 1,
        "name": body.get("name"),
        "description": body.get("description"),
        "model": body.get("model"),
        "system": body.get("system"),
        "tools": body.get("tools") or [],
        "skills": body.get("skills") or [],
        "mcp_servers": body.get("mcp_servers") or [],
        "metadata": body.get("metadata") or {},
        "created_at": now,
        "updated_at": now,
    }
    await request.app.state.store.put_agent(rec)
    return _agent_resource(rec)


@router.get("/agents/{agent_id}")
async def get_agent(agent_id: str, request: Request) -> dict:
    rec = await request.app.state.store.get_agent(agent_id)
    if rec is None:
        raise not_found("agent", agent_id)
    return _agent_resource(rec)


@router.get("/agents")
async def list_agents(request: Request) -> dict:
    recs = await request.app.state.store.list_agents()
    return {"data": [_agent_resource(r) for r in recs], "has_more": False}


# --------------------------------------------------------------------------
# Environments (stub — the local sandbox needs no container config)
# --------------------------------------------------------------------------

def _environment_resource(rec: dict) -> dict:
    return {
        "type": "environment",
        "id": rec["id"],
        "name": rec.get("name"),
        "metadata": rec.get("metadata") or {},
        "created_at": rec["created_at"],
        "updated_at": rec["updated_at"],
    }


@router.post("/environments")
async def create_environment(request: Request, body: dict[str, Any] = Body(default={})) -> dict:
    now = _now()
    rec = {
        "id": _new_id("env"),
        "name": body.get("name"),
        "metadata": body.get("metadata") or {},
        "created_at": now,
        "updated_at": now,
    }
    await request.app.state.store.put_environment(rec)
    return _environment_resource(rec)


@router.get("/environments/{environment_id}")
async def get_environment(environment_id: str, request: Request) -> dict:
    rec = await request.app.state.store.get_environment(environment_id)
    if rec is None:
        raise not_found("environment", environment_id)
    return _environment_resource(rec)


# --------------------------------------------------------------------------
# Files (SDK Files API) — upload a video and get a file id to mount on a session
# --------------------------------------------------------------------------

def _file_metadata(rec: dict, *, include_description: bool = False) -> dict:
    # The file id is the video_id, so a {"type":"file","file_id":...} session
    # resource resolves straight back to the video.
    meta = {
        "type": "file",
        "id": rec["video_id"],
        "filename": rec["filename"],
        "mime_type": rec["mime_type"],
        "size_bytes": rec["size_bytes"],
        "created_at": rec["created_at"],
        "downloadable": False,
        "source_type": rec.get("source_type"),
        "source_status": rec.get("source_status"),
        # Background ingestion state (None for inprocess/no-R2 uploads).
        "description_status": rec.get("description_status"),
    }
    if include_description:
        meta["description"] = rec.get("description")
        meta["description_error"] = rec.get("description_error")
        meta["source_error"] = rec.get("source_error")
        if rec.get("youtube"):
            meta["youtube"] = rec.get("youtube")
    return meta


@router.post("/files")
async def upload_file(request: Request) -> dict:
    form = await request.form()
    upload = next((v for v in form.values() if isinstance(v, StarletteUploadFile)), None)
    if upload is None:
        raise bad_request("multipart form must include a file part")
    data = await upload.read()
    if not data:
        raise bad_request("uploaded file is empty")
    # Content-addressed video_id: if the same bytes were ingested before, reuse
    # that row (and its ready/in-flight description) instead of re-writing the
    # file, re-uploading to R2, resetting status, or re-enqueueing ingestion.
    existing = await request.app.state.store.get_file(content_video_id(data))
    if existing is not None:
        log.info("Reusing existing file %s for duplicate upload", existing["video_id"])
        return _file_metadata(existing)
    meta = store_video(data, upload.filename, upload.content_type)
    # Kick off background description ingestion only when the video reached R2 —
    # the ephemeral ingest sandbox fetches it from there. Without an R2 copy
    # (local inprocess dev) there's no job; the session computes live instead.
    if meta.get("r2_key"):
        meta.update(
            {
                "description": None,
                "description_status": "pending",
                "description_error": None,
                "description_attempts": 0,
            }
        )
    await request.app.state.store.put_file(meta)
    if meta.get("r2_key"):
        log.info(f"Enqueuing video {meta['video_id']} for description ingestion")
        await request.app.state.ingest.enqueue(meta["video_id"])
    return _file_metadata(meta)


@router.post("/files/import")
async def import_file(request: Request, body: dict[str, Any] = Body(...)) -> dict:
    if not settings.youtube_import_enabled:
        raise bad_request("YouTube URL imports are disabled")
    raw_source_type = body.get("source_type") or "youtube"
    if not isinstance(raw_source_type, str):
        raise bad_request("source_type must be a string")
    source_type = raw_source_type.lower()
    if source_type != "youtube":
        raise bad_request("only source_type='youtube' is supported")
    url = body.get("url")
    if not isinstance(url, str) or not url.strip():
        raise bad_request("url is required")
    try:
        meta = create_youtube_video(url.strip(), filename=body.get("filename"))
    except ValueError as exc:
        raise bad_request(str(exc))

    # Content-addressed video_id (keyed on the YouTube URL): reuse a prior import
    # of the same video rather than re-preparing the source and description.
    existing = await request.app.state.store.get_file(meta["video_id"])
    if existing is not None:
        log.info("Reusing existing file %s for duplicate YouTube import", meta["video_id"])
        return _file_metadata(existing)

    metadata = body.get("metadata") or {}
    if metadata and isinstance(metadata, dict):
        meta["metadata"] = metadata
    await request.app.state.store.put_file(meta)
    log.info("Enqueuing YouTube import %s for source preparation", meta["video_id"])
    await request.app.state.ingest.enqueue(meta["video_id"])
    return _file_metadata(meta)


@router.get("/files/{file_id}")
async def get_file(file_id: str, request: Request) -> dict:
    rec = await request.app.state.store.get_file(file_id)
    if rec is None:
        raise not_found("file", file_id)
    return _file_metadata(rec, include_description=True)


@router.get("/files")
async def list_files(request: Request) -> dict:
    recs = await request.app.state.store.list_files()
    return {"data": [_file_metadata(r) for r in recs], "has_more": False}


@router.delete("/files/{file_id}")
async def delete_file(file_id: str, request: Request) -> dict:
    rec = await request.app.state.store.delete_file(file_id)
    if rec is None:
        raise not_found("file", file_id)
    local_path = rec.get("local_path")
    if local_path and os.path.exists(local_path):
        try:
            os.remove(local_path)
        except OSError:
            pass
    return {"type": "file_deleted", "id": file_id}


# --------------------------------------------------------------------------
# Sessions
# --------------------------------------------------------------------------

async def _resolve_agent_ref(agent: Any, store) -> Optional[dict]:
    """`agent` is the agent id string, or {"id": ..., "version": ...}."""
    agent_id = agent.get("id") if isinstance(agent, dict) else agent
    if not agent_id:
        return None
    return await store.get_agent(agent_id)


async def _video_id_from_request(body: dict, store) -> Optional[str]:
    """Where the video lives in the Managed Agents shape.

    Preferred: session metadata {"video_id": "..."}. Also accept a session
    resource — either a Files API mount {"type":"file","file_id":...} (resolved
    via the uploaded-files store) or a resource carrying video_id/id/name.
    """
    meta = body.get("metadata") or {}
    if meta.get("video_id"):
        return meta["video_id"]
    for r in body.get("resources") or []:
        if not isinstance(r, dict):
            continue
        if r.get("type") == "file" and r.get("file_id"):
            rec = await store.get_file(r["file_id"])
            return rec["video_id"] if rec else r["file_id"]
        vid = r.get("video_id") or r.get("id") or r.get("name")
        if vid:
            return vid
    return None


def _session_resource(record: dict) -> dict:
    sb = record.get("sandbox") or {}
    return {
        "type": "session",
        "id": record["session_id"],
        "status": _session_status(record.get("status")),
        "agent": record.get("ma_agent") or {"type": "agent", "id": None, "version": None},
        "environment_id": record.get("environment_id"),
        "title": record.get("title"),
        "metadata": record.get("metadata") or {},
        "resources": record.get("ma_resources") or [],
        "outcome_evaluations": [],
        "vault_ids": [],
        "stats": {},
        "usage": record.get("usage") or {},
        "created_at": record["created_at"],
        "updated_at": record["updated_at"],
        "sandbox": {"backend": sb.get("backend"), "status": sb.get("status"), "id": sb.get("id")},
    }


def _session_status(ambient_status: Optional[str]) -> str:
    # Map our lifecycle onto the Managed Agents session status literals.
    return {
        "created": "pending",
        "starting": "pending",
        "ready": "idle",
        "running": "running",
        "failed": "terminated",
        "terminated": "terminated",
    }.get(ambient_status or "", "idle")


def _mode_from_agent(agent_rec: dict, metadata: dict) -> str:
    """Select the track: 'agent' (tool loop) vs 'fast' (single dense-frame call).

    An explicit metadata.mode wins; otherwise it's read from the agent's declared
    tools — a `video_agent_*` (or legacy `agent_toolset_*`) marker means agent
    mode, and anything else (a `video_fast_*` marker or no tools) means fast mode.
    """
    m = (metadata or {}).get("mode")
    if m in ("fast", "agent"):
        return m
    for t in agent_rec.get("tools") or []:
        typ = (t or {}).get("type", "") if isinstance(t, dict) else ""
        if typ.startswith("video_agent") or typ.startswith("agent_toolset"):
            return "agent"
    return "fast"


async def build_session_record(body: dict, store) -> dict:
    agent_rec = await _resolve_agent_ref(body.get("agent") or DEFAULT_AGENT_ID, store)
    if agent_rec is None:
        raise bad_request("session.agent must reference an existing agent (call agents.create first)")
    environment_id = body.get("environment_id") or DEFAULT_ENV_ID
    
    video_id = await _video_id_from_request(body, store)
    if not video_id:
        raise bad_request("video_id is required: pass metadata={'video_id': '...'} or a resource carrying it")

    model = _model_id(agent_rec.get("model"))
    session_id = _new_id("ses")
    now = _now()
    # The Managed Agents session has no turns field; the video agent typically
    # needs ~6 turns, so default well above settings.max_turns_per_run (which
    # exists for the native API). Allow metadata override for tuning.
    meta = body.get("metadata") or {}
    try:
        max_turns = int(meta.get("max_turns_per_run") or 20)
    except (TypeError, ValueError):
        max_turns = 20
        
    backend = settings.sandbox_backend
    limits_dict = {
        "cpu_seconds": settings.sandbox_cpu_seconds,
        "memory_mb": settings.sandbox_memory_mb,
        "wall_seconds": settings.sandbox_wall_seconds,
    }
    ma_agent = {"type": "agent", "id": agent_rec["id"], "version": agent_rec["version"]}
    mode = _mode_from_agent(agent_rec, meta)
    return {
        "session_id": session_id,
        "status": "created",
        "video_id": video_id,
        "model": model,
        "mode": mode,
        "agent_system": agent_rec.get("system"),
        "title": body.get("title"),
        "metadata": body.get("metadata") or {},
        "permission_policy": {"type": "always_allow", "tools": None},
        "sandbox": {"backend": backend, "id": None, "status": "starting",
                    "boot_ms": None, "limits": limits_dict, "region": None},
        "quota": {},
        "usage": _empty_usage(),
        "max_turns_per_run": max_turns,
        "subtitle_path": None,
        # Managed-Agents echo-back fields:
        "environment_id": environment_id,
        "ma_agent": ma_agent,
        "ma_resources": body.get("resources") or [],
        "created_at": now,
        "updated_at": now,
    }


@router.post("/sessions", status_code=201)
async def create_session(request: Request, body: dict[str, Any] = Body(...)) -> dict:
    store = request.app.state.store
    broker = request.app.state.broker
    record = await build_session_record(body, store)
    await store.create_session(record)

    runner = SessionRunner(
        session_id=record["session_id"],
        video_id=record["video_id"],
        model=record["model"],
        max_turns_per_run=record["max_turns_per_run"],
        backend=record["sandbox"]["backend"],
        limits=SandboxLimits(**record["sandbox"]["limits"]),
        store=store,
        broker=broker,
        mode=record.get("mode", "agent"),
        system=record.get("agent_system"),
    )
    request.app.state.runners[record["session_id"]] = runner

    async def _boot() -> None:
        try:
            # fast mode or self video analysis tool does not need a sandbox
            if not settings.self_video_analysis_tool or not record.get("mode") == "fast":
                await runner.start_sandbox()
            await store.update_session(record["session_id"], lambda r: {
                **r, "status": "ready",
                "sandbox": {**(r.get("sandbox") or {}), "id": runner.sandbox_id, "status": "ready"},
            })
        except Exception:
            pass

    asyncio.create_task(_boot(), name=f"sandbox-boot-{record["session_id"]}")
    return _session_resource(record)


@router.get("/sessions/{session_id}")
async def get_session(session_id: str, request: Request) -> dict:
    record = await request.app.state.store.get_session(session_id)
    if record is None:
        raise not_found("session", session_id)
    return _session_resource(record)


# --------------------------------------------------------------------------
# Events: send (input) + stream (output)
# --------------------------------------------------------------------------

def _extract_user_text(event: dict) -> Optional[str]:
    """Pull plain text out of a Managed Agents input event (a user.message)."""
    if not isinstance(event, dict):
        return None
    content = event.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                parts.append(block.get("text", ""))
            elif isinstance(block, str):
                parts.append(block)
        return "\n".join(p for p in parts if p).strip() or None
    # Some shapes nest under message: {"message": {"content": ...}}
    msg = event.get("message")
    if isinstance(msg, dict):
        return _extract_user_text(msg)
    return None


async def _get_or_rehydrate_runner(session_id: str, request: Request) -> Optional[SessionRunner]:
    """Return this worker's cached runner, or rebuild one from the store.

    A worker that didn't create the session (or that restarted) has no runner in
    memory. We reconstruct it from the persisted session record + `messages`, and
    reattach to the live sandbox (e2b reconnect by id; no-op for inprocess).
    Caller must already hold the ownership lease.
    """
    runner = request.app.state.runners.get(session_id)
    if runner is not None:
        return runner
    store = request.app.state.store
    record = await store.get_session(session_id)
    if record is None:
        return None
    messages = await store.get_messages(session_id)
    sandbox = record.get("sandbox") or {}
    runner = SessionRunner(
        session_id=session_id,
        video_id=record["video_id"],
        model=record["model"],
        max_turns_per_run=record["max_turns_per_run"],
        backend=sandbox.get("backend") or settings.sandbox_backend,
        limits=SandboxLimits(**(sandbox.get("limits") or {})),
        store=store,
        broker=request.app.state.broker,
        messages=messages or None,
        mode=record.get("mode", "agent"),
        system=record.get("agent_system"),
    )
    await runner.attach_sandbox(sandbox)
    request.app.state.runners[session_id] = runner
    return runner


@router.post("/sessions/{session_id}/events")
async def send_events(session_id: str, request: Request, body: dict[str, Any] = Body(...)) -> dict:
    store = request.app.state.store
    broker = request.app.state.broker
    if await store.get_session(session_id) is None:
        raise not_found("session", session_id)

    events = body.get("events")
    output_structure = body.get("output_structure")
    if not isinstance(events, list) or not events:
        raise bad_request("body.events must be a non-empty list")

    text = None
    for ev in events:
        text = _extract_user_text(ev) or text

    if not text:
        raise bad_request("no user text found in events[].content")

    # Claim run ownership across all workers. A held lease means a run is already
    # in flight (here or on another worker). The runner releases it when the run
    # completes.
    if not await broker.claim(session_id):
        raise conflict("a run is already in flight for this session")
    try:
        log.info(f"Sending user message for session {session_id}")
        runner = await _get_or_rehydrate_runner(session_id, request)
        if runner is None:
            raise not_found("session", session_id)
        record = await store.append_event(session_id, "user.message", {"content": text})
        await runner.submit_user_message(text, output_structure=output_structure)
    except SandboxNotReady:
        # Creator worker is still booting the sandbox; release and let the client retry.
        await broker.release(session_id)
        raise conflict("session sandbox is still starting; retry shortly")
    except Exception:
        await broker.release(session_id)
        raise
    return {"event_id": record["event_id"], "seq": record["seq"], "accepted_at": record["ts"]}


def _to_managed_events(ev: dict) -> list[dict]:
    """Map one Ambient runner event into zero or more Managed Agents session events.

    The Anthropic SDK's stream decoder only yields events whose SSE name (== the
    data `type` discriminator) is in its allowlist, and constructs each into a
    member of BetaManagedAgentsStreamSessionEvents. So we emit exactly those
    shapes:
        run.started            -> session.status_running
        agent.reasoning        -> agent.thinking (reasoning text attached)
        tool.scheduled         -> agent.tool_use
        tool.result            -> agent.tool_result
        tool.failed            -> agent.tool_result (is_error)
        span.model_request_end -> span.model_request_end (per-request token usage)
        error                  -> session.error
        run.completed          -> agent.message (the answer) + session.status_idle
        user.message           -> user.message
    Everything else (turn.*, tool.started/progress, per-token chunks,
    status_changed) is dropped — it has no SDK-visible counterpart.
    """
    etype = ev["type"]
    p = ev["payload"]
    ts = ev["ts"]

    if etype == "run.started":
        return [{"type": "session.status_running", "id": _new_id("evt"), "processed_at": ts}]

    if etype == "agent.reasoning":
        # `agent.thinking` is a progress signal in the SDK schema (no content
        # field), but the SDK's models allow extra fields, so we attach the
        # reasoning text as `content` for clients that want to render it.
        text = str(p.get("reasoning") or "")
        if not text:
            return []
        return [{
            "type": "agent.thinking",
            "id": _new_id("evt"),
            "content": [{"type": "text", "text": text}],
            "processed_at": ts,
        }]

    if etype == "span.model_request_end":
        usage = p.get("usage") or {}
        return [{
            "type": "span.model_request_end",
            "id": _new_id("evt"),
            "model_request_start_id": p.get("run_id") or _new_id("evt"),
            "is_error": False,
            "model_usage": {
                "input_tokens": int(usage.get("input_tokens") or 0),
                "output_tokens": int(usage.get("output_tokens") or 0),
                "cache_read_input_tokens": int(usage.get("cache_read_input_tokens") or 0),
                "cache_creation_input_tokens": int(usage.get("cache_creation_input_tokens") or 0),
            },
            # Extra (non-schema) fields for richer clients.
            "source": p.get("source"),
            "model": p.get("model"),
            "cost": p.get("cost"),
            "processed_at": ts,
        }]

    if etype == "error":
        return [{
            "type": "session.error",
            "id": _new_id("evt"),
            "error": {
                "type": "unknown_error",
                "message": str(p.get("message") or p.get("code") or "error"),
                "retry_status": {"type": "terminal"},
            },
            "processed_at": ts,
        }]

    if etype == "tool.scheduled":
        return [{
            "type": "agent.tool_use",
            "id": p.get("tool_use_id") or _new_id("evt"),
            "name": p.get("name"),
            "input": p.get("input") or {},
            "processed_at": ts,
        }]

    if etype == "tool.result":
        return [{
            "type": "agent.tool_result",
            "id": _new_id("evt"),
            "tool_use_id": p.get("tool_use_id"),
            "content": [{"type": "text", "text": str(p.get("analysis") or "")}],
            "is_error": False,
            "processed_at": ts,
        }]

    if etype == "tool.failed":
        return [{
            "type": "agent.tool_result",
            "id": _new_id("evt"),
            "tool_use_id": p.get("tool_use_id"),
            "content": [{"type": "text", "text": str((p.get("error") or {}).get("message", "tool failed"))}],
            "is_error": True,
            "processed_at": ts,
        }]

    if etype == "run.completed":
        out: list[dict] = []
        answer = p.get("answer")
        if answer:
            out.append({
                "type": "agent.message",
                "id": _new_id("evt"),
                "content": [{"type": "text", "text": answer}],
                "processed_at": ts,
            })
        stop = p.get("stop_reason")
        out.append({
            "type": "session.status_idle",
            "id": _new_id("evt"),
            "stop_reason": "end_turn" if stop in (None, "end_turn") else stop,
            # Extra (non-schema) field: cumulative session usage snapshot so the
            # client can render totals without a separate session fetch.
            "usage": p.get("usage") or {},
            "processed_at": ts,
        })
        return out

    if etype == "user.message":
        return [{
            "type": "user.message",
            "id": _new_id("evt"),
            "content": [{"type": "text", "text": str(p.get("content") or "")}],
            "processed_at": ts,
        }]

    if etype == "chat.completion.chunk":
        # Forward incremental reasoning + answer text so clients can render
        # token-by-token. (The official SDK ignores unknown event names; the demo
        # UI renders them.)
        content_delta = ""
        reasoning_delta = ""
        for ch in (p.get("choices") or []):
            d = ch.get("delta") or {}
            if isinstance(d.get("content"), str):
                content_delta += d["content"]
            r = d.get("reasoning") or d.get("reasoning_content")
            if isinstance(r, str):
                reasoning_delta += r
        out: list[dict] = []
        if reasoning_delta:
            out.append({"type": "agent.reasoning.delta", "id": _new_id("evt"),
                        "reasoning": reasoning_delta, "processed_at": ts})
        if content_delta:
            out.append({"type": "agent.message.delta", "id": _new_id("evt"),
                        "content": [{"type": "text", "text": content_delta}], "processed_at": ts})
        return out

    return []


@router.get("/sessions/{session_id}/events/stream")
async def stream_events(
    session_id: str,
    request: Request,
    after_seq: int = Query(0),
):
    store = request.app.state.store
    broker = request.app.state.broker
    record = await store.get_session(session_id)
    if record is None:
        raise not_found("session", session_id)

    async def gen():
        last_event_id = request.headers.get("last-event-id")
        replay_after = after_seq
        if last_event_id:
            try:
                replay_after = max(replay_after, int(last_event_id))
            except ValueError:
                pass

        # When the SDK opens a fresh stream (no cursor), scope the replay to the
        # latest run only — otherwise the replay loop hits an earlier run's
        # run.completed and returns before ever reaching the current run's events.
        #
        # Anchor on the latest `user.message` as well as `run.started`. The client
        # sends its message (persisted synchronously) and *then* opens the stream,
        # but the run task that emits `run.started` may not have run yet. Anchoring
        # only on run.started would race: we'd pick the *previous* run's start,
        # replay that whole run (and stop at its run.completed) instead of the new
        # one — surfacing prior tool calls/results again. The current run's
        # user.message always has the highest seq at stream-open, so it's the
        # reliable boundary.
        if not last_event_id and after_seq == 0:
            all_events = await store.list_events(session_id, after_seq=0, limit=10000)
            last_start = max(
                (e["seq"] for e in all_events
                 if e["type"] in ("run.started", "user.message")),
                default=0,
            )
            replay_after = max(replay_after, last_start - 1)

        def emit(ev: dict):
            out = []
            for me in _to_managed_events(ev):
                out.append({"event": me["type"], "id": str(ev["seq"]), "data": json.dumps(me)})
            return out

        # Subscribe to the Redis fan-out *before* the catch-up replay so no live
        # event is lost in the gap; the seq dedupe below drops any overlap. This
        # is what lets a stream on one worker tail a run executing on another.
        sub = await broker.subscribe(session_id)
        try:
            seen_max = replay_after
            # Replay from the durable log, closing on run end (run-scoped).
            for ev in await store.list_events(session_id, after_seq=replay_after, limit=10000):
                seen_max = max(seen_max, ev["seq"])
                for frame in emit(ev):
                    yield frame
                if ev["type"] == "run.completed":
                    return

            # Live tail from Redis pub/sub.
            while True:
                if await request.is_disconnected():
                    return
                ev = await sub.get(timeout=15)
                if ev is None:
                    continue
                if ev.get("__close__"):
                    return
                if ev["seq"] <= seen_max:
                    continue
                seen_max = ev["seq"]
                for frame in emit(ev):
                    yield frame
                if ev["type"] == "run.completed":
                    return
        finally:
            await sub.close()

    return EventSourceResponse(gen())


@router.get("/sessions/{session_id}/events")
async def list_events(
    session_id: str,
    request: Request,
    after_seq: int = Query(0),
    limit: int = Query(1000),
):
    store = request.app.state.store
    if await store.get_session(session_id) is None:
        raise not_found("session", session_id)
    raw = await store.list_events(session_id, after_seq=after_seq, limit=limit)
    data = []
    for ev in raw:
        data.extend(_to_managed_events(ev))
    return {"data": data, "has_more": False}

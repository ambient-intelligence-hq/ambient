# Ambient Agent API — v1 spec

## 0. Framework, conventions, auth

- **FastAPI** + **`sse-starlette`** for SSE. Pydantic models for all request/response shapes.
- Mount under `/v1`. All JSON is snake_case.
- **Persistence**: SQLite (`settings.sqlite_db_path`). New tables: `sessions`, `events`. Trajectories continue to be persisted as JSON files in `ambient/trajectories/` and are referenced by `session_id`.
- **Sandbox**: v1 uses an in-process multiprocess worker per session (see §4). The API is intentionally agnostic to the backend so it can be swapped for a hosted sandbox (Modal, E2B, Daytona, custom microVM) without protocol changes.
- **Auth**: `Authorization: Bearer $TOKEN` validated against `settings.api_key`. 401 on miss.
- **Errors** (non-stream): `{"error": {"code": "...", "message": "...", "details": {...}}}` with appropriate HTTP status. SSE errors are emitted as an `error` event and the stream closes with HTTP 200.

## 1. Concepts

- **Session**: a stateful, server-side conversation bound to a single `video_id`. Holds the full message history, the sandbox handle, cumulative usage, and a status. Long-lived; survives client disconnects.
- **Run**: one user message triggers a *run*. A run is composed of one or more turns and terminates when the agent stops (`end_turn`), hits `max_turns`, is cancelled, or errors. Concurrent runs within a single session are not allowed — sessions are single-threaded by design.
- **Turn**: one round-trip to the LLM. A run that triggers tool use produces multiple turns (`tool_use → tool_results → next turn`). Numbered from 1 within a run.
- **Event**: the unit of communication on `/v1/sessions/{id}/events`. Events flow both ways: inputs (`user.message`, `user.tool_confirmation`) are POSTed by the client; outputs (everything else) are streamed/listed.
- **Sandbox**: an isolated execution context attached to a session that runs the agent's tools. Asynchronously provisioned at session create. Identified by an opaque handle; the API does not assume a specific backend.
- **Permission policy**: rule that decides whether a tool call runs immediately or pauses for human approval. v1 default: `always_allow`.

### 1.1 Session lifecycle

```
created → starting → ready → running → ready → ... → terminated
                       ↓        ↓                       ↑
                     failed   failed ──────────────────┘
```

- `created` is a synchronous response to `POST /v1/sessions`; the session row exists, sandbox boot is in flight.
- `starting` → `ready` happens when the sandbox handle is healthy.
- `running` is set while a run is in flight; flips back to `ready` on run completion.
- `failed` is terminal-ish: sandbox boot failed or sandbox crashed unrecoverably. Client may `DELETE` and recreate.
- `terminated` is set by `DELETE /v1/sessions/{id}` or by idle TTL.

Status transitions are surfaced as `session.status_changed` events (§3.3).

## 2. Endpoints

| Method | Path | Purpose |
| --- | --- | --- |
| `POST` | `/v1/sessions` | Create session + boot sandbox. |
| `GET` | `/v1/sessions` | List sessions (paginated). |
| `GET` | `/v1/sessions/{id}` | Session metadata + status + usage. |
| `DELETE` | `/v1/sessions/{id}` | Terminate sandbox + persist final trajectory. |
| `POST` | `/v1/sessions/{id}/events` | Submit `user.message` or `user.tool_confirmation`. |
| `GET` | `/v1/sessions/{id}/events` | List past events (paginated) or stream live (`?stream=true`). |
| `POST` | `/v1/sessions/{id}/cancel` | Cancel the in-flight run. |
| `POST` | `/v1/sessions/{id}/compact` | Compact context (drop old clips/frames, summarize). |
| `GET` | `/v1/sessions/{id}/trajectory` | Fetch persisted trajectory JSON. |
| `GET` | `/v1/healthz` | Liveness. |

### 2.1 `POST /v1/sessions`

**Request body**
```json
{
  "video_id": "string",
  "model": "string",
  "subtitle_path": "string",
  "max_turns_per_run": 5,
  "permission_policy": {"type": "always_allow"},
  "sandbox": {
    "backend": "local",
    "limits": {"cpu_seconds": 600, "memory_mb": 4096, "wall_seconds": 1800}
  },
  "quota": {"max_tokens": 200000, "max_wall_seconds": 7200},
  "metadata": {"trace_id": "..."},
  "title": "string"
}
```

All fields except `video_id` are optional. Defaults:
- `model` → `settings.agent_model`.
- `max_turns_per_run` → 5.
- `permission_policy` → `{"type": "always_allow"}`.
- `sandbox.backend` → `settings.sandbox_backend` (defaults to `"local"`).
- `sandbox.limits` → backend defaults; see §4.
- `quota` → unlimited.
- `title` → first five words of the first user message; backfilled lazily.

**Response** (HTTP 201)
```json
{
  "session_id": "ses_...",
  "status": "created",
  "video_id": "...",
  "model": "...",
  "sandbox": {
    "backend": "local",
    "id": "sb_...",
    "limits": { ... }
  },
  "created_at": "2026-05-24T10:00:00Z"
}
```

The sandbox boots asynchronously. Clients that need to wait subscribe to the event stream and wait for `session.status_changed → ready`. `POST /v1/sessions/{id}/events` calls submitted before `ready` are queued and processed once the sandbox is up.

### 2.2 `GET /v1/sessions/{id}`

Returns the full `Session` resource:
```json
{
  "session_id": "ses_...",
  "status": "ready | starting | running | failed | terminated",
  "video_id": "...",
  "model": "...",
  "title": "...",
  "metadata": { ... },
  "permission_policy": { ... },
  "sandbox": {
    "backend": "local",
    "id": "sb_...",
    "status": "ready | starting | failed | terminated",
    "boot_ms": 87,
    "limits": { ... },
    "region": null
  },
  "quota": { ... },
  "usage": {
    "input_tokens": 0,
    "output_tokens": 0,
    "cache_read_input_tokens": 0,
    "cache_creation_input_tokens": 0,
    "turns": 0,
    "runs": 0
  },
  "stats": {
    "created_at": "...",
    "last_event_at": "...",
    "active_seconds": 0,
    "duration_seconds": 0
  },
  "trajectory_path": "ambient/trajectories/FILE.json"
}
```

### 2.3 `DELETE /v1/sessions/{id}`

Terminates the sandbox, drains pending events, flushes the trajectory file, marks the session `terminated`. Idempotent. Returns `{"session_id": "...", "status": "terminated"}`.

### 2.4 `POST /v1/sessions/{id}/events` — submit input

**Body** is one of:

```json
{"type": "user.message", "content": "string", "model": "string?", "metadata": {...}?}
```
```json
{"type": "user.tool_confirmation", "tool_use_id": "...", "result": "allow | deny"}
```

Semantics:
- `user.message` starts a new run. 409 if a run is already in flight. The new event is appended to the session log with a monotonically-increasing `seq` and an `event_id`.
- `user.tool_confirmation` is only meaningful when the current run is paused on a `tool.requires_approval` event (see §5). 409 otherwise.
- `model` on `user.message` overrides the session default for *this run only*. Useful for cheap follow-ups.
- Returns `{event_id, seq, accepted_at}`.

### 2.5 `GET /v1/sessions/{id}/events`

Two modes by query string:

- **Paginated history** (default): `?after_seq=&limit=`. Returns past events as JSON array. Use for reconstructing UI state after a hard restart.
- **Live stream**: `?stream=true&after_seq=`. Returns `text/event-stream`. Events with `seq > after_seq` are replayed first, then new events stream as they happen. Each SSE event sets `id:` to the event's `seq`, so reconnect via `Last-Event-ID` works without query-string fiddling.

The very first event on a fresh stream is always the current `session.status_changed` so the client can sync.

### 2.6 `POST /v1/sessions/{id}/cancel`

Cancels the in-flight run. Body empty. Response: `{"session_id": "...", "status": "cancelled | not_running"}` (HTTP 200 in both cases — idempotent).

Effect:
- Cancels the asyncio task driving the run.
- Aborts any in-flight tool call via the sandbox handle (`sandbox.cancel(tool_call_id)`).
- Emits a final `turn.end` with `finish_reason: "cancelled"` then `run.completed` with `stop_reason: "cancelled"`.
- Partial trajectory is persisted.
- Session returns to `ready`.

Client disconnect on the live stream does **not** cancel the run — the run continues server-side and the client can reconnect with `Last-Event-ID` to catch up. Cancellation is always explicit.

### 2.7 `POST /v1/sessions/{id}/compact`

**Body**: `{"strategy": "drop_old_assets" | "summarize", "target_tokens": 50000}` (all optional).

Server applies the strategy to the session's stored messages (e.g. drop `image_url` / `video_url` blocks beyond a recent window, replace with a textual summary). Emits a `session.compacted` event with the before/after token counts. 409 if a run is in flight.

### 2.8 `GET /v1/sessions/{id}/trajectory`

Returns the persisted trajectory JSON for the session. 404 if not yet flushed. Useful for offline debugging and for replaying a finished session.

## 3. Event schema

The stream is two channels multiplexed by SSE `event:` type:

1. **`chat.completion.chunk`** — the upstream OpenAI streaming chunk, relayed verbatim. One per chunk per turn.
2. **Server-emitted events** — `session.*`, `run.*`, `turn.*`, `tool.*`, `usage.delta`, `error`. These wrap the chunks with agent-loop framing and surface tool/sandbox state that has no OpenAI equivalent.

`chat.completion.chunk` payloads are not extended or wrapped. A client that already speaks OpenAI streaming reuses its parser for assistant content + tool calls; only the framing events are new.

Every server-emitted event carries `{session_id, seq, event_id, ts}`. The `seq` is the monotonically-increasing session-wide counter used for replay (`Last-Event-ID`).

### 3.1 `chat.completion.chunk`

Verbatim OpenAI chunk:
```
event: chat.completion.chunk
id: 4217
data: {"id":"chatcmpl-…","object":"chat.completion.chunk","created":…,"model":"…","choices":[{"index":0,"delta":{"content":"Hel"},"finish_reason":null}]}
```

Notes:
- Server **does not forward upstream `data: [DONE]`** — `turn.end` plays that role per turn.
- Tool calls stream through standard `delta.tool_calls[].function.{name,arguments}`. Server accumulates them, runs the tool after `finish_reason == "tool_calls"`, emits `tool.*` events.
- Reasoning streams as `delta.reasoning` (string deltas), matching `_send_request`'s current handling. Absent if the provider doesn't emit it.
- `id` field inside the chunk is the upstream chat-completion id; the SSE `id:` line is the session `seq`.

### 3.2 `session.*`

| Event | Data |
| --- | --- |
| `session.status_changed` | `{status, previous_status, sandbox: {status, boot_ms?, region?}}` |
| `session.compacted` | `{strategy, tokens_before, tokens_after}` |

### 3.3 `run.*`

| Event | Data | Fires |
| --- | --- | --- |
| `run.started` | `{run_id, triggered_by_event_id, model}` | When a `user.message` triggers the agent loop. |
| `run.completed` | `{run_id, stop_reason, usage, answer}` | When the loop ends. `stop_reason` ∈ `end_turn` \| `max_turns` \| `cancelled` \| `error`. `answer` is the concatenated text of the final assistant message. |

### 3.4 `turn.*`

| Event | Data | Fires |
| --- | --- | --- |
| `turn.start` | `{run_id, turn}` | Before each turn's chunks begin. Clients bucket chunks by turn from here. |
| `turn.end` | `{run_id, turn, finish_reason, usage}` | After upstream `[DONE]`. `usage` is per-turn token counts when the upstream provides them. |

### 3.5 `tool.*`

Sandboxed tool execution has its own lifecycle, separate from the chunks that surfaced the call.

| Event | Data | Fires |
| --- | --- | --- |
| `tool.requires_approval` | `{run_id, turn, tool_use_id, name, input}` | Only when `permission_policy` matches `always_ask`. Run pauses until `user.tool_confirmation` arrives. |
| `tool.scheduled` | `{run_id, turn, tool_use_id, name, input}` | Tool call dispatched to the sandbox. |
| `tool.started` | `{run_id, turn, tool_use_id, started_at}` | Sandbox confirmed it picked up the call. |
| `tool.progress` | `{run_id, turn, tool_use_id, progress: {phase, percent?, message?}}` | Optional, for long-running tools (e.g. ffmpeg frame extraction). Tools opt in. |
| `tool.result` | `{run_id, turn, tool_use_id, name, analysis, attachments, elapsed_ms}` | Sandbox returned. `attachments` are S3 URLs the tool produced (`image_url` / `video_url`). |
| `tool.failed` | `{run_id, turn, tool_use_id, name, error: {code, message}, elapsed_ms}` | Sandbox raised. Treated as `tool_result` with an error payload upstream. |

### 3.6 `usage.delta` and `error`

| Event | Data |
| --- | --- |
| `usage.delta` | `{usage: {input_tokens, output_tokens, cache_read_input_tokens, cache_creation_input_tokens}}` — emitted at most once per turn when the provider returns a `usage` block. Cumulative session usage is in `GET /v1/sessions/{id}`. |
| `error` | `{code, message, details?, fatal: bool}` — if `fatal:true`, session transitions to `failed`. |

### 3.7 Event ordering

For a run that ends in tool use:
```
run.started
turn.start
chat.completion.chunk × N    # assistant text + tool_calls deltas
turn.end                     # finish_reason: tool_calls
tool.scheduled × M           # one per tool call, parallel
tool.started × M
tool.progress × ?            # optional, interleaved
tool.result × M              # one per tool call, in completion order
turn.start                   # next turn
…
turn.end                     # finish_reason: stop
run.completed                # stop_reason: end_turn
```

When `permission_policy` requires approval, between `turn.end` and `tool.scheduled`:
```
turn.end (finish_reason: tool_calls)
tool.requires_approval × M
  ← (POST /events: user.tool_confirmation for each)
tool.scheduled × M (only for approved)
tool.failed × ? (for denied, with code "denied_by_user")
```

### 3.8 Reconnect, replay, heartbeats

- Every event sets the SSE `id:` to the session `seq`. Clients reconnect with `Last-Event-ID: SEQ` and the server replays missed events from the SQLite event log before resuming live tailing.
- `: keep-alive\n\n` comments every 15s during quiet periods.

## 4. Tool execution and sandbox model

This section is the most important one for the cloud-upgrade path. The HTTP/event surface above does not depend on the sandbox implementation. The server interacts with the sandbox through a single internal interface; backends are swappable.

### 4.1 v1 backend: `local` (multiprocess)

- One `multiprocessing.Process` per session, started at `POST /v1/sessions`.
- IPC over an asyncio-wrapped duplex pipe (length-prefixed JSON frames).
- Per-session `TMPDIR=/tmp/ambient/sessions/SESSION_ID/` set on spawn; tools that need scratch space respect `tempfile.gettempdir()`.
- Resource limits applied in the child via `resource.setrlimit`:
  - `RLIMIT_AS` from `sandbox.limits.memory_mb`
  - `RLIMIT_CPU` from `sandbox.limits.cpu_seconds`
  - Wall-clock cap enforced by the server with `asyncio.wait_for` (not in-child).
- Boot time: ~50–100ms on macOS/Linux for a forked worker that lazy-imports the tools.
- Sandbox identity: `sb_UUID`. `region: null`.

### 4.2 `SandboxHandle` interface (internal, language-level)

The server only knows about this interface. Backends implement it.

```python
class SandboxHandle(Protocol):
    backend: Literal["local", "modal", "e2b", ...]
    id: str
    region: str | None

    async def start(self, *, video_id: str, limits: SandboxLimits, env: dict[str, str]) -> SandboxStartInfo: ...
    async def call_tool(
        self,
        tool_use_id: str,
        name: str,
        input: dict,
    ) -> AsyncIterator[ToolEvent]:  # yields started / progress / result / failed
        ...
    async def cancel(self, tool_use_id: str) -> None: ...
    async def attach_resource(self, resource: Resource) -> ResourceHandle: ...
    async def detach_resource(self, handle: ResourceHandle) -> None: ...
    async def health(self) -> SandboxHealth: ...
    async def terminate(self) -> None: ...
```

Constraints that make this swappable:

- **All inputs and outputs are JSON-serializable.** No in-process object passing. The trajectory's `messages` are passed as plain dicts; tool inputs/outputs are dicts; attachments are URLs (presigned S3) or `Resource` references.
- **Tools never read or write the server's local filesystem directly.** Anything the tool needs is either:
  - Fetched from an S3 URL (already the pattern in `ambient/tools/`), or
  - Attached as a `Resource` (see §4.4) that the sandbox is responsible for materializing.
- **Cancellation is cooperative, propagated by `tool_use_id`.** A cloud backend with a managed-job model can map this onto its own cancel API.
- **Sandbox state is not assumed to outlive the handle.** Caches inside the sandbox are best-effort; reboots are permitted.
- **No reliance on shared memory, signals, or `fork()` semantics outside the local backend.** The local backend may use them internally but the interface does not expose them.
- **Health checks are explicit.** The server polls `health()` between runs; an unhealthy handle is torn down and rebuilt, surfacing `session.status_changed → failed → starting → ready`.

### 4.3 Tool RPC contract

Each tool registered in `TOOL_REGISTRY` exposes:
- `name: str`
- `input_schema: dict` (already present via Pydantic models)
- Async callable `tool(input: dict) -> ToolResult` where:
  ```python
  class ToolResult(TypedDict):
      analysis: str
      attachments: list[Attachment]  # {type: "image_url"|"video_url", url, metadata?}
      progress_events: list[ProgressEvent]  # optional, emitted during call
  ```
  This replaces the current `(analysis, user_message_contents)` tuple in [ambient/agent.py](ambient/agent.py) `_run_tool_calls`. The new shape is the same data, just labelled.

The local backend imports the registry and dispatches in-process inside the child. A cloud backend ships the tool registry as part of its agent image, or exposes tools as managed primitives — either way, only the names + schemas cross the wire.

### 4.4 Resources

Some tool inputs are large (videos, subtitle files). To avoid shipping bytes through the IPC channel and to give cloud sandboxes a clean way to fetch on their side:

```python
class Resource(TypedDict):
    type: "video" | "subtitle" | "frame" | "transcript"
    id: str          # video_id or arbitrary
    source: dict     # {kind: "s3", url: "..."} or {kind: "local", path: "..."}
    metadata: dict
```

- The server `attach_resource()`s resources at session start (the current video; subtitle if provided).
- The sandbox is responsible for materializing them on its side (download from S3, mount local path, etc.).
- Tools take `Resource` references as input, not raw paths. The local backend dereferences to a path; a cloud backend dereferences to whatever its FS conventions are.
- This is the **one place** in §4 where the v1 local backend differs from a cloud backend operationally — but the *interface* is identical.

### 4.5 Backend selection and the upgrade path

- Selected per-session via `sandbox.backend` in the create request, defaulting to `settings.sandbox_backend` env var (`local` in v1).
- Adding a new backend means implementing `SandboxHandle` + registering in a backend table. No HTTP surface changes.
- Upgrade-path safeguards baked into v1:
  - **Asynchronous boot** is in the protocol from day one (`session.status_changed: starting → ready`). Cloud boots take seconds; clients already wait.
  - **Sandbox identity / region / boot_ms** are first-class fields in `GET /v1/sessions/{id}`. Cloud backends populate `region` and produce non-trivial `boot_ms`.
  - **Per-tool cancellation** is required, not a "nice to have". Local backend cancels via task cancel; cloud backends use their job cancel API.
  - **Resource attachment** is the abstraction that avoids leaking local-filesystem assumptions into the tool layer.
  - **Limits expressed declaratively** (`cpu_seconds`, `memory_mb`, `wall_seconds`) — backend-agnostic. Local maps to `setrlimit`; cloud maps to its quota knobs.
  - **All tool I/O is JSON.** No pickled objects, no shared memory, no `multiprocessing.Queue`-specific shapes leaking into the contract.
  - **Sandbox failures surface as session-level events**, not exceptions in random places. A cloud backend reporting "node lost" looks like any other `session.status_changed → failed`.

## 5. Permission policy

### 5.1 v1 default: `always_allow`

```json
{"type": "always_allow"}
```

Tools dispatch as soon as the LLM emits them. No `tool.requires_approval` events. This is the path the CLI/server-to-server callers use.

### 5.2 Other policies (defined now, enforced when needed)

```json
{"type": "always_ask", "tools": ["search_clip", "focus_clip"]}
```
```json
{"type": "deny", "tools": ["..."]}
```

When `always_ask` matches a tool call:
- Server emits `tool.requires_approval`.
- Run task awaits a corresponding `user.tool_confirmation` event posted by the client.
- On `allow` → `tool.scheduled` follows. On `deny` → `tool.failed` with `code: "denied_by_user"`.
- The run task has no internal timeout for approval; rely on session idle TTL.

`deny` synthesizes a `tool.failed` with `code: "denied_by_policy"` without emitting `requires_approval`.

The event schema for `tool.requires_approval` / `user.tool_confirmation` is fixed in v1 so future clients don't need a protocol bump to opt in.

## 6. Cancellation, compaction, TTL

- **Cancel** (§2.6): per-run. Session stays alive.
- **Compact** (§2.7): trims the stored `messages` of stale visual content. Existing in-process retention (`_retain_last_n_by_type`) handles the per-turn budget; `compact` is for cross-turn cleanup.
- **Idle TTL**: a session with no `user.message` activity for `settings.session_idle_ttl_seconds` (default 30 min) is terminated. Configurable per session via metadata in a future revision.
- **Hard cap**: `settings.session_hard_ttl_seconds` (default 24h) regardless of activity.
- **Quota enforcement**: per-session `max_tokens` and `max_wall_seconds`. On breach: emit `error {code: "quota_exceeded", fatal: true}`, terminate.

## 7. Usage and quotas

- `usage` on the session is cumulative across runs and turns. Updated as `usage.delta` events arrive.
- `quota` is set at session create and immutable. Quota fields:
  ```json
  {"max_tokens": int?, "max_wall_seconds": int?, "max_cost_usd": float?}
  ```
- `max_cost_usd` is reserved — v1 has no per-model cost table.

## 8. Implementation notes

What changes in [ambient/agent.py](ambient/agent.py) and what's new in [ambient/server/](ambient/server/):

1. **`_send_request` becomes a streaming async generator.** Switch to `aiohttp` (already in `llm.py`) with `stream=true`. Parse upstream SSE line-by-line, skip `[DONE]`, yield parsed chunk dicts. Server SSE handler re-serializes each as `event: chat.completion.chunk`.
2. **`run_agent` is replaced by a `SessionRunner` class.** Long-lived per session. Methods: `start()`, `submit(event)`, `cancel_run()`, `terminate()`. Drives the agent loop in response to inputs; yields events on an internal `asyncio.Queue`.
3. **Tool dispatch goes through the `SandboxHandle`.** `ambient/agent.py:execute_tool` is replaced by `sandbox.call_tool(tool_use_id, name, input)`. The current in-process `TOOL_REGISTRY` lives inside the local backend's child process.
4. **Local sandbox lives in `ambient/server/sandbox/local.py`** with a sibling `interface.py` defining the `SandboxHandle` protocol. Future backends slot in as `sandbox/modal.py`, `sandbox/e2b.py`, etc.
5. **Persistence in `ambient/server/store.py`.** SQLite tables:
   - `sessions(id, status, video_id, model, ...)`
   - `events(id, session_id, seq, type, payload_json, ts)`
   - `runs(id, session_id, run_id, status, stop_reason, started_at, ended_at)`
   `events` is the source of truth for replay; trajectory JSON files are materialized snapshots on session close.
6. **FastAPI app in `ambient/server/app.py`.** Routes thin; business logic in `SessionRunner` and the store. SSE handled with `sse-starlette`'s `EventSourceResponse`.
7. **Stop using `print` in agent code.** Wire to `logging`; let the event stream be the user-facing source of truth.

## 9. Open questions / non-goals for v1

- **Multi-user / auth granularity.** Single static bearer token. Per-user keys, ACLs on sessions: out of scope.
- **Rate limiting.** Out of scope; revisit when there's more than one caller.
- **Server-side history mutation by the client.** Clients cannot edit prior events. To branch, create a new session.
- **Cross-session memory.** No. Each session is independent.
- **`max_cost_usd` quota enforcement.** Field reserved; no cost table yet.
- **Multi-modal direct input from the caller.** Only `video_id` + `question`. Direct upload is a separate endpoint.
- **Resumable streams across server restarts.** SQLite event log makes this *possible*; v1 ships only in-memory live tailing plus SQLite-backed replay.
- **Hosted sandbox backends.** Interface is ready (§4.2); implementations are follow-up work.

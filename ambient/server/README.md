# Ambient Agent API

A stateful, streaming agent API that ties a conversation session to a video, runs the video-analysis tools, and streams all activity (status, tool lifecycle, the final answer) as events.

The server exposes **two interfaces over the same engine**:

| Interface | Prefix | Auth | Use it from |
|---|---|---|---|
| **Anthropic Managed Agents protocol** | `/v1` | `x-api-key` (or `Authorization: Bearer`) | the official `anthropic` Python SDK (`client.beta.agents` / `client.beta.sessions`) |
| **Native Ambient API** | `/ambient/v1` | `Authorization: Bearer` | curl / Postman / any HTTP client |

Both drive the same `SessionRunner` + in-process tools (with media ops optionally offloaded to an E2B sandbox). The Managed Agents routes let the Anthropic SDK talk to our video agent unmodified; the native routes are the lower-level, video-first surface.

## Starting the server

```bash
uv run python -m ambient.server
# → http://127.0.0.1:8080
```

Default auth token is `dev-token` (override with `AMBIENT_API_KEY`). Health check (no auth):

```bash
curl http://127.0.0.1:8080/v1/healthz   # → {"status":"ok"}
```

---

# 1. Using the Anthropic SDK (Managed Agents protocol)

The server implements enough of the Anthropic Managed Agents protocol that the real SDK drives the video agent end-to-end. Point the client's `base_url` at the server and use any string as the API key (it's checked against `AMBIENT_API_KEY`).

```bash
pip install "anthropic>=0.105"
```

```python
import time
from anthropic import Anthropic

client = Anthropic(base_url="http://127.0.0.1:8080", api_key="dev-token")

# 1. Agent — a reusable template. Only `model` is really used by the video
#    engine today (system/tools are stored and echoed back).
agent = client.beta.agents.create(
    name="Video Analyst",
    model="openai/gpt-5.5",
    system="Analyze videos and answer with citations.",
    tools=[{"type": "agent_toolset_20260401"}],
)

# 2. Environment — a stub locally (the sandbox needs no container config).
env = client.beta.environments.create(name="local-video")

# 3. Session — binds the agent to a run context. The VIDEO is passed here,
#    via metadata.video_id (this is the Ambient-specific bit).
session = client.beta.sessions.create(
    agent=agent.id,
    environment_id=env.id,
    metadata={"video_id": "Seattle_bad_driver_accident"},  # required
    title="Seattle accident Q&A",
)

# 4. Wait for the sandbox to boot (status: pending → idle).
for _ in range(40):
    if client.beta.sessions.retrieve(session.id).status == "idle":
        break
    time.sleep(1)

# 5. Send a question (events.send).
client.beta.sessions.events.send(session.id, events=[{
    "type": "user.message",
    "content": "How long after the accident did the police car arrive?",
}])

# 6. Stream the run (events.stream). The stream is run-scoped: it closes when
#    the run idles.
answer = ""
with client.beta.sessions.events.stream(session.id) as stream:
    for ev in stream:
        if ev.type == "agent.tool_use":
            print("tool:", ev.name)
        elif ev.type == "agent.message":
            answer = "".join(b.text for b in ev.content)
        elif ev.type == "session.status_idle":
            print("done:", ev.stop_reason)
print(answer)
```

### Running the media ops in E2B

The Python above is **identical** whichever backend you use — the Managed Agents
protocol has no sandbox field, so the backend is chosen **server-side**. To make
the booted session run frame/clip extraction in an E2B sandbox (the LLM call still
runs on the host), start the server with:

```bash
export SANDBOX_BACKEND=e2b
export E2B_API_KEY=e2b_xxx
export E2B_TEMPLATE=video-analysis-v1        # a built media template (see note)
# S3 must be configured AND the source video present at
#   s3://$S3_BUCKET/$S3_VIDEO_BASE_KEY/<video_id>.mp4
export S3_BUCKET=...  S3_ENDPOINT=...  AWS_ACCESS_KEY_ID=...  AWS_SECRET_ACCESS_KEY=...
uv run python -m ambient.server
```

Then run the SDK snippet **unchanged**. The session's `sandbox.backend` comes back
as `e2b`, and tool results carry S3 presigned URLs instead of local files. Build
the template once with
`python ambient/sandboxes/e2b/video-analysis-v1/build_prod.py` (it produces the
`video-analysis-v1` template).

> The **native** API can pick the backend *per session* instead, without an env
> var — pass `sandbox.backend` on create (see [§3](#3-native-ambient-api-ambientv1)).

### Where the video comes from

A `video_id` identifies a video the server already has. Three ways to supply it:

**a) Upload via the Files API, then mount it as a session resource** (SDK-native):

```python
with open("accident.mp4", "rb") as f:
    file = client.beta.files.upload(file=("accident.mp4", f, "video/mp4"))

session = client.beta.sessions.create(
    agent=agent.id, environment_id=env.id,
    resources=[{"type": "file", "file_id": file.id}],   # mounts the uploaded video
)
```

**b) Pass an existing id in session `metadata`:**

```python
client.beta.sessions.create(..., metadata={"video_id": "<id>"})
```

**c)** A resource carrying `video_id` / `id` / `name` also resolves.

If none is present, `sessions.create` returns `400`. Uploading stores the video
in the server's video folder (so the tools can read it) **and** pushes a copy to
R2 (best-effort); the returned file `id` is the `video_id`.

### Turn budget

The protocol has no per-run turn field, so the video engine defaults to **20**
turns (it typically needs ~6). Override per session:

```python
client.beta.sessions.create(..., metadata={"video_id": "...", "max_turns_per_run": "12"})
```

### Event mapping (engine → SDK)

The SDK's stream decoder only yields events whose name is in its allowlist, so
the engine's native events are translated into Managed Agents events:

| Ambient engine event | SDK stream event | Notes |
|---|---|---|
| `run.started` | `session.status_running` | run began |
| `tool.scheduled` | `agent.tool_use` | `name`, `input` |
| `tool.result` | `agent.tool_result` | `content` = analysis text |
| `tool.failed` | `agent.tool_result` (`is_error: true`) | |
| `run.completed` | `agent.message` + `session.status_idle` | `agent.message.content[].text` is the answer; `stop_reason` on idle |
| `user.message` | `user.message` | replayed on connect |
| `turn.*`, `tool.started/progress`, `chat.completion.chunk`, `session.status_changed` | *(dropped)* | no SDK-visible counterpart |

> The final answer arrives as a single `agent.message` (not token-by-token text deltas). Tool activity *does* stream live as `agent.tool_use` / `agent.tool_result`.

### Managed Agents HTTP endpoints (raw)

If you're not using the SDK, the same routes are plain HTTP. Auth header is
`x-api-key: dev-token` (bearer also accepted). The SDK appends `?beta=true`; the
server ignores it.

```bash
# Create agent
curl -X POST http://127.0.0.1:8080/v1/agents \
  -H "x-api-key: dev-token" -H "Content-Type: application/json" \
  -d '{"name":"Video Analyst","model":"openai/gpt-5.5"}'

# Create environment
curl -X POST http://127.0.0.1:8080/v1/environments \
  -H "x-api-key: dev-token" -H "Content-Type: application/json" \
  -d '{"name":"local-video"}'

# Create session (note metadata.video_id)
curl -X POST http://127.0.0.1:8080/v1/sessions \
  -H "x-api-key: dev-token" -H "Content-Type: application/json" \
  -d '{"agent":"agt_xxx","environment_id":"env_xxx","metadata":{"video_id":"Seattle_bad_driver_accident"}}'

# Send a user message (events.send — note the {"events":[...]} envelope)
curl -X POST http://127.0.0.1:8080/v1/sessions/ses_xxx/events \
  -H "x-api-key: dev-token" -H "Content-Type: application/json" \
  -d '{"events":[{"type":"user.message","content":"How long until police arrived?"}]}'

# Stream events (events.stream — distinct /events/stream path, not ?stream=true)
curl -N "http://127.0.0.1:8080/v1/sessions/ses_xxx/events/stream?after_seq=0" \
  -H "x-api-key: dev-token" -H "Accept: text/event-stream"
```

| Method | Path | Purpose |
|---|---|---|
| `POST` | `/v1/agents` | create agent |
| `GET` | `/v1/agents/{id}` · `/v1/agents` | retrieve · list |
| `POST` | `/v1/environments` | create environment (stub) |
| `GET` | `/v1/environments/{id}` | retrieve |
| `POST` | `/v1/files` | upload a video → file id (`client.beta.files.upload`) |
| `GET` | `/v1/files/{id}` · `/v1/files` | retrieve metadata · list |
| `DELETE` | `/v1/files/{id}` | delete (removes local cache) |
| `POST` | `/v1/sessions` | create session (`agent` + `environment_id` + `metadata.video_id` or file resource) |
| `GET` | `/v1/sessions/{id}` | retrieve |
| `POST` | `/v1/sessions/{id}/events` | send events (`{"events":[...]}`) |
| `GET` | `/v1/sessions/{id}/events/stream` | SSE stream (run-scoped) |
| `GET` | `/v1/sessions/{id}/events` | list mapped events |

### Endpoint reference

All routes take `x-api-key: dev-token` (bearer also works) and ignore the SDK's
`?beta=true`. Responses are JSON-serializable and read leniently by the SDK
(unset fields come back `null`).

---

#### `POST /v1/agents` — Create agent

Request:

```json
{
  "name": "Video Analyst",
  "model": "openai/gpt-5.5",
  "system": "Analyze videos.",
  "tools": [{ "type": "agent_toolset_20260401" }]
}
```

| Field | Type | Required | Notes |
|---|---|---|---|
| `model` | string \| object | ✓ | Model id, or `{ "id": "...", "speed": "..." }` |
| `name` | string | | Label |
| `system` | string | | Stored/echoed; the engine uses its own video prompt |
| `tools` / `skills` / `mcp_servers` | array | | Stored/echoed; engine runs its own video tools |
| `description`, `metadata` | string / object | | |

Response — `200`:

```json
{
  "type": "agent",
  "id": "agt_0bf5eb740eb8460b",
  "version": 1,
  "name": "Video Analyst",
  "description": null,
  "model": { "id": "openai/gpt-5.5", "speed": null },
  "system": "Analyze videos.",
  "tools": [{ "type": "agent_toolset_20260401" }],
  "skills": [],
  "mcp_servers": [],
  "metadata": {},
  "multiagent": null,
  "created_at": "2026-06-05T10:28:51.697Z",
  "updated_at": "2026-06-05T10:28:51.697Z"
}
```

#### `GET /v1/agents/{agent_id}` · `GET /v1/agents`

Retrieve returns the agent object above. List returns
`{ "data": [ ...agents... ], "has_more": false }`.

---

#### `POST /v1/environments` — Create environment (stub)

Request (all optional): `{ "name": "local-video", "metadata": {} }`

Response — `200`:

```json
{
  "type": "environment",
  "id": "env_6fe408d051ee4d8e",
  "name": "local-video",
  "metadata": {},
  "created_at": "2026-06-05T10:28:51.758Z",
  "updated_at": "2026-06-05T10:28:51.758Z"
}
```

`GET /v1/environments/{environment_id}` returns the same shape. The environment
is a placeholder locally — the in-process sandbox needs no container config.

---

#### `POST /v1/files` — Upload a video (`client.beta.files.upload`)

`multipart/form-data` with a `file` part. Stores the video in the server's video
folder (so the tools resolve it) and pushes a copy to R2 (best-effort). The
returned `id` **is** the `video_id`; mount it on a session via
`resources=[{"type":"file","file_id":id}]`.

```python
with open("accident.mp4", "rb") as f:
    file = client.beta.files.upload(file=("accident.mp4", f, "video/mp4"))
```

Response — `200` (`FileMetadata`):

```json
{
  "type": "file",
  "id": "vid_4ea586d917904a69",
  "filename": "accident.mp4",
  "mime_type": "video/mp4",
  "size_bytes": 10998,
  "created_at": "2026-06-05T10:46:00.000Z",
  "downloadable": false
}
```

`GET /v1/files/{id}` (retrieve metadata) and `GET /v1/files` (`{ "data": [...],
"has_more": false }`) return the same shape. `DELETE /v1/files/{id}` →
`{ "type": "file_deleted", "id": "..." }` and removes the local cache.

---

#### `POST /v1/sessions` — Create session

Request:

```json
{
  "agent": "agt_0bf5eb740eb8460b",
  "environment_id": "env_6fe408d051ee4d8e",
  "title": "Seattle Q&A",
  "metadata": { "video_id": "Seattle_bad_driver_accident", "max_turns_per_run": "20" }
}
```

| Field | Type | Required | Notes |
|---|---|---|---|
| `agent` | string \| object | ✓ | Agent id, or `{ "id": ..., "version": ... }`. Must exist. |
| `environment_id` | string | ✓ | Must reference an existing environment |
| `metadata.video_id` | string | ✓ | The video to analyze (or a `resources[]` entry carrying `video_id`/`id`/`name`) |
| `metadata.max_turns_per_run` | string | | Per-run turn budget (default `20`) |
| `title`, `resources` | string / array | | |

Boots the sandbox asynchronously. Response — `201` (status `pending` →
`running` ⇄ `idle`):

```json
{
  "type": "session",
  "id": "ses_beccdf5c07ab416d",
  "status": "pending",
  "agent": { "type": "agent", "id": "agt_0bf5eb740eb8460b", "version": 1 },
  "environment_id": "env_6fe408d051ee4d8e",
  "title": "Seattle Q&A",
  "metadata": { "video_id": "Seattle_bad_driver_accident" },
  "resources": [],
  "outcome_evaluations": [],
  "vault_ids": [],
  "stats": {},
  "usage": { "input_tokens": 0, "output_tokens": 0, "turns": 0, "runs": 0 },
  "created_at": "2026-06-05T10:28:51.807Z",
  "updated_at": "2026-06-05T10:28:51.807Z",
  "sandbox": { "backend": "inprocess", "status": "starting", "id": null }
}
```

`GET /v1/sessions/{session_id}` returns the same shape with live `status` (poll
until `idle` before sending). Errors: `400` if the agent/environment is unknown
or `video_id` is missing.

---

#### `POST /v1/sessions/{session_id}/events` — Send events (`events.send`)

Note the `{"events":[...]}` envelope. Each event's text is extracted from
`content` (string, or a list of `{ "type": "text", "text": ... }` blocks).
Starts a run.

Request:

```json
{ "events": [ { "type": "user.message", "content": "How long until police arrived?" } ] }
```

Response — `200`:

```json
{ "event_id": "evt_04d9cb8394724c62", "seq": 7286, "accepted_at": "2026-06-05T10:28:51.858Z" }
```

Errors: `404` unknown session, `400` empty/invalid events, `409` if a run is
already in flight.

---

#### `GET /v1/sessions/{session_id}/events/stream` — Stream (`events.stream`)

SSE (`text/event-stream`). Query: `after_seq` (default `0`). Replays mapped
events after `after_seq`, then tails live. **Run-scoped** — closes after
`session.status_idle`. Each frame is `event: <type>` / `id: <seq>` /
`data: <json>`, where `event` == the data `type`. Representative frames:

```
event: session.status_running
id: 7287
data: {"type":"session.status_running","id":"evt_...","processed_at":"...Z"}

event: agent.tool_use
id: 7290
data: {"type":"agent.tool_use","id":"toolu_...","name":"search_clip","input":{"video_id":"...","query":"...","start_time":300,"end_time":360},"processed_at":"...Z"}

event: agent.tool_result
id: 7305
data: {"type":"agent.tool_result","id":"evt_...","tool_use_id":"toolu_...","content":[{"type":"text","text":"Agent Reasoning: ... Final Response: ..."}],"is_error":false,"processed_at":"...Z"}

event: agent.message
id: 7420
data: {"type":"agent.message","id":"evt_...","content":[{"type":"text","text":"The police car arrived about 4 minutes 24 seconds after the accident..."}],"processed_at":"...Z"}

event: session.status_idle
id: 7421
data: {"type":"session.status_idle","id":"evt_...","stop_reason":"end_turn","processed_at":"...Z"}
```

The answer is `agent.message.content[].text`. See the
[event mapping table](#event-mapping-engine--sdk) for the full translation.

---

#### `GET /v1/sessions/{session_id}/events` — List mapped events

Non-streaming. Query: `after_seq` (default `0`), `limit` (default `1000`).
Returns the same mapped events as the stream:

```json
{ "data": [ { "type": "session.status_running", "id": "evt_...", "processed_at": "...Z" }, "..." ], "has_more": false }
```

---

# 2. Threads (not implemented — extension point)

In the Managed Agents model the object hierarchy is:

```
Agent          reusable template (model, system, tools)
  └─ Session    one running instance: an Environment (container) + resources/memory
       └─ Thread   an independent line of execution/conversation within the session
            └─ Event  the individual messages/actions (user.message, agent.tool_use, …)
```

A **thread** is one "train of thought" inside a session. The session is the
workspace (sandbox, loaded video, memory); threads are concurrent or branching
conversations within it. The SDK exposes them as
`client.beta.sessions.threads.*` with `/v1/sessions/{id}/threads/...` endpoints
and `session.thread_*` / `agent.thread_message_*` events. Threads enable:

- **Concurrency** — multiple sub-tasks at once in the same environment.
- **Branching** — fork the agent's state to explore alternatives (the protocol even has `branch`/`commit` event types).
- **Multi-agent** — a coordinator spawns worker threads sharing the session's resources but with separate context windows.

### What we do today

We **flatten threads away**. A session maps to a single `SessionRunner`, and
each `events.send` starts a run whose events go straight onto the *session*
stream — effectively one implicit thread. That's why:

- `client.beta.sessions.threads.*` is not backed yet (the endpoints 404).
- We emit `session.status_idle` (session-level) rather than `session.thread_status_idle`.

For single-question video Q&A, one implicit thread is all you need.

### How to extend to real threads later

The engine is structured so this is additive, not a rewrite:

1. **Storage** — add a `threads` table (or `app.state.threads`) keyed by `(session_id, thread_id)`; tag persisted events with a `thread_id`.
2. **Runner** — let a `SessionRunner` own multiple in-flight runs keyed by `thread_id` (today it allows one run at a time). Each thread keeps its own message list / context window; the sandbox (video, tools) stays shared at the session level.
3. **Routes** — implement `POST /v1/sessions/{id}/threads` (create), `…/threads/{tid}/events` (send/list), `…/threads/{tid}/stream` (per-thread SSE). The session-level stream stays as the multiplexed view across threads.
4. **Events** — emit the thread-scoped variants the SDK already understands: `session.thread_created`, `session.thread_status_running` / `…_idle` / `…_terminated`, `agent.thread_message_received` / `…_sent`, `agent.thread_context_compacted`.

Concretely for the video agent, threads would let one session (one booted
sandbox, one loaded video) run several analyses as independent branches — e.g.
fan out "police arrival", "tow arrival", and "injuries" as parallel threads —
instead of serial runs.

---

# 3. Native Ambient API (`/ambient/v1`)

The original video-first API. Same engine and event semantics; auth is
`Authorization: Bearer <token>`. These routes moved under `/ambient` so the
Managed Agents router can own `/v1/sessions`.

### Videos

#### `POST /ambient/v1/videos` — Upload a video

`multipart/form-data` with a `file` part. Returns a `video_id` to pass into
session create. Stores locally (for the tools) **and** to R2 (best-effort).

```bash
curl -X POST http://127.0.0.1:8080/ambient/v1/videos \
  -H "Authorization: Bearer dev-token" \
  -F "file=@accident.mp4"
```

```json
{
  "video_id": "vid_d0b91cb7f1e1492e",
  "filename": "accident.mp4",
  "mime_type": "video/mp4",
  "size_bytes": 10998,
  "r2_key": "vid_d0b91cb7f1e1492e/source.mp4",
  "created_at": "2026-06-05T10:46:54.630Z"
}
```

`GET /ambient/v1/videos` lists uploads; `GET /ambient/v1/videos/{video_id}` retrieves
one. Uploads share storage with the Files API — a `vid_…` id works as both a
native `video_id` and an SDK file id.

> Videos can also be pre-placed: drop `<video_id>.mp4` into `settings.video_folder`
> and reference it directly (no upload needed).

### Sessions

A **session** binds one video to one agent conversation and boots a sandbox in
the background immediately after creation.

#### `POST /ambient/v1/sessions` — Create

```bash
curl -X POST http://127.0.0.1:8080/ambient/v1/sessions \
  -H "Authorization: Bearer dev-token" -H "Content-Type: application/json" \
  -d '{
    "video_id": "Seattle_bad_driver_accident",
    "title": "Seattle accident questions",
    "model": "openai/gpt-5.5",
    "max_turns_per_run": 20,
    "permission_policy": {"type": "always_allow"}
  }'
```

To run this session's media ops in E2B (per-session, no server env needed — the
server must still have `E2B_API_KEY` + S3 + a built `E2B_TEMPLATE`):

```bash
curl -X POST http://127.0.0.1:8080/ambient/v1/sessions \
  -H "Authorization: Bearer dev-token" -H "Content-Type: application/json" \
  -d '{
    "video_id": "Seattle_bad_driver_accident",
    "sandbox": { "backend": "e2b", "limits": { "memory_mb": 8192, "wall_seconds": 1800 } }
  }'
```

| Field | Type | Required | Description |
|---|---|---|---|
| `video_id` | string | ✓ | Identifies the video in the tool registry |
| `model` | string | | Override the agent model (default from server config) |
| `title` | string | | Human-readable label |
| `subtitle_path` | string | | Local path to an SRT file for this video |
| `max_turns_per_run` | integer | | Max agent turns per user message (default: 5) |
| `metadata` | object | | Arbitrary string key-value pairs |
| `sandbox` | object | | `{ "backend": "inprocess" \| "e2b", "limits": { "cpu_seconds", "memory_mb", "wall_seconds" } }` (selects the media backend; see [Tool execution & media backends](#tool-execution--media-backends)) |
| `quota` | object | | `{ "max_tokens", "max_wall_seconds", "max_cost_usd" }` |
| `permission_policy` | object | | `{ "type": "always_allow" \| "always_ask" \| "deny" }` (`always_ask`/`deny` reserved) |

Returns `201` with the `SessionResource` (`session_id`, `status`, `sandbox`,
`usage`, …). The sandbox boots asynchronously — poll `GET` until `status` is
`ready`, or post immediately (the runner queues it).

#### Other session routes

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/ambient/v1/sessions?limit=N` | list sessions |
| `GET` | `/ambient/v1/sessions/{id}` | get one (live `status`, `usage`) |
| `DELETE` | `/ambient/v1/sessions/{id}` | cancel run, tear down sandbox, remove record |
| `POST` | `/ambient/v1/sessions/{id}/cancel` | cancel in-flight run, keep session `ready` |

### Events

#### `POST /ambient/v1/sessions/{id}/events` — Send a user message

```bash
curl -X POST http://127.0.0.1:8080/ambient/v1/sessions/ses_xxx/events \
  -H "Authorization: Bearer dev-token" -H "Content-Type: application/json" \
  -d '{"type":"user.message","content":"What time did the police arrive?"}'
```

| Field | Type | Required | Description |
|---|---|---|---|
| `type` | string | ✓ | Must be `"user.message"` |
| `content` | string | ✓ | The user's message text |
| `model` | string | | Override the model for this run only |
| `output_structure` | object | | JSON Schema the final answer must follow (dict form of a Pydantic `.model_json_schema()`) |
| `metadata` | object | | Arbitrary string key-value pairs |

Returns `200` with `{event_id, seq, accepted_at}`; `409` if a run is already in
flight.

#### `GET /ambient/v1/sessions/{id}/events` — Fetch or stream

- **Batch (default):** returns persisted events. Query: `after_seq` (default `0`), `limit` (default `1000`).
- **Stream (`?stream=true`):** SSE. Replays events after `after_seq`, then tails live. **Run-scoped** — the stream closes after `run.completed`. Resume with `Last-Event-ID: <seq>` (or `after_seq`). Reconnecting with `after_seq=0` on a session with a prior run will close on that earlier run's `run.completed`, so point `after_seq` past it.

```bash
curl -N "http://127.0.0.1:8080/ambient/v1/sessions/ses_xxx/events?stream=true&after_seq=0" \
  -H "Authorization: Bearer dev-token" -H "Accept: text/event-stream"
```

### Native SSE event reference

Each frame is `event: <type>` / `id: <seq>` / `data: <json>`. All events except
`chat.completion.chunk` include `session_id`, `event_id`, `seq`, `ts`.

| Event | Payload | When |
|---|---|---|
| `session.status_changed` | `status` | `created`→`starting`→`ready`→`running`→`ready`→`terminated` |
| `run.started` | `run_id` | run begins |
| `turn.start` / `turn.end` | `run_id`, `turn`, `finish_reason` | per turn (`finish_reason`: `stop`\|`tool_calls`\|`length`) |
| `chat.completion.chunk` | raw OpenAI chunk | relayed verbatim (final chunk carries `usage`) |
| `tool.scheduled` | `tool_use_id`, `name`, `input` | LLM requested a tool |
| `tool.started` | `tool_use_id`, `name` | dispatcher began executing the tool |
| `tool.result` | `analysis`, `attachments` | tool succeeded |
| `tool.failed` | `error: {code, message, traceback}` | tool raised |
| `run.completed` | `stop_reason`, `answer` | run ended (`end_turn`\|`max_turns`\|`cancelled`\|`error`) |
| `error` | `code`, `message`, `fatal` | unrecoverable error |

Heartbeat: sse-starlette emits a `: ping` comment every ~15s during quiet
periods (single source — no app-level ping event).

---

# Operational notes

### HTTP error responses

```json
{ "detail": { "code": "not_found", "message": "session ses_xxx not found" } }
```

| Status | Code | When |
|---|---|---|
| `401` | `unauthorized` | Missing/invalid key (`x-api-key` or Bearer) |
| `404` | `not_found` | Session / agent / environment doesn't exist |
| `400` | `bad_request` | Malformed body, missing `video_id`, unknown agent/env |
| `409` | `conflict` | Run already in flight |

### Tool execution & media backends

Tools **always run in the server process**. A `ToolDispatcher` invokes each tool
(`get_video_description`, `search_clip`, `focus_clip`) on a worker thread, so
blocking media work and the LLM HTTP calls never stall the event loop or live SSE
streams. **The LLM call always stays on the host** — it is never run in a sandbox.

The `sandbox.backend` selects only **where `VideoFrameTools`' media ops (frame /
clip extraction) execute**. Set it per session via `sandbox.backend` (native API)
or server-wide via the `SANDBOX_BACKEND` env var (used by the Managed Agents path,
which has no per-session sandbox field).

| Backend | Where media runs | Notes |
|---|---|---|
| `inprocess` | **Default.** Local ffmpeg/decord ([video_tools.py](../tools/video_tools.py)). | No isolation. Frames/clips are local files; the host base64-embeds them for the LLM. |
| `e2b` | A per-session [E2B](https://e2b.dev) sandbox runs the media-only [main.py](../sandboxes/e2b/video-analysis-v1/main.py) (`extract-frames` / `fetch-clip`). | The box fetches the source video from S3, runs ffmpeg/decord, uploads frames/clips back to S3, and returns presigned URLs. The host then calls the LLM with those URLs. |

For `e2b`, the sandbox is owned by the `SessionRunner` (created in
`start_sandbox()`, killed on teardown) and handed to the `ToolDispatcher`, which
publishes it to the tools through a `ContextVar`. It requires:

- `E2B_API_KEY` set;
- a built template (`E2B_TEMPLATE`, default `video-analysis-v1`) that contains the media `main.py` — build it with `ambient/sandboxes/e2b/video-analysis-v1/build_prod.py`;
- S3 configured (`S3_*` / `AWS_*`) with the source video at `s3://<S3_BUCKET>/<S3_VIDEO_BASE_KEY>/<video_id>.mp4`.

Sandbox `limits` (`cpu_seconds` / `memory_mb` / `wall_seconds`) size the E2B box
and its idle keepalive; they're inert for `inprocess`.

### Session status state machine

```
created → starting (sandbox booting) → ready ⇄ running → terminated (DELETE / idle TTL)
```

In the Managed Agents view these map to `pending` → `idle` ⇄ `running` →
`terminated`.

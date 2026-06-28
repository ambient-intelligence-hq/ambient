# Production-readiness TODO

Functional gaps between the current managed-agents video server and a production
deployment. Ordered by blocking severity. The server today is a correct
**single-process prototype**; the Tier 1 items are what break first in real use.

## Tier 1 — blockers (state & lifecycle)

- [x] **Persist non-session state.** `agents`, `environments`, and `files` now
  live in Postgres tables (`ambient/server/store.py`), not in-memory `app.state`
  dicts; routes read/write them through the store. Survives restart and is shared
  across workers.
- [x] **Rehydrate runner state on access.** `SessionRunner.messages` is persisted
  to the `sessions.messages` JSONB column after every turn. A worker without the
  runner in memory rebuilds it via `_get_or_rehydrate_runner`
  (`ambient/server/routes/managed_agents.py`) from the persisted messages, so
  history survives restarts and cross-worker handoff.
- [ ] **Session + sandbox teardown.** No terminate/cancel endpoint in the
  managed-agents router (only `DELETE /files`). Idle/hard TTL are specced
  (`ambient/server/SPEC.md:421-422`) and configured (`session_idle_ttl_seconds`,
  `session_hard_ttl_seconds`) but **not implemented** — no reaper. e2b sandboxes
  are killed only on app shutdown (`ambient/server/app.py:21-22`) → cost leak +
  resource exhaustion.
  → Background reaper (idle + hard TTL) and an explicit terminate endpoint.
- [x] **Multi-process — multi-worker.** Run ownership is a Redis lease and SSE
  fan-out is Redis pub/sub (`ambient/server/broker.py`), so `send`/`stream` work
  across workers and `server_workers > 1` is supported. Remaining for full **HA**:
  auto-resume of a run interrupted by a worker crash (lease self-expires today,
  but the half-finished run isn't restarted), and graceful drain on rolling
  deploy. e2b reattach-by-id (`media_sandbox.connect`) is in place.

## Tier 2 — correctness / feature gaps

- [ ] **Enforce structured output.** Currently prompt-only steering (schema
  appended to the user turn); no validation, no retry, no provider
  structured-output/tool-forcing. The metadata→`output_structure` path isn't read
  by the SDK flow. → Validate against schema + retry, or use provider mode.
- [ ] **Usage / token accounting.** Nothing in the runner or `llm_client` updates
  `usage` — stays `{input:0, output:0, turns:0, runs:0}`; `quota` unused. No
  billing or per-session budget. → Fold token counts from stream chunks into the
  session; gate on budget.
- [ ] **Retry the streaming LLM call.** `stream_chat_completion` has no retry; a
  mid-stream provider hiccup drops the whole turn (tool sub-calls retry, the main
  loop doesn't).

## Tier 3 — multi-tenancy / security

- [ ] **Real auth + tenant scoping.** Single static API key; the store isn't
  scoped by owner, so any valid key can read/drive every session. No rate
  limiting. → Per-tenant auth, owner-scoped store, quotas.

## Tier 4 — video pipeline / content

- [ ] **Standardize video storage.** In-process backend resolves videos from a
  hardcoded personal path (`settings.video_folder`); only the e2b path reliably
  pulls from S3. → Object storage as source of truth; validate format/size/
  duration.
- [ ] **Audio / transcription.** STT is stubbed → frames-only; speech-dependent
  questions can't be answered. (Scope-dependent.)
- [x] **Wire fetch_clip to the tile fast path.** Ingestion pre-transcodes the
  whole video into GOP-aligned tiles + manifest (`transcode-tiles`), and the read
  side now serves from tiles via `fetch_clips` → `(clips, global_offset)`:
  * **no size cap** (e.g. qwen): single clip stream-copy-assembled from covering
    tiles via `concat-tiles` (snapped to tile boundaries);
  * **size cap set** (e.g. Gemini 14 MB): covering tiles returned as separate
    clips (each already under the cap) on one continuous 0-based timeline, so the
    model reads them as one clip and cites mm:ss within the window;
  * citations offset by `global_offset` in `focus_clip` / `search_clip`.
  Falls back to on-demand transcode on miss / param-mismatch / oversized tile.

## Tier 5 — observability / ops

- [ ] Structured logging (replace scattered `print()` in tools), metrics, tracing.
- [ ] Server Dockerfile (only the e2b sandbox has one).
- [ ] Readiness probe reflecting LLM/sandbox reachability (`/health` is shallow).
- [ ] Graceful shutdown that drains in-flight runs instead of hard-cancelling.

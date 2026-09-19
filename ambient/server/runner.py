"""SessionRunner: the per-session agent loop.

One instance per session. Owns:
  - the message list (system + user + assistant + tool_result blocks)
  - the in-process ToolDispatcher (and, for the "e2b" backend, the per-session
    media sandbox handed to it)
  - a fan-out registry of live SSE subscribers
  - the asyncio.Task that drives the current run

Each user.message triggers a run. A run = one or more turns through the LLM, with
tool calls dispatched in-process between turns.

Events are persisted to the SQLite store *and* fanned out to subscribers in real time.
"""
from __future__ import annotations

import asyncio
import json
import time
import uuid
import os
from typing import Any, Optional

from ambient.config import settings
from ambient.llm import normalize_usage
from ambient.prompt import get_system_prompt
from ambient.server.broker import Broker
from ambient.server.llm_client import assemble_assistant_message, stream_chat_completion
from ambient.server.pricing import estimate_cost
from ambient.server.sandbox import SandboxLimits, ToolDispatcher
from ambient.server.store import Store
from ambient.tools import TOOLS
from ambient.server.ingest import ensure_description
from ambient.config import get_model_modalities, self_video_analysis_enabled
import logging

log = logging.getLogger(__name__)

MAX_CLIPS = 25
MAX_FRAMES = 3000

_TOKEN_FIELDS = (
    "input_tokens",
    "output_tokens",
    "cache_read_input_tokens",
    "cache_creation_input_tokens",
    "total_tokens",
)


def _empty_usage() -> dict:
    """Zeroed session usage aggregate (ATIF-style token fields + cost + rollups)."""
    return {
        **{f: 0 for f in _TOKEN_FIELDS},
        "cost": 0.0,
        "turns": 0,
        "runs": 0,
        "requests": 0,
        "by_source": {},   # "agent" / "tool:<name>" -> {tokens..., cost, requests}
        "by_model": {},    # model id -> {tokens..., cost, requests}
    }


def _add_usage_bucket(bucket: dict, usage_norm: dict, cost: float) -> None:
    for f in _TOKEN_FIELDS:
        bucket[f] = int(bucket.get(f, 0)) + int(usage_norm.get(f, 0))
    bucket["cost"] = round(float(bucket.get("cost", 0.0)) + cost, 6)
    bucket["requests"] = int(bucket.get("requests", 0)) + 1


class SandboxNotReady(Exception):
    """A rehydrating worker found no sandbox id yet (creator boot in progress)."""


def _retain_last_n_by_type(messages: list[dict], content_type: str, n: int) -> list[dict]:
    new_messages: list[dict] = []
    seen = 0
    for message in messages[::-1]:
        if message.get("role") == "user" and isinstance(message.get("content"), list):
            kept = []
            for content in message["content"]:
                if isinstance(content, dict) and content.get("type") == content_type:
                    seen += 1
                    if seen >= n:
                        continue
                kept.append(content)
            new_msg = dict(message)
            new_msg["content"] = kept
            if kept:
                new_messages.append(new_msg)
        else:
            new_messages.append(message)
    return new_messages[::-1]


def _normalize_for_openai(messages: list[dict]) -> list[dict]:
    """Anthropic-style assistant/tool blocks -> OpenAI chat-completions shape."""
    out: list[dict] = []
    for message in messages:
        role = message.get("role")
        content = message.get("content")

        if role == "assistant" and isinstance(content, list):
            text_parts: list[str] = []
            tool_calls: list[dict] = []
            for block in content:
                if not isinstance(block, dict):
                    continue
                if block.get("type") == "text":
                    text_parts.append(block.get("text", ""))
                elif block.get("type") == "tool_use":
                    tool_calls.append({
                        "id": block.get("id"),
                        "type": "function",
                        "function": {
                            "name": block.get("name"),
                            "arguments": json.dumps(block.get("input", {})),
                        },
                    })
            am: dict[str, Any] = {"role": "assistant", "content": "\n".join(p for p in text_parts if p).strip()}
            if tool_calls:
                am["tool_calls"] = tool_calls
            out.append(am)
            continue

        if role == "user" and isinstance(content, list):
            tool_results = [b for b in content if isinstance(b, dict) and b.get("type") == "tool_result"]
            if tool_results and len(tool_results) == len(content):
                for tr in tool_results:
                    tc = tr.get("content", "")
                    if isinstance(tc, (dict, list)):
                        tc = json.dumps(tc)
                    out.append({"role": "tool", "tool_call_id": tr.get("tool_use_id"), "content": str(tc)})
                continue
            only_text = all(isinstance(b, dict) and b.get("type") == "text" for b in content)
            if only_text:
                merged = "\n".join(b.get("text", "") for b in content).strip()
                out.append({"role": "user", "content": merged})
                continue

        out.append(message)
    return out


class SessionRunner:
    def __init__(
        self,
        *,
        session_id: str,
        video_id: str,
        model: str,
        max_turns_per_run: int,
        backend: str,
        limits: SandboxLimits,
        store: Store,
        broker: Broker,
        messages: Optional[list[dict]] = None,
        mode: str = "agent",
        system: Optional[str] = None,
    ):
        self.session_id = session_id
        self.video_id = video_id
        self.video_description = None
        # "agent" -> multi-step tool loop; "fast" -> single dense-frame vision call.
        self.mode = mode
        # Optional per-agent task prompt appended to the mode's base system prompt.
        self.agent_system = system
        # Response schema for the current run (set by submit_user_message).
        self.output_structure: Optional[dict] = None
        self._fast_frames_loaded = False
        self.model = model
        self.max_turns_per_run = max_turns_per_run
        self.backend = backend
        self.limits = limits
        self.store = store
        self.broker = broker

        # Tools always run in-process; only VideoFrameTools' media ops vary by
        # backend. For "e2b" the runner owns a per-session media sandbox and
        # hands it to the dispatcher. The dispatcher forces this session's video_id
        # onto every tool call (the model can't be trusted to echo the opaque id).
        self._dispatcher = ToolDispatcher(video_id=self.video_id)
        self._media_box: Optional[object] = None
        self.sandbox_id: Optional[str] = None
        model_modalities = [m.value for m in get_model_modalities(self.model)]

        # On a fresh session `messages` is None -> empty list; seed is built in
        # start_sandbox() after the description is ready. On rehydration the
        # caller passes the persisted list (already includes the seeded turns).
        self.messages: list[dict] = messages or self._get_seed_messages(model_modalities)
        self._run_task: Optional[asyncio.Task] = None
        self._run_lock = asyncio.Lock()
        self._status = "created"
        self._cancel_requested = False
        # Background task that resolves the video description off the critical
        # path (Phase 1: session readiness never blocks on the ingest pipeline).
        self._description_task: Optional[asyncio.Task] = None
        # Set once the youtube source MP4 is confirmed on S3, so media tools don't
        # fetch a not-yet-uploaded clip. Non-youtube sources are ready immediately.
        self._source_ready = False
        # Running token/cost totals for the session. Seeded from the persisted
        # session record on the first run (survives rehydration), then kept in
        # memory and written back after each turn.
        self.usage: dict = _empty_usage()
        self._usage_seeded = False

    # --- lifecycle --------------------------------------------------------



    def _get_seed_messages(self, model_modalities: list[str]) -> list[dict]:
        if self.mode == "fast":
            from ambient.prompt import get_fast_system_prompt
            system = get_fast_system_prompt()
        else:
            system = get_system_prompt(model_modalities)
        # A per-agent system prompt (agents.create system=...) is appended as the
        # task, keeping the mode's base (tools / frame instructions + answering).
        if self.agent_system:
            system = f"{system}\n\n## Task\n{self.agent_system}"
        return [{"role": "system", "content": system}]
    
    def _video_source_hint(self, rec: dict | None) -> Optional[str]:
        """Seed blurb telling the model where the video lives and how to get a
        local copy if it needs one for shell/ffmpeg (the bash tool).

        The video *tools* resolve by video_id and never need this, so it's only
        emitted when the bash tool is enabled. For e2b the source may be streamed
        (not on disk), so we share the cache folder + the S3 location + the exact
        command to materialize it on demand — rather than advertising a path that
        may not exist. For the host backend the file is always local, so we hand
        over the path directly."""
        if not settings.enable_bash_tool:
            return None
        ext = os.path.splitext((rec or {}).get("local_path") or (rec or {}).get("r2_key") or "")[1] or ".mp4"
        if self.backend == "e2b":
            folder = f"{settings.box_video_folder}/{self.video_id}"
            lines = [
                "Video source (E2B sandbox):",
                f"- Local cache folder: {folder}/ — check here first for {self.video_id}{ext}.",
                f"- If it is not there, the source is being streamed from S3. To get a local "
                f"file for ffmpeg/shell, run: `python /app/main.py ensure-source --video-id "
                f"{self.video_id}` — it downloads the source and prints the local path. Only "
                f"do this if you actually need the file on disk.",
            ]
            if settings.s3_bucket:
                lines.append(
                    f"- S3 source: s3://{settings.s3_bucket}/{settings.s3_video_base_key}/{self.video_id}{ext}"
                )
            return "\n".join(lines)
        # Host backend: the file is always on disk (store_video writes it, or the
        # range proxy convention). Hand over the path + folder; no download needed.
        lp = (rec or {}).get("local_path") or os.path.join(settings.video_folder, f"{self.video_id}{ext}")
        if os.path.exists(lp):
            return (
                f"Video source (host): {lp} — already on disk, use it directly; do NOT "
                f"search the filesystem for it.\n- Video folder: {settings.video_folder}"
            )
        return None
        

    async def start_sandbox(self) -> None:
        await self._set_status("starting")
        t0 = time.monotonic()
        try:
            if self.backend == "e2b":
                # Create the per-session media sandbox (blocking boot offloaded).
                from ambient.sandboxes.e2b.media_sandbox import E2BMediaSandbox
                self._media_box = E2BMediaSandbox()
                await asyncio.to_thread(
                    self._media_box.create,template=settings.e2b_template, limits=self.limits
                )
                self._dispatcher.media_box = self._media_box
                self.sandbox_id = self._media_box.id
            else:
                self.sandbox_id = f"ip_{uuid.uuid4().hex[:12]}"
        except Exception as exc:  # noqa: BLE001
            await self._emit("error", {"code": "sandbox_boot_failed", "message": str(exc), "fatal": True})
            await self._set_status("failed")
            raise

        # Resolve the video description in the background: the session is ready to
        # take input as soon as the sandbox boots. `_run_until_done` injects the
        # description into the conversation once it lands.
        if not self_video_analysis_enabled(self.model):
            self._maybe_start_description()

        boot_ms = int((time.monotonic() - t0) * 1000)
        await self._emit("session.status_changed", {
            "status": "ready",
            "previous_status": "starting",
            "sandbox": {"status": "ready", "id": self.sandbox_id, "boot_ms": boot_ms, "region": None},
        })
        self._status = "ready"

    _DESC_INJECT_MARKER = "[system update] Video description"

    def _maybe_start_description(self) -> None:
        """Start the background description resolver unless it's already running,
        already resolved, or already present in the (rehydrated) conversation."""
        if self._description_task is not None:
            return
        if self.video_description is not None or self._description_injected():
            return
        self._description_task = asyncio.create_task(
            self._resolve_description(), name=f"desc-{self.session_id}"
        )

    async def _resolve_description(self) -> None:
        """Resolve the video description off the critical path.

        Never flips the session to `failed`: the agent's tools work without a
        description, so a failure here is a non-fatal error event and the session
        stays usable. On success the description is also published to the tool
        dispatcher's cache so clip tools receive it.
        """
        try:
            description = await ensure_description(self.store, self.video_id, box=self._media_box)
            self.video_description = description
            if description:
                self._dispatcher._video_description[self.video_id] = description
            log.info(f"Description ready for video {self.video_id}")
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001
            log.error(f"Error ensuring description for video {self.video_id}: {str(exc)}")
            await self._emit("error", {
                "code": "video_description_failed", "message": str(exc), "fatal": False,
            })

    async def attach_sandbox(self, sandbox_rec: dict) -> None:
        """Reattach a rehydrated runner to its session's sandbox.

        Called when a worker picks up a session it didn't create. For `e2b` we
        reconnect to the live media sandbox by id. If no id is recorded yet the
        creating worker's boot hasn't finished, so we raise `SandboxNotReady`
        (a retryable condition) rather than booting a duplicate sandbox and
        leaking the first. For `inprocess` there is no real sandbox to reattach.
        """
        if self._media_box is not None:
            return
        sid = (sandbox_rec or {}).get("id")
        if self.backend != "e2b":
            self.sandbox_id = sid or f"ip_{uuid.uuid4().hex[:12]}"
            if not self_video_analysis_enabled(self.model):
                self._maybe_start_description()
            return
        if not sid:
            raise SandboxNotReady(self.session_id)
        from ambient.sandboxes.e2b.media_sandbox import E2BMediaSandbox
        box = E2BMediaSandbox()
        await asyncio.to_thread(box.connect, sid)
        self._media_box = box
        self._dispatcher.media_box = box
        self.sandbox_id = box.id
        self._status = "ready"
        # A rehydrated worker may pick up a session whose description never landed
        # (creator died mid-boot); resolve it here unless it's already in the
        # persisted conversation.
        if not self_video_analysis_enabled(self.model):
            self._maybe_start_description()

    async def terminate(self) -> None:
        if self._description_task and not self._description_task.done():
            self._description_task.cancel()
            try:
                await self._description_task
            except (asyncio.CancelledError, Exception):
                pass
        if self._run_task and not self._run_task.done():
            self._run_task.cancel()
            try:
                await self._run_task
            except (asyncio.CancelledError, Exception):
                pass
        try:
            if self._media_box is not None:
                await asyncio.to_thread(self._media_box.kill)
        finally:
            self._media_box = None
            self._dispatcher.media_box = None
            await self._set_status("terminated")
            # Tell any live SSE subscribers (possibly on other workers) to close.
            await self.broker.publish(self.session_id, {"__close__": True, "seq": 0})

    # --- user input -------------------------------------------------------

    async def submit_user_message(
        self,
        content: str,
        model_override: Optional[str] = None,
        output_structure: Optional[dict] = None,
    ) -> None:
        """Start a new run from a user message. Caller has already verified no run is in flight."""
        self.output_structure = output_structure
        self._cancel_requested = False
        model = model_override or self.model

        # FAST mode: one vision pass over a densely-sampled, timestamp-labeled frame
        # set (loaded once, reused across follow-ups). No tools, no schema injected
        # into the text — response_format enforces the schema natively.
        if self.mode == "fast":
            # Frame sampling happens inside the run (streamed) so the /events POST
            # returns immediately instead of blocking on extraction.
            self.messages.append({"role": "user", "content": [{"type": "text", "text": content}]})
            await self.store.save_messages(self.session_id, self.messages)
            self._run_task = asyncio.create_task(self._run_fast(model), name=f"fast-{self.session_id}")
            return

        text = content

        if output_structure:
            # Mirror ambient.agent.run_agent: steer the final answer to the schema.
            text = (
                f"{content}\n\nYour final answer must strictly follow this JSON schema:\n"
                f"{json.dumps(output_structure)}"
            )

        if len(self.messages) == 1:
            attach_video_description = not self_video_analysis_enabled(self.model)
            seed = await self._build_seed_text(attach_video_description)
            self.messages.append({"role": "user", "content": [{"type": "text", "text": seed}]})
        self.messages.append({"role": "user", "content": [{"type": "text", "text": text}]})
        # Persist the user turn up front so a rehydrating worker sees it even if
        # this run is interrupted before producing an assistant turn.
        await self.store.save_messages(self.session_id, self.messages)
        self._run_task = asyncio.create_task(self._run_until_done(model), name=f"run-{self.session_id}")

    def is_running(self) -> bool:
        return self._run_task is not None and not self._run_task.done()

    async def cancel_run(self) -> bool:
        if not self.is_running():
            return False
        self._cancel_requested = True
        assert self._run_task is not None
        self._run_task.cancel()
        try:
            await self._run_task
        except (asyncio.CancelledError, Exception):
            pass
        return True

    # --- agent loop -------------------------------------------------------

    async def _run_until_done(self, model: str) -> None:
        run_id = f"run_{uuid.uuid4().hex[:12]}"
        await self._set_status("running")
        await self._emit("run.started", {"run_id": run_id, "model": model})
        stop_reason = "end_turn"
        final_answer = ""
        last_assistant_text = ""

        await self._seed_usage_once()
        self.usage["runs"] = int(self.usage.get("runs", 0)) + 1

        # Ground the first answer in the description: wait (bounded) for it before
        # the first LLM call. Session readiness is unaffected; only this run waits,
        # and only until the description lands or the timeout elapses.
        if settings.first_turn_wait_for_description and not self._description_injected():
            await self._emit("tool.progress", {
                "run_id": run_id,
                "turn": 1,
                "progress": {"message": "preparing video description"},
            })
            await self._await_description(settings.first_turn_description_timeout_seconds)

        # Self-video-analysis: attach the whole video as a cacheable prefix on the
        # first run (streamed here, so submit_user_message stays instant). media_plan
        # decides video_url vs sampled frames per the AGENT endpoint's capability
        # (openrouter_providers.json). Skipped on follow-ups/rehydration — the context
        # is already in the persisted messages.
        if self_video_analysis_enabled(self.model) and not self._has_video_context():
            await self._emit("tool.progress", {
                "run_id": run_id,
                "turn": 1,
                "progress": {"message": "attaching video"},
            })
            await self._await_source_ready()
            await self._attach_video_context(
                settings.agent_base_url or settings.llm_base_url, model)

        try:
            for turn in range(1, self.max_turns_per_run + 1):
                # Keep the ownership lease alive for the duration of the run.
                await self.broker.renew(self.session_id)
                await self._emit("turn.start", {"run_id": run_id, "turn": turn})

                # Fold in the background-resolved description as soon as it lands.
                if self.video_description and not self._description_injected():
                    self.messages.append({"role": "user", "content": [{
                        "type": "text",
                        "text": f"{self._DESC_INJECT_MARKER} now available:\n{self.video_description}",
                    }]})

                prepared = _normalize_for_openai(self.messages)
                prepared = _retain_last_n_by_type(prepared, "video_url", MAX_CLIPS)
                prepared = _retain_last_n_by_type(prepared, "image_url", MAX_FRAMES)

                chunks: list[dict] = []
                try:
                    async for chunk in stream_chat_completion(
                        model=model,
                        messages=prepared,
                        tools=TOOLS,
                    ):
                        chunks.append(chunk)
                        await self._emit("chat.completion.chunk", chunk)
                except asyncio.CancelledError:
                    raise
                except Exception as exc:
                    await self._emit("error", {"code": "llm_error", "message": str(exc), "fatal": False})
                    stop_reason = "error"
                    break

                assistant_msg, finish_reason, tool_uses, reasoning, agent_usage = (
                    assemble_assistant_message(chunks)
                )
                self.messages.append(assistant_msg)

                self.usage["turns"] = int(self.usage.get("turns", 0)) + 1
                await self._record_usage(run_id, turn, model, agent_usage, "agent")

                # Surface the agent's extended thinking as its own event (a text
                # carrier via the SDK-visible agent.thinking event) so clients can
                # display it. Emitted before tool dispatch, mirroring turn order.
                if settings.expose_thinking and reasoning:
                    await self._emit("agent.reasoning", {
                        "run_id": run_id,
                        "turn": turn,
                        "reasoning": reasoning,
                    })

                text = self._extract_answer_text(assistant_msg)
                if text:
                    last_assistant_text = text

                await self._emit("turn.end", {
                    "run_id": run_id,
                    "turn": turn,
                    "finish_reason": finish_reason or "stop",
                })

                if finish_reason in (None, "stop", "length"):
                    final_answer = text
                    stop_reason = "end_turn"
                    break

                if finish_reason == "tool_calls" and tool_uses:
                    await self._run_tool_calls(run_id, turn, tool_uses)
                    # Persist the grown context so another worker can rehydrate it
                    # mid-conversation.
                    await self.store.save_messages(self.session_id, self.messages)
                    await self._persist_usage()
                else:
                    final_answer = text
                    stop_reason = "end_turn"
                    break
            else:
                stop_reason = "max_turns"
                # The loop hit the turn cap mid tool-use. Fall back to the last
                # non-empty assistant text so the client still gets something
                # rather than an empty "no response".
                final_answer = final_answer or last_assistant_text

            await self._emit("run.completed", {
                "run_id": run_id,
                "stop_reason": stop_reason,
                "answer": final_answer,
                "usage": self.usage,
            })
        except asyncio.CancelledError:
            await self._emit("run.completed", {
                "run_id": run_id,
                "stop_reason": "cancelled",
                "answer": "",
            })
        finally:
            await self.store.save_messages(self.session_id, self.messages)
            await self._persist_usage()
            await self._set_status("ready")
            # Release the run-ownership lease so any worker can serve the next turn.
            await self.broker.release(self.session_id)

    # --- fast mode --------------------------------------------------------

    def _has_video_context(self) -> bool:
        """True if the whole-video context (a `video_url` or sampled-frame message) is
        already attached near the top of the conversation. Used as a rehydration-safe
        'attach once' guard: the persisted messages are the source of truth, so a
        rehydrated or follow-up run won't attach a second copy. Only the attached
        context carries media blocks this early — tool-result media come later, after
        an assistant turn."""
        for msg in self.messages[:3]:
            if msg.get("role") != "user":
                continue
            content = msg.get("content")
            if isinstance(content, list) and any(
                isinstance(b, dict) and b.get("type") in ("video_url", "image_url")
                for b in content
            ):
                return True
        return False

    async def _attach_video_context(self, base_url: Optional[str], model: str) -> None:
        """Attach the whole video once as a cacheable prefix (right after the system
        prompt): a single `video_url` when the endpoint samples video densely
        (self-hosted vLLM, or a pinned dense OpenRouter provider), else uniformly-
        sampled `image_url` frames.

        The choice is `plan_media("context", base_url, model, ...)` against the
        endpoint that will READ this context — so the openrouter_providers.json
        lookup drives it. Shared by fast mode (vision endpoint) and agent
        self-video-analysis (agent orchestrator endpoint); the caller owns the
        'attach once' guard, this only decides + inserts."""
        from ambient import fast_mode as fm
        from ambient.media_plan import plan_media

        # Duration drives the per-step sampling budget (frames = fps x duration,
        # capped). Youtube imports carry it under "youtube"; uploads carry a top-level
        # "duration" probed at ingest (store_video). 0 -> tools probe for it.
        duration = 0.0
        try:
            rec = await self.store.get_file(self.video_id)
            if rec:
                duration = float(
                    (rec.get("youtube") or {}).get("duration")
                    or rec.get("duration") or 0
                ) or 0.0
        except Exception:  # noqa: BLE001
            pass

        plan = plan_media("context", base_url, model, duration)

        # Dense-video path (endpoint samples internally): send the source URL — no
        # sandbox, no frame upload (~3-4x faster than the e2b frame pipeline).
        if plan.method == "video":
            print(f"[plan_media] attaching video context for session {self.session_id}, video_id: {self.video_id} , duration: {duration}")
            url = await fm.source_video_url(self.video_id, self.store)
            if url:
                self.messages.insert(1, fm.build_fast_video_message(url, video_id=self.video_id))
                log.info(f"[video_context] session {self.session_id}: video_url "
                         f"(fps~{plan.fps:.2f}, <={plan.max_frames}f)")
                return
            # No S3 URL to host the video -> force the frames path instead.
            plan = plan_media("context", base_url, model, duration, force_frames=True)

        # Frames path (endpoints that can't control video sampling / image-only
        # models): uniformly sample and attach as timestamp-labeled image blocks.
        # MediaPlan is duck-compatible with sample_fast_frames (.max_frames+.max_dim).
        frames, _dur = await fm.sample_fast_frames(self.video_id, self._media_box, plan,
                                                   duration=duration or None)
        print(f"[plan_media] attaching frames context for session {self.session_id}, Number of frames: {len(frames)}")

        self.messages.insert(1, fm.build_fast_frames_message(frames))
        log.info(f"[video_context] session {self.session_id}: sampled {len(frames)} frames "
                 f"(cap {plan.max_frames})")

    async def _ensure_fast_frames(self) -> None:
        """Fast mode: attach the whole-video context once, reused across follow-up
        turns. Fast mode is a direct vision call, so it classifies the LLM (vision)
        endpoint — not the agent orchestrator."""
        if self._fast_frames_loaded:
            return
        # Media tools / the video URL read the source from S3 for youtube imports.
        await self._await_source_ready()
        await self._attach_video_context(settings.llm_base_url, self.model)
        self._fast_frames_loaded = True

    async def _run_fast(self, model: str) -> None:
        run_id = f"run_{uuid.uuid4().hex[:12]}"
        await self._set_status("running")
        await self._emit("run.started", {"run_id": run_id, "model": model})
        await self._seed_usage_once()
        self.usage["runs"] = int(self.usage.get("runs", 0)) + 1

        final_answer = ""
        stop_reason = "end_turn"
        try:
            # Sample + insert the frame set on the first run (streamed, so the UI
            # shows activity); reused across follow-up turns.
            if not self._fast_frames_loaded:
                await self._emit("tool.progress", {"run_id": run_id, "turn": 1,
                                                   "progress": {"message": "sampling frames"}})
                await self._ensure_fast_frames()

            response_format = None
            if self.output_structure:
                from ambient.fast_mode import build_response_format
                response_format = build_response_format("answer", self.output_structure)

            await self.broker.renew(self.session_id)
            prepared = _normalize_for_openai(self.messages)
            # Disable thinking so the model emits the answer directly rather than
            # burning the budget on a reasoning trace (covers OpenRouter via
            # reasoning_enabled=False and vLLM via chat_template_kwargs).
            reasoning_enabled = not settings.fast_disable_thinking
            extra_body = ({"chat_template_kwargs": {"enable_thinking": False}}
                          if settings.fast_disable_thinking else None)
            chunks: list[dict] = []
            try:
                async for chunk in stream_chat_completion(
                    model=model, messages=prepared, tools=[], response_format=response_format,
                    reasoning_enabled=reasoning_enabled, max_tokens=settings.fast_max_tokens,
                    extra_body=extra_body,
                    # Fast mode is a direct vision call -> use the LLM (vision)
                    # endpoint, not the agent-orchestrator endpoint.
                    base_url=settings.llm_base_url, api_key=settings.llm_api_key,
                ):
                    chunks.append(chunk)
                    await self._emit("chat.completion.chunk", chunk)
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001
                await self._emit("error", {"code": "llm_error", "message": str(exc), "fatal": False})
                stop_reason = "error"
                chunks = []

            if chunks:
                assistant_msg, _finish, _tools, reasoning, agent_usage = assemble_assistant_message(chunks)
                self.messages.append(assistant_msg)
                self.usage["turns"] = int(self.usage.get("turns", 0)) + 1
                await self._record_usage(run_id, 1, model, agent_usage, "agent")
                if settings.expose_thinking and reasoning:
                    await self._emit("agent.reasoning", {"run_id": run_id, "turn": 1, "reasoning": reasoning})
                final_answer = self._extract_answer_text(assistant_msg)

            await self._emit("run.completed", {
                "run_id": run_id, "stop_reason": stop_reason,
                "answer": final_answer, "usage": self.usage,
            })
        except asyncio.CancelledError:
            await self._emit("run.completed", {"run_id": run_id, "stop_reason": "cancelled", "answer": ""})
        finally:
            await self.store.save_messages(self.session_id, self.messages)
            await self._persist_usage()
            await self._set_status("ready")
            await self.broker.release(self.session_id)

    async def _run_tool_calls(self, run_id: str, turn: int, tool_uses: list[dict]) -> None:
        # Media tools read the source from S3; for a fresh youtube import that
        # upload may still be in flight (the description no longer gates readiness).
        # Wait for it once, telling the client why (only if we actually block).
        async def _on_wait() -> None:
            await self._emit("tool.progress", {
                "run_id": run_id,
                "turn": turn,
                "progress": {"message": "waiting for video import to finish"},
            })

        await self._await_source_ready(on_wait=_on_wait)

        tool_uses_by_id = {b["id"]: b["name"] for b in tool_uses}

        async def _run_one(block: dict) -> tuple[str, dict | None, dict | None]:
            tool_use_id = block["id"]
            name = block["name"]
            tool_input = block.get("input") or {}
            await self._emit("tool.scheduled", {
                "run_id": run_id,
                "turn": turn,
                "tool_use_id": tool_use_id,
                "name": name,
                "input": tool_input,
            })
            result_data: dict | None = None
            failure: dict | None = None
            async for ev in self._dispatcher.call_tool(tool_use_id, name, tool_input):
                if ev.type == "started":
                    await self._emit("tool.started", {
                        "run_id": run_id,
                        "turn": turn,
                        "tool_use_id": tool_use_id,
                    })
                elif ev.type == "progress":
                    await self._emit("tool.progress", {
                        "run_id": run_id,
                        "turn": turn,
                        "tool_use_id": tool_use_id,
                        "progress": ev.data,
                    })
                elif ev.type == "result":
                    result_data = ev.data
                elif ev.type == "failed":
                    failure = ev.data
            return tool_use_id, result_data, failure

        results = await asyncio.gather(*(_run_one(b) for b in tool_uses))

        for tool_use_id, result_data, failure in results:
            if failure is not None:
                await self._emit("tool.failed", {
                    "run_id": run_id,
                    "turn": turn,
                    "tool_use_id": tool_use_id,
                    "error": failure,
                })
                self.messages.append({
                    "role": "user",
                    "content": [{
                        "type": "tool_result",
                        "tool_use_id": tool_use_id,
                        "content": f"Tool failed: {failure.get('message', 'unknown error')}",
                    }],
                })
                continue

            data = result_data or {}
            analysis = data.get("analysis", "")
            attachments = data.get("attachments") or []
            user_message_contents = data.get("user_message_contents") or []

            # Attribute any LLM usage this tool incurred (e.g. clip/frame analysis)
            # to the session totals, keyed by tool name.
            for rec in data.get("usage") or []:
                await self._record_usage(
                    run_id, turn, rec.get("model"), rec.get("usage"),
                    f"tool:{tool_uses_by_id.get(tool_use_id, 'tool')}",
                )

            await self._emit("tool.result", {
                "run_id": run_id,
                "turn": turn,
                "tool_use_id": tool_use_id,
                "analysis": analysis,
                "attachments": attachments,
            })

            self.messages.append({
                "role": "user",
                "content": [{
                    "type": "tool_result",
                    "tool_use_id": tool_use_id,
                    "content": analysis,
                }],
            })
            if user_message_contents:
                self.messages.append({"role": "user", "content": user_message_contents})

    # --- helpers ----------------------------------------------------------

    async def _build_seed_text(self, attach_video_description: bool = False) -> str:
        """First-turn context. Uses the full description if it's already resolved,
        otherwise a metadata-only seed so the turn never blocks on ingestion."""
        
        title = duration = None
        try:
            rec = await self.store.get_file(self.video_id)
        except Exception:  # noqa: BLE001 - direct video id / store hiccup
            rec = None
        if rec:
            yt = rec.get("youtube") or {}
            title = yt.get("title") or rec.get("filename")
            duration = yt.get("duration") or rec.get("duration")

        if duration is None and self.backend != "e2b":
            from ambient.tools.rangeproxy_video_tools import RangeProxyVideoFrameTools
            duration = await asyncio.to_thread(
                lambda: RangeProxyVideoFrameTools(self.video_id).probe_duration()
            )

            if duration is not None and rec is not None:
                try:
                    await self.store.update_file(self.video_id, lambda r: {**r, "duration": duration})
                except Exception:  # noqa: BLE001 - caching is best-effort
                    pass
        
        seed_text = (
            f"Video id: {self.video_id}\n"
            f"Title: {title or 'unknown'}\n"
            f"Duration: {duration if duration is not None else 'unknown'} seconds\n"   
        )

        if self.video_description:
            seed_text += f"\nDescription: {self.video_description}\n\n"

        if attach_video_description:
            seed_text += "Note: a detailed video description is being generated in the background and will be provided in a later message. You can use your video tools immediately." + "\n"
        
        source_hint = self._video_source_hint(await self.store.get_file(self.video_id))
        if source_hint:
            seed_text += f"\n{source_hint}\n"

        return seed_text


    def _description_injected(self) -> bool:
        """True if the full description is already in the conversation — either as
        the first-turn seed (`Video id: ..., Description: ...`) or a later
        `[system update]` injection. Scans `self.messages` so it survives
        rehydration (a rebuilt runner has `video_description=None`)."""
        for message in self.messages:
            if message.get("role") != "user":
                continue
            content = message.get("content")
            if not isinstance(content, list):
                continue
            for block in content:
                if not isinstance(block, dict) or block.get("type") != "text":
                    continue
                text = block.get("text", "")
                if text.startswith("Video id:") and "Description:" in text:
                    return True
                if self._DESC_INJECT_MARKER in text:
                    return True
        return False

    async def _await_description(self, timeout: float) -> None:
        """Wait (bounded) for the background description task to finish.

        Used on the first run so the first answer is grounded in the description.
        Returns immediately if it's already resolved, never started, or already
        done (including failed). On timeout the description task keeps running
        (shielded) and is injected on a later turn instead."""
        task = self._description_task
        if self.video_description is not None or task is None or task.done():
            return
        try:
            await asyncio.wait_for(asyncio.shield(task), timeout)
        except asyncio.TimeoutError:
            pass
        except Exception:  # noqa: BLE001 - resolver reports its own failure
            pass

    async def _await_source_ready(self, on_wait=None) -> None:
        """Block until a youtube source MP4 is on S3 (uploaded by the ingest job).

        With the description off the critical path a session can start before its
        source is uploaded; media tools would then fetch a missing S3 object. This
        waits cleanly instead. Non-youtube sources (direct uploads) are ready
        immediately. `on_wait` (if given) is awaited once, only when we actually
        have to block. Cached after the first success."""
        if self._source_ready:
            return
        try:
            rec = await self.store.get_file(self.video_id)
        except Exception:  # noqa: BLE001
            rec = None
        if not rec or rec.get("source_type") != "youtube" or rec.get("source_status") == "ready":
            self._source_ready = True
            return
        if on_wait is not None:
            await on_wait()
        while rec.get("source_status") != "ready":
            if rec.get("source_status") == "failed":
                raise RuntimeError(f"video source failed: {rec.get('source_error')}")
            await asyncio.sleep(2)
            rec = await self.store.get_file(self.video_id)
        self._source_ready = True

    def _extract_answer_text(self, assistant_msg: dict) -> str:
        parts: list[str] = []
        for block in assistant_msg.get("content") or []:
            if isinstance(block, dict) and block.get("type") == "text":
                t = block.get("text", "")
                if t and not t.startswith("<think>"):
                    parts.append(t)
        return "\n".join(parts).strip()

    async def _seed_usage_once(self) -> None:
        """Load the persisted usage aggregate on the first run so accumulation
        survives rehydration on another worker."""
        if self._usage_seeded:
            return
        self._usage_seeded = True
        try:
            record = await self.store.get_session(self.session_id)
        except Exception:  # noqa: BLE001
            record = None
        persisted = (record or {}).get("usage")
        if isinstance(persisted, dict) and persisted.get("requests") is not None:
            # Merge onto the empty template so any newly added fields exist.
            merged = _empty_usage()
            merged.update(persisted)
            self.usage = merged

    async def _record_usage(
        self, run_id: str, turn: int, model: str, raw_usage: Optional[dict], source: str
    ) -> None:
        """Attribute one LLM request's usage to the session totals and emit a
        `span.model_request_end` event carrying its per-request token usage."""
        if not settings.track_usage or not raw_usage:
            return
        norm = normalize_usage(raw_usage)
        # Prefer the provider-reported cost (OpenRouter returns actual USD in the
        # usage block); fall back to the catalog estimate when it's absent.
        cost = norm.pop("cost", None)
        if cost is None:
            cost = estimate_cost(model, norm)
        # Totals.
        _add_usage_bucket(self.usage, norm, cost)
        # Rollups by source (agent / tool:<name>) and by model.
        by_source = self.usage.setdefault("by_source", {})
        _add_usage_bucket(by_source.setdefault(source, {}), norm, cost)
        by_model = self.usage.setdefault("by_model", {})
        _add_usage_bucket(by_model.setdefault(model or "unknown", {}), norm, cost)

        await self._emit("span.model_request_end", {
            "run_id": run_id,
            "turn": turn,
            "source": source,
            "model": model,
            "usage": norm,
            "cost": cost,
        })

    async def _persist_usage(self) -> None:
        if not settings.track_usage:
            return
        snapshot = dict(self.usage)
        try:
            await self.store.update_session(
                self.session_id, lambda r: {**r, "usage": snapshot}
            )
        except KeyError:
            pass

    async def _set_status(self, status: str) -> None:
        prev = self._status
        if prev == status:
            return
        self._status = status
        try:
            await self.store.update_session(self.session_id, lambda r: {**r, "status": status})
        except KeyError:
            return
        if status != "running":  # already emitted run.started/turn.* for running
            await self._emit("session.status_changed", {
                "status": status,
                "previous_status": prev,
            })

    async def _emit(self, type: str, payload: dict[str, Any]) -> None:
        ev = await self.store.append_event(self.session_id, type, payload)
        # Fan out to SSE subscribers on any worker via Redis pub/sub. The store
        # write above is the durable record; this is the live push.
        await self.broker.publish(self.session_id, ev)

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
from typing import Any, Optional

from ambient.config import settings
from ambient.prompt import SYSTEM_PROMPT
from ambient.server.broker import Broker
from ambient.server.llm_client import assemble_assistant_message, stream_chat_completion
from ambient.server.sandbox import SandboxLimits, ToolDispatcher
from ambient.server.store import Store
from ambient.tools import TOOLS
from ambient.server.ingest import ensure_description
import logging

log = logging.getLogger(__name__)

MAX_CLIPS = 5
MAX_FRAMES = 30


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
    ):
        self.session_id = session_id
        self.video_id = video_id
        self.video_description = None
        self.model = model
        self.max_turns_per_run = max_turns_per_run
        self.backend = backend
        self.limits = limits
        self.store = store
        self.broker = broker

        # Tools always run in-process; only VideoFrameTools' media ops vary by
        # backend. For "e2b" the runner owns a per-session media sandbox and
        # hands it to the dispatcher.
        self._dispatcher = ToolDispatcher()
        self._media_box: Optional[object] = None
        self.sandbox_id: Optional[str] = None

        # On a fresh session `messages` is None -> empty list; seed is built in
        # start_sandbox() after the description is ready. On rehydration the
        # caller passes the persisted list (already includes the seeded turns).
        self.messages: list[dict] = messages or self._get_seed_messages()
        self._run_task: Optional[asyncio.Task] = None
        self._run_lock = asyncio.Lock()
        self._status = "created"
        self._cancel_requested = False

    # --- lifecycle --------------------------------------------------------

    def _get_seed_messages(self) -> None:
        return [
            {"role": "system", "content": SYSTEM_PROMPT}
        ]
        

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
        
        try:
            self.video_description = await ensure_description(self.store, self.video_id, box=self._media_box)
            log.info(f"Description for video {self.video_id} is {self.video_description}")
        except Exception as exc:  # noqa: BLE001
            log.error(f"Error ensuring description for video {self.video_id}: {str(exc)}")
            await self._emit("error", {"code": "video_description_failed", "message": str(exc), "fatal": True})
            await self._set_status("failed")
            raise
        
        boot_ms = int((time.monotonic() - t0) * 1000)
        await self._emit("session.status_changed", {
            "status": "ready",
            "previous_status": "starting",
            "sandbox": {"status": "ready", "id": self.sandbox_id, "boot_ms": boot_ms, "region": None},
        })
        self._status = "ready"

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

    async def terminate(self) -> None:
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
        text = content
        if output_structure:
            # Mirror ambient.agent.run_agent: steer the final answer to the schema.
            text = (
                f"{content}\n\nYour final answer must strictly follow this JSON schema:\n"
                f"{json.dumps(output_structure)}"
            )

        if len(self.messages) == 1:
            log.info(f"Waiting for video description for video {self.video_id}")
            while self.video_description is None:
                self.video_description = await ensure_description(self.store, self.video_id, box=self._media_box)
                await asyncio.sleep(1)
            log.info(f"Got video description for video {self.video_id}")
            self.messages.append({"role": "user", "content": [{"type": "text", "text": f"Video id: {self.video_id}, Description: {self.video_description}"}]})
        self.messages.append({"role": "user", "content": [{"type": "text", "text": text}]})
        # Persist the user turn up front so a rehydrating worker sees it even if
        # this run is interrupted before producing an assistant turn.
        await self.store.save_messages(self.session_id, self.messages)
        model = model_override or self.model
        self._cancel_requested = False
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
        try:
            for turn in range(1, self.max_turns_per_run + 1):
                # Keep the ownership lease alive for the duration of the run.
                await self.broker.renew(self.session_id)
                await self._emit("turn.start", {"run_id": run_id, "turn": turn})

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

                assistant_msg, finish_reason, tool_uses = assemble_assistant_message(chunks)
                self.messages.append(assistant_msg)
                await self._emit("turn.end", {
                    "run_id": run_id,
                    "turn": turn,
                    "finish_reason": finish_reason or "stop",
                })

                if finish_reason in (None, "stop", "length"):
                    final_answer = self._extract_answer_text(assistant_msg)
                    stop_reason = "end_turn"
                    break

                if finish_reason == "tool_calls" and tool_uses:
                    await self._run_tool_calls(run_id, turn, tool_uses)
                    # Persist the grown context so another worker can rehydrate it
                    # mid-conversation.
                    await self.store.save_messages(self.session_id, self.messages)
                else:
                    stop_reason = "end_turn"
                    break
            else:
                stop_reason = "max_turns"

            await self._emit("run.completed", {
                "run_id": run_id,
                "stop_reason": stop_reason,
                "answer": final_answer,
            })
        except asyncio.CancelledError:
            await self._emit("run.completed", {
                "run_id": run_id,
                "stop_reason": "cancelled",
                "answer": "",
            })
        finally:
            await self.store.save_messages(self.session_id, self.messages)
            await self._set_status("ready")
            # Release the run-ownership lease so any worker can serve the next turn.
            await self.broker.release(self.session_id)

    async def _run_tool_calls(self, run_id: str, turn: int, tool_uses: list[dict]) -> None:
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

    def _extract_answer_text(self, assistant_msg: dict) -> str:
        parts: list[str] = []
        for block in assistant_msg.get("content") or []:
            if isinstance(block, dict) and block.get("type") == "text":
                t = block.get("text", "")
                if t and not t.startswith("<think>"):
                    parts.append(t)
        return "\n".join(parts).strip()

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

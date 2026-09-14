"""In-process tool dispatcher.

Tools always run in the server process. Each call is offloaded to a worker
thread running its own event loop, so blocking media work (when the inprocess
video backend shells out to ffmpeg) and aiohttp LLM calls don't stall the
server's event loop or any live SSE stream. Concurrent tool calls in a turn run
on separate threads, mirroring `ambient.agent.run_agent`.

The dispatcher publishes the per-session E2B media box (if any) to the tools via
the `current_media_box` ContextVar before running each tool. `asyncio.to_thread`
copies the current context into the worker thread, so `make_video_tools` reads
the right box. For the `inprocess` backend the box is None and ignored.
"""
from __future__ import annotations

import asyncio
import inspect
import traceback
from typing import Any, AsyncIterator, Optional

from ambient.server.sandbox.interface import ToolEvent
from ambient.tools.video_backend import current_media_box
from ambient.llm import usage_sink


def _run_tool_blocking(func, tool_input: dict[str, Any]) -> Any:
    """Run an async tool to completion in a fresh event loop on this thread."""
    return asyncio.run(func(**tool_input))


def _normalize_tool_result(raw: Any) -> dict[str, Any]:
    """Tools today return either (analysis, user_message_contents) or a dict."""
    if isinstance(raw, tuple) and len(raw) == 2:
        analysis, user_contents = raw
        attachments: list[dict[str, Any]] = []
        for block in user_contents or []:
            if not isinstance(block, dict):
                continue
            btype = block.get("type")
            if btype == "image_url":
                url = (block.get("image_url") or {}).get("url")
                if url:
                    attachments.append({"type": "image_url", "url": url})
            elif btype == "video_url":
                url = (block.get("video_url") or {}).get("url")
                if url:
                    attachments.append({"type": "video_url", "url": url})
        return {
            "analysis": analysis,
            "attachments": attachments,
            "user_message_contents": user_contents or [],
        }
    if isinstance(raw, dict):
        raw.setdefault("attachments", [])
        raw.setdefault("user_message_contents", [])
        return raw
    return {"analysis": str(raw), "attachments": [], "user_message_contents": []}


class ToolDispatcher:
    """Dispatches agent tools by name, caching the video description per video_id."""

    def __init__(self, video_id: Optional[str] = None) -> None:
        # The per-session E2B media box, set by the runner for the "e2b" backend.
        # None for "inprocess".
        self.media_box: Optional[object] = None
        # The session's real video_id. Forced onto every tool call so the model can't
        # mis-supply it (some models pass a placeholder like "video" or mangle the
        # opaque id) or target a different video.
        self.session_video_id: Optional[str] = video_id
        # Per-session cache of the high-level video description, keyed by video_id.
        # get_video_description fills it; search_clip / focus_clip receive it as
        # `video_description` on later calls (mirrors ambient.agent.execute_tool).
        self._video_description: dict[str, str] = {}

    async def call_tool(
        self,
        tool_use_id: str,
        name: str,
        input: dict[str, Any],
    ) -> AsyncIterator[ToolEvent]:
        from ambient.tools import TOOL_REGISTRY  # lazy: keep import cost off boot path

        func = TOOL_REGISTRY.get(name)
        if func is None:
            yield ToolEvent(
                type="failed",
                tool_use_id=tool_use_id,
                name=name,
                data={"code": "unknown_tool", "message": f"no tool named {name!r}"},
            )
            return

        yield ToolEvent(type="started", tool_use_id=tool_use_id, name=name)

        # Publish the media box for the worker thread (to_thread copies this context).
        token = current_media_box.set(self.media_box)
        # Collect any LLM usage this tool incurs. `to_thread` copies the context,
        # so llm_call (running off-thread) appends into this same list object.
        usage_records: list = []
        usage_token = usage_sink.set(usage_records)
        try:
            tool_input = dict(input or {})
            func_args = inspect.getfullargspec(func).args

            # video_id is session state, not the model's to choose: force the session's
            # real id onto any tool that takes one. Models sometimes pass a placeholder
            # ("video") or mangle the opaque id (breaking source resolution), and this
            # also stops a tool call from targeting a different video. Fall back to the
            # model-supplied value only when we have no session id (older callers).
            if self.session_video_id and "video_id" in func_args:
                tool_input["video_id"] = self.session_video_id
            video_id = tool_input.get("video_id")

            # Inject the cached video description for tools that accept it.
            if name != "get_video_description":
                if (
                    "video_description" in func_args
                    and video_id
                    and self._video_description.get(video_id)
                    and "video_description" not in tool_input
                ):
                    tool_input["video_description"] = self._video_description[video_id]

            raw = await asyncio.to_thread(_run_tool_blocking, func, tool_input)

            # Cache the description so later clip tools can reuse it.
            if name == "get_video_description":
                description = raw[0] if isinstance(raw, tuple) else raw
                if video_id and description:
                    self._video_description[video_id] = description

            result = _normalize_tool_result(raw)
            # Attach the LLM usage this tool incurred so the runner can attribute
            # tool-side token/cost to the session totals.
            if usage_records:
                result["usage"] = list(usage_records)
            yield ToolEvent(
                type="result",
                tool_use_id=tool_use_id,
                name=name,
                data=result,
            )
        except Exception as exc:  # noqa: BLE001 - surfaced to the runner as tool.failed
            yield ToolEvent(
                type="failed",
                tool_use_id=tool_use_id,
                name=name,
                data={
                    "code": "tool_exception",
                    "message": str(exc),
                    "traceback": traceback.format_exc(),
                },
            )
        finally:
            current_media_box.reset(token)
            usage_sink.reset(usage_token)

"""Selects where VideoFrameTools' media ops run.

`settings.sandbox_backend`:
  "inprocess" -> local ffmpeg/decord (ambient.tools.video_tools.VideoFrameTools)
  "e2b"       -> an E2B media sandbox (ambient.tools.video_tools_e2b.E2BVideoFrameTools)

An active `current_media_box` always wins. The background ingest worker may boot
an ephemeral E2B sandbox to materialize remote sources even when the interactive
session backend is configured as "inprocess".

For the "e2b" backend the per-session sandbox is owned by the SessionRunner and
published to tools via `current_media_box` (the ToolDispatcher sets it before
running each tool; asyncio.to_thread copies the context into the worker thread).
"""
from __future__ import annotations

from contextvars import ContextVar
from typing import Optional

from ambient.config import settings

# Holds the active MediaSandbox for the current tool call (None for inprocess).
current_media_box: ContextVar[Optional[object]] = ContextVar("current_media_box", default=None)


def make_video_tools(video_id: str, max_frame_dimention: Optional[int] = None):
    """Return the VideoFrameTools implementation for the configured backend."""
    box = current_media_box.get()
    if box is not None:
        from ambient.tools.sandbox_video_tools import SandboxVideoFrameTools

        return SandboxVideoFrameTools(video_id, max_frame_dimention, box=box)

    if settings.sandbox_backend == "e2b":
        if box is None:
            raise RuntimeError(
                "sandbox_backend='e2b' but no media sandbox is active for this call; "
                "the runner must create it in start_sandbox()."
            )
        from ambient.tools.sandbox_video_tools import SandboxVideoFrameTools

        return SandboxVideoFrameTools(video_id, max_frame_dimention, box=box)

    from ambient.tools.inprocess_video_tools import VideoFrameTools

    return VideoFrameTools(video_id, max_frame_dimention=max_frame_dimention)

"""Selects where the video tools' media ops run.

Two backends:
  host (default) -> ffmpeg on the host, in-process, via the range proxy
                    (ambient.tools.rangeproxy_video_tools.RangeProxyVideoFrameTools):
                    reads a locally-cached source file when present, else HTTP
                    range-reads the S3/R2 source. No sandbox, no tiling.
  e2b            -> media ops run in an E2B sandbox
                    (ambient.tools.sandbox_video_tools.SandboxVideoFrameTools).

An active `current_media_box` always wins: the background ingest worker boots an
ephemeral E2B sandbox to materialize/describe remote sources even when the
interactive session backend is the host range proxy. For the "e2b" backend the
per-session sandbox is owned by the SessionRunner and published to tools via
`current_media_box` (the ToolDispatcher sets it before running each tool;
asyncio.to_thread copies the context into the worker thread).
"""
from __future__ import annotations

from contextvars import ContextVar
from typing import Optional

from ambient.config import settings

# Holds the active MediaSandbox for the current tool call (None -> host backend).
current_media_box: ContextVar[Optional[object]] = ContextVar("current_media_box", default=None)


def make_video_tools(video_id: str, max_frame_dimention: Optional[int] = None,
                     duration: Optional[float] = None):
    """Return the video-tools implementation for the active call.

    An active `current_media_box` (or `sandbox_backend='e2b'`) routes to the E2B
    sandbox backend; otherwise media runs on the host via the range proxy.

    `duration` (seconds), when known from the file record, is handed to the range
    proxy so it skips probing the source's duration.
    """
    box = current_media_box.get()
    if box is not None:
        from ambient.tools.sandbox_video_tools import SandboxVideoFrameTools

        return SandboxVideoFrameTools(video_id, max_frame_dimention, box=box)

    if settings.sandbox_backend == "e2b":
        raise RuntimeError(
            "sandbox_backend='e2b' but no media sandbox is active for this call; "
            "the runner must create it in start_sandbox()."
        )

    from ambient.tools.rangeproxy_video_tools import RangeProxyVideoFrameTools

    return RangeProxyVideoFrameTools(video_id, max_frame_dimention, duration=duration)

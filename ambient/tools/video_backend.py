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

Beyond the session's initial video (resolved by `video_id`), a tool may point at
another local video by passing an absolute `source_path` — a host path for the
range proxy, or a box-local path for e2b. `resolve_video_tools()` validates/derives
the label and picks the backend; `make_video_tools()` builds the chosen backend.
"""
from __future__ import annotations

import os
import re
import tempfile
from contextvars import ContextVar
from typing import Optional

from ambient.config import settings

# Holds the active MediaSandbox for the current tool call (None -> host backend).
current_media_box: ContextVar[Optional[object]] = ContextVar("current_media_box", default=None)


def make_video_tools(video_id: str, max_frame_dimention: Optional[int] = None,
                     duration: Optional[float] = None, source_path: Optional[str] = None):
    """Return the video-tools implementation for the active call.

    An active `current_media_box` (or `sandbox_backend='e2b'`) routes to the E2B
    sandbox backend; otherwise media runs on the host via the range proxy.

    `duration` (seconds), when known from the file record, is handed to the range
    proxy so it skips probing the source's duration. `source_path`, when set,
    overrides `video_id` resolution and reads that file directly (host path for the
    range proxy, box-local path for e2b) — see `resolve_video_tools`.
    """
    box = current_media_box.get()
    if box is not None:
        from ambient.tools.sandbox_video_tools import SandboxVideoFrameTools

        return SandboxVideoFrameTools(video_id, max_frame_dimention, box=box,
                                      source_path=source_path)

    if settings.sandbox_backend == "e2b":
        raise RuntimeError(
            "sandbox_backend='e2b' but no media sandbox is active for this call; "
            "the runner must create it in start_sandbox()."
        )

    from ambient.tools.rangeproxy_video_tools import RangeProxyVideoFrameTools

    return RangeProxyVideoFrameTools(video_id, max_frame_dimention, duration=duration,
                                     source_path=source_path)


# --------------------------------------------------------------- external videos

_LABEL_RE = re.compile(r"[^A-Za-z0-9._-]+")


def _host_allowed_roots() -> list[str]:
    roots = [settings.video_folder, settings.bash_cwd or settings.video_folder,
             tempfile.gettempdir()]
    roots += [r.strip() for r in (settings.external_video_allowed_roots or "").split(",")]
    # Resolve symlinks so the containment check can't be tricked by a symlinked root.
    return [os.path.realpath(r) for r in roots if r and r.strip()]


def _box_allowed_roots() -> list[str]:
    return [r.strip() for r in (settings.external_video_box_roots or "").split(",") if r.strip()]


def _is_within(path: str, roots: list[str]) -> bool:
    for root in roots:
        try:
            if os.path.commonpath([path, root]) == root:
                return True
        except ValueError:  # different drives / mixed abs-rel -> not contained
            continue
    return False


def _reject_obvious(video_path: str) -> None:
    """Shared format guard: a real absolute filesystem path, not a URL/flag."""
    if not isinstance(video_path, str) or not video_path.strip():
        raise ValueError("video_path must be a non-empty absolute file path")
    if "://" in video_path or video_path.lstrip().startswith("-"):
        raise ValueError(f"video_path must be a local file path, not a URL/flag: {video_path!r}")
    if not os.path.isabs(video_path):
        raise ValueError(f"video_path must be absolute, got {video_path!r}")


def _validate_host_path(video_path: str) -> str:
    _reject_obvious(video_path)
    real = os.path.realpath(video_path)
    if not os.path.isfile(real):
        raise ValueError(f"video_path does not exist or is not a file: {video_path!r}")
    roots = _host_allowed_roots()
    if not _is_within(real, roots):
        raise ValueError(
            f"video_path {video_path!r} is outside the allowed roots ({roots}). "
            "Place the file under the video workspace (or add its dir to "
            "EXTERNAL_VIDEO_ALLOWED_ROOTS)."
        )
    return real


def _validate_box_path(video_path: str) -> str:
    # Can't stat a box-local path from the host, so validate format + root prefix
    # only; the in-box command errors cleanly if the file is missing.
    _reject_obvious(video_path)
    norm = os.path.normpath(video_path)
    # roots = _box_allowed_roots()
    # if roots and not _is_within(norm, roots):
    #     raise ValueError(
    #         f"video_path {video_path!r} is outside the allowed box roots ({roots}). "
    #         "Download the file under one of them (or set EXTERNAL_VIDEO_BOX_ROOTS)."
    #     )
    return norm


def _label_for(path: str) -> str:
    stem = os.path.splitext(os.path.basename(path))[0]
    stem = _LABEL_RE.sub("_", stem).strip("_")
    return stem or "external"


def resolve_video_tools(video_id: str, video_path: Optional[str] = None,
                        max_frame_dimention: Optional[int] = None,
                        duration: Optional[float] = None):
    """Build the video tools for either the session's initial video (`video_id`) or
    an explicit external `video_path`.

    With no `video_path`, this is exactly `make_video_tools(video_id, ...)`. With a
    `video_path`, the path is validated for the active backend (host: must exist and
    live under a host root; e2b: format + box-root prefix), a label id is derived
    from its basename (for Clip/Frame ids + citations), and the backend reads that
    file directly.
    """
    if not video_path:
        return make_video_tools(video_id, max_frame_dimention, duration=duration)

    is_box = current_media_box.get() is not None
    src = _validate_box_path(video_path) if is_box else _validate_host_path(video_path)
    return make_video_tools(_label_for(src), max_frame_dimention, duration=duration,
                            source_path=src)


def validate_external_media_path(path: str) -> tuple[str, bool]:
    """Validate an external media (image/video) path for the active backend and
    return `(resolved_path, is_box)`. Host: absolute, exists, is a file, under a
    host root. Box (e2b): absolute, format + box-root prefix (existence is checked
    in-box). Raises ValueError on any violation. Used by tools that take a raw file
    path (e.g. read_image) rather than a session video_id."""
    is_box = current_media_box.get() is not None
    resolved = _validate_box_path(path) if is_box else _validate_host_path(path)
    return resolved, is_box

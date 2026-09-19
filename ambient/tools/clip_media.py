"""Realize a clip window as either video clips or image frames, per media_plan.

`search_clip` and `focus_clip` both analyze a `[start, end]` window with a vision
call. Whether that window is delivered as `video_url` clips (self-hosted vLLM
samples the video internally — cheapest via temporal merge) or as `image_url`
frames (external providers re-sample any `video_url` to a fixed budget we can't
control, so frames are the only way to pin the count) is decided ONCE by
`media_plan.plan_media`. This module is the shared realizer so that decision — and
the integer-fps sampling grid it implies — lives in exactly one place instead of
being duplicated (and drifting) across the two tools.

Behavior is unchanged on self-hosted endpoints (they get the video path at the
same fps as before); the frames path is additive, taken only where the endpoint
can't control a video's frame count.
"""
from __future__ import annotations

import asyncio
import math
from typing import List, Literal, Tuple

from ambient.config import settings, get_provider_quality_settings
from ambient.media_plan import MediaPlan, plan_media


async def produce_window_media(
    video_tools,
    step: str,
    start_time: float,
    end_time: float,
    *,
    model: str | None = None,
    base_url: str | None = None,
) -> Tuple[Literal["clips", "frames"], list, float, MediaPlan]:
    """Produce the media for a `[start, end]` window per `plan_media(step, ...)`.

    Returns `(kind, media, clip_start, plan)`:
      * kind="clips"  -> `media` is a list of Clip (render as `video_url` blocks).
      * kind="frames" -> `media` is a list of Frame (render as `image_url` blocks).
      * clip_start is the window's global start, so a model's local mm:ss citations
        map back to absolute video time by adding it.

    `step` is the media_plan key ("search_clip" / "focus_clip"). `model`/`base_url`
    default to the vision endpoint (`settings.llm_model` / `settings.llm_base_url`),
    the endpoint these clips are actually analyzed on.
    """
    q = get_provider_quality_settings(settings.llm_model)
    model = model or settings.llm_model
    base_url = base_url if base_url is not None else settings.llm_base_url
    span = max(float(end_time) - float(start_time), 0.1)
    plan = plan_media(step, base_url, model, span)

    if plan.method == "frames":
        # External / image-only: send a controlled number of frames. fetch_frames
        # wants an integer fps >= 1 and downsamples to max_frames, so pick the
        # smallest integer grid that yields >= n candidates (mirrors fast mode).
        n = plan.frames_estimate(span)
        grid_fps = max(1, math.ceil(n / span))
        frames = await asyncio.to_thread(
            video_tools.fetch_frames, grid_fps, float(start_time), float(end_time), n
        )
        return "frames", frames, float(start_time), plan

    # Video path (self-hosted vLLM samples internally): unchanged from before —
    # same fps (plan.fps == analysis_fps for these steps) + quality caps.
    clips, clip_start = await video_tools.fetch_clips(
        start_time,
        end_time,
        fps=max(1, int(round(plan.fps))),
        crf=q.crf,
        max_size_mb=q.max_size_mb,
    )
    return "clips", clips, float(clip_start), plan

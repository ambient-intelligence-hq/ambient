"""Measured provider capabilities (vendored from vlm.fyi), resolved per endpoint + model.

`ambient/artifacts/capabilities.json` is a pinned copy of vlm.fyi's dataset
(refresh with scripts/sync_capabilities.py). Each entry is one model on one
provider: images per request, real-size frames per request, byte limits, whether a
video reaches the model and how many frames of it, request fields the endpoint
rejects.

`resolve(base_url, model)` matches the session's endpoint (service from the base
URL; on OpenRouter, the pinned upstream from openrouter_providers.json) and model
(core key: org prefix, case, quant suffix and Fireworks "4p1" versions ignored).
It returns None when nothing was measured, and callers keep their old behavior.

Values are what the provider accepted. A limit the probe actually hit gets `MARGIN`,
because limits drift and other messages share the request. An "at least" value is
used in full: the probe stopped there without reaching a limit, so it is already
below the real one. An inconclusive value keeps the margin (something failed above
it, possibly the limit).
"""
from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass
from functools import lru_cache
from typing import Optional

from ambient.config import match_model, settings

log = logging.getLogger(__name__)

MARGIN = 0.85
# Videos per request, measured directly on these deployments (2026-11-08: a second
# video gets "The model rejected the request parameters"). The vlm.fyi probe doesn't
# measure this yet; a `videos_per_request` limit in the data takes precedence.
_VIDEOS_PER_REQUEST = {"jarvislabs/qwen3.8-27b": 1, "jarvislabs/gemma-4-31b": 1}
_USABLE = ("measured", "stated", "at_least")


@dataclass(frozen=True)
class Capabilities:
    entry_id: str
    measured_at: str
    images_per_request: Optional[int] = None
    frames_768x432: Optional[int] = None
    inline_bytes: Optional[int] = None
    url_image_bytes: Optional[int] = None
    video_file_bytes: Optional[int] = None
    video: Optional[str] = None              # "yes" | "no" | "ignored" | None (unknown)
    frames_per_min: Optional[float] = None   # distinct frames per minute of a 5-min clip
    frames_per_video: Optional[int] = None   # ... and in total for that clip
    capped: bool = False                     # frames per video are capped, not a fixed rate
    tokens_per_frame_768x432: Optional[int] = None
    rejected_fields: tuple[str, ...] = ()
    strict_tool_order: Optional[bool] = None
    lower_bounds: frozenset[str] = frozenset()   # fields whose value is "at least"
    videos_per_request: Optional[int] = None

    def usable(self, name: str, margin: float = MARGIN) -> Optional[int]:
        """How much of limit `name` to plan against (see the module docstring)."""
        v = getattr(self, name)
        if not v:
            return None
        return int(v) if name in self.lower_bounds else max(1, math.floor(v * margin))

    def image_cap(self, margin: float = MARGIN) -> Optional[int]:
        """Images to send per request: the lower of the count limit and how many
        real-size frames fit (pixels / context)."""
        vals = [v for v in (self.usable("images_per_request", margin),
                            self.usable("frames_768x432", margin)) if v]
        return min(vals) if vals else None

    def video_vetoed(self, frames_needed: int, video_bytes: Optional[int] = None,
                     frames_alternative: Optional[int] = None, margin: float = MARGIN) -> Optional[str]:
        """Why a video_url would fail or be wasted here, or None if nothing measured
        says so. Only ever vetoes: whether to *use* video stays the caller's call.

        `frames_alternative` is how many images the frames path could send instead.
        A frame cap on video only matters if frames would deliver more: some
        deployments cap video at ~32 frames but take only 4 images per request."""
        if self.video == "no":
            return "provider rejects video input"
        if self.video == "ignored":
            return "provider accepts video but the model never sees it"
        if (self.capped and self.frames_per_video and frames_needed > self.frames_per_video
                and (frames_alternative is None or frames_alternative > self.frames_per_video)):
            return f"provider caps video at ~{self.frames_per_video} frames; plan needs {frames_needed}"
        if video_bytes and self.video_file_bytes and video_bytes > self.usable("video_file_bytes", margin):
            return f"video is {video_bytes / 1e6:.0f} MB; provider accepts ~{self.video_file_bytes / 1e6:.0f} MB"
        return None


def _value(limit: Optional[dict]) -> Optional[float]:
    if not limit:
        return None
    if limit.get("status") in _USABLE:
        return limit.get("value") or None
    if limit.get("status") == "inconclusive":
        return limit.get("at_least")
    return None


@lru_cache(maxsize=1)
def _entries() -> list[dict]:
    try:
        with open(settings.capabilities_file) as f:
            return json.load(f).get("entries", [])
    except FileNotFoundError:
        log.warning("capabilities: %s not found; using built-in endpoint profiles only", settings.capabilities_file)
        return []


def _service(base_url: Optional[str]) -> str:
    from ambient.media_plan import classify_service
    return classify_service(base_url)


def _from_entry(e: dict) -> Capabilities:
    L = e.get("limits") or {}
    as_int = lambda k: (lambda v: int(v) if v else None)(_value(L.get(k)))  # noqa: E731
    fpm = L.get("video_frames_per_min") or {}
    measured_frames = fpm.get("status") == "measured"
    fields = {"images_per_request": "images_per_request", "frames_768x432": "frames_768x432",
              "inline_bytes": "inline_request_bytes", "url_image_bytes": "url_image_bytes",
              "video_file_bytes": "video_file_bytes"}
    return Capabilities(
        lower_bounds=frozenset(f for f, k in fields.items() if (L.get(k) or {}).get("status") == "at_least"),
        entry_id=e["id"],
        videos_per_request=as_int("videos_per_request") or _VIDEOS_PER_REQUEST.get(e["id"]),
        measured_at=e.get("measured_at", ""),
        images_per_request=as_int("images_per_request"),
        frames_768x432=as_int("frames_768x432"),
        inline_bytes=as_int("inline_request_bytes"),
        url_image_bytes=as_int("url_image_bytes"),
        video_file_bytes=as_int("video_file_bytes"),
        video=(e.get("accepts") or {}).get("video"),
        frames_per_min=fpm.get("value") if measured_frames else None,
        frames_per_video=fpm.get("frames_per_video") if measured_frames else None,
        capped=bool(fpm.get("capped")) if measured_frames else False,
        tokens_per_frame_768x432=as_int("tokens_per_frame_768x432"),
        rejected_fields=tuple(k for k, ok in ((e.get("accepts") or {}).get("request_fields") or {}).items() if ok is False),
        strict_tool_order=(e.get("behaviour") or {}).get("strict_tool_message_order"),
    )


@lru_cache(maxsize=256)
def resolve(base_url: Optional[str], model: str) -> Optional[Capabilities]:
    """Capabilities for `model` on the endpoint at `base_url`, or None if unmeasured."""
    service = _service(base_url)
    upstream: Optional[str] = None
    if service == "openrouter":
        from ambient.config import get_openrouter_route
        upstream = get_openrouter_route(model).get("provider")   # None = auto-routed
    for e in _entries():
        p = e.get("provider") or {}
        if service == "openrouter":
            if p.get("via") != "openrouter" or (p.get("tag") or None) != upstream:
                continue
        elif p.get("via") or p.get("id") != service:
            continue
        m = e.get("model") or {}
        if match_model(model, m.get("api_id")) or match_model(model, m.get("id")):
            return _from_entry(e)
    return None

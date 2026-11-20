"""Fast-mode strategy selection + structured-output helpers.

Fast mode consumes a whole video in a single vision call (no agent loop, no
tools) and returns the response, optionally as a strict JSON object. The
*approach* — dense frames vs split clips — depends on the LLM service, the
model, and the video duration, because of hard limits we measured empirically
against OpenRouter:

  * `image_url` frames are deterministic (~338 tok/frame @768px, provider-
    invariant) but a single request caps at ~200 image blocks (auto-route;
    pinning caps lower). Great for short videos + frame-exact citations.
  * `video_url` clips get re-sampled to a fixed ~5.5k-token budget per clip
    (provider-dependent), so density is set by clip *length*; cost scales with
    the clip count. Few blocks, so they beat the image-count cap and cover long
    videos — at the cost of clip-level (not frame-exact) timestamps.

`pick_fast_strategy` encodes that policy so callers don't hardcode it.
"""
from __future__ import annotations

import asyncio
import math
from dataclasses import dataclass
from typing import Any, Literal, Optional


# --- tunables (measured defaults) -------------------------------------------
FRAMES_MAX = 200            # reliable single-request image_url count on OpenRouter (auto-route)
FRAMES_MAX_DIM = 768        # longest-edge px; ~338 tokens/frame at this size
CLIP_SECONDS = 16           # sweet spot: 2fps * 16s ≈ the provider's ~32-frame per-clip budget
CLIP_FPS = 2
CLIP_TOKENS_EST = 5500      # measured tokens per (long) clip on OpenRouter/CoreWeave
TOKEN_BUDGET = 120_000      # target input-token ceiling for a single fast call
SHORT_VIDEO_S = 300         # <= 5 min -> frames; longer -> clips


@dataclass
class FastStrategy:
    """How fast mode should feed a given video to the vision model."""
    method: Literal["frames", "clips", "video"]
    max_dim: int = FRAMES_MAX_DIM
    # frames
    max_frames: Optional[int] = None
    # clips
    clip_seconds: Optional[float] = None
    fps: int = CLIP_FPS
    # routing / budget
    provider_pin: Optional[str] = None
    token_budget: int = TOKEN_BUDGET

    def effective_fps(self, duration_s: float) -> float:
        if self.method == "frames" and self.max_frames and duration_s > 0:
            return self.max_frames / duration_s
        return float(self.fps)


def pick_fast_strategy(
    service: str,
    model: str,
    duration_s: float,
    *,
    need_frame_exact_ts: bool = False,
) -> FastStrategy:
    """Choose how to feed the video for (service, model, duration).

    service: "openrouter" | "self_hosted" (extend as backends are added).
    """
    svc = (service or "").lower()

    if svc == "self_hosted":
        # The served vLLM samples the video internally (its `mm_processor_kwargs`
        # video budget, e.g. size.longest_edge). Send the video URL directly — no
        # frame extraction/upload, ~half the tokens (temporal merge), and ~3-4x
        # faster than the e2b frame pipeline (benchmarked).
        return FastStrategy("video")

    # OpenRouter: its video_url path is provider-downsampled to ~16-32 frames with
    # no control, so we send frames instead.
    if duration_s <= SHORT_VIDEO_S or need_frame_exact_ts:
        # Short (or precision-critical): 200 frames, auto-route (no pin — pinning
        # caps image count well below 200; see the probes).
        return FastStrategy("frames", max_frames=FRAMES_MAX, provider_pin=None)

    # Long video: split into clips, lengthening them so the estimated token cost
    # stays within budget (density degrades gracefully instead of exploding).
    max_clips = max(1, TOKEN_BUDGET // CLIP_TOKENS_EST)
    clip_seconds = max(CLIP_SECONDS, duration_s / max_clips)
    return FastStrategy("clips", clip_seconds=clip_seconds, fps=CLIP_FPS, provider_pin=None)


# --- structured output: Pydantic / JSON schema -> strict response_format -----

def build_response_format(name: str, schema_or_model: Any) -> dict:
    """Wrap a schema as OpenRouter/OpenAI strict structured-output `response_format`.

    Accepts a raw JSON-schema dict or a Pydantic model class. The schema is
    normalized to satisfy `strict: true` (additionalProperties:false, all keys
    required, `$ref`/`$defs` inlined).
    """
    if hasattr(schema_or_model, "model_json_schema"):
        schema = schema_or_model.model_json_schema()
    else:
        schema = dict(schema_or_model)
    schema = _to_strict_schema(schema)
    return {
        "type": "json_schema",
        "json_schema": {"name": name, "strict": True, "schema": schema},
    }


def _to_strict_schema(schema: dict) -> dict:
    """Return a strict-mode-compatible copy: inline $refs, force
    additionalProperties:false + required=all on every object."""
    defs = schema.get("$defs") or schema.get("definitions") or {}
    resolved = _inline_refs(schema, defs)
    _strictify(resolved)
    if isinstance(resolved, dict):
        resolved.pop("$defs", None)
        resolved.pop("definitions", None)
    return resolved


def _inline_refs(node: Any, defs: dict, _seen: Optional[frozenset] = None) -> Any:
    _seen = _seen or frozenset()
    if isinstance(node, dict):
        ref = node.get("$ref")
        if isinstance(ref, str) and ref.startswith(("#/$defs/", "#/definitions/")):
            key = ref.split("/")[-1]
            if key in defs and key not in _seen:
                return _inline_refs(defs[key], defs, _seen | {key})
            return {}  # unresolvable / recursive -> permissive empty schema
        return {k: _inline_refs(v, defs, _seen) for k, v in node.items()
                if k not in ("$defs", "definitions")}
    if isinstance(node, list):
        return [_inline_refs(v, defs, _seen) for v in node]
    return node


def _strictify(node: Any) -> None:
    if isinstance(node, dict):
        if node.get("type") == "object" and isinstance(node.get("properties"), dict):
            node["additionalProperties"] = False
            node["required"] = list(node["properties"].keys())
        for v in node.values():
            _strictify(v)
    elif isinstance(node, list):
        for v in node:
            _strictify(v)


# --- frame sampling for fast mode -------------------------------------------

def service_from_base_url(base_url: Optional[str]) -> str:
    """Classify the LLM endpoint so `pick_fast_strategy` can apply the right caps."""
    b = (base_url or "").lower()
    if "openrouter" in b:
        return "openrouter"
    return "self_hosted"


async def sample_fast_frames(video_id: str, media_box, strategy: "FastStrategy",
                             duration: Optional[float] = None,
                             source_fps: Optional[float] = None):
    """Uniformly sample `strategy.max_frames` frames across the whole video.

    Runs the (blocking) media work on a worker thread; `media_box` is published on
    the `current_media_box` ContextVar so the e2b backend picks it up (None for the
    inprocess backend). `duration` (seconds), when known from the file record, is
    passed to the tools so they don't probe it. `source_fps`, when known, caps the
    sample at the frames the source actually has: asking a 10.4 s, 10 fps clip
    (104 frames) for 192 made the extractor sample at 19 fps and repeat frames.
    Returns `(frames, duration_seconds)`.
    """
    from ambient.tools.video_backend import current_media_box, make_video_tools

    n = strategy.max_frames or FRAMES_MAX

    def _work():
        tools = make_video_tools(video_id, max_frame_dimention=strategy.max_dim,
                                 duration=duration)
        # `dur` (local) keeps the outer `duration` param unshadowed. For the
        # range-proxy the passed duration is already on tools._duration_sec.
        dur = getattr(tools, "_duration_sec", None) or duration
        if not dur and hasattr(tools, "probe_duration"):
            # Range-proxy: cheap ffprobe (no frame extraction) for the duration.
            dur = tools.probe_duration()
        if not dur:
            # inprocess/e2b: a 1-frame fetch populates their _duration_sec.
            tools.fetch_frames(fps=1, start_time_sec=0, end_time_sec=None, max_frames=1)
            dur = getattr(tools, "_duration_sec", None)
        if dur and dur > 0:
            # Never ask for more frames than the source has: above the source rate
            # the extractor repeats frames (pure token cost, no new content).
            cap = source_frame_count(dur, source_fps)
            want = min(n, cap) if cap else n
            # The extractor needs an integer fps >= 1 and samples `max_frames`
            # uniformly across the window. Pick the smallest integer fps whose grid
            # yields >= want candidates, then let max_frames downsample to exactly want.
            fps = max(1, math.ceil(want / dur))
            frames = tools.fetch_frames(fps=fps, start_time_sec=0,
                                        end_time_sec=dur, max_frames=want)
        else:
            frames = tools.fetch_frames(fps=1, start_time_sec=0, end_time_sec=None, max_frames=n)
        return frames, (dur or 0.0)

    token = current_media_box.set(media_box)
    try:
        return await asyncio.to_thread(_work)
    finally:
        current_media_box.reset(token)


def source_frame_count(duration: Optional[float], source_fps: Optional[float]) -> Optional[int]:
    """Frames a source of `duration` s at `source_fps` holds (None when unknown)."""
    if not duration or duration <= 0 or not source_fps or source_fps <= 0:
        return None
    return max(1, int(duration * source_fps))


def build_fast_frames_message(frames: list, note: Optional[str] = None) -> dict:
    """A stable user message carrying the sampled frames as timestamp-labeled
    image blocks. Kept identical across follow-up turns so it stays a cacheable
    prefix. `note` (see `frames_context_note`) tells the agent how much of the
    video these frames show."""
    from ambient.llm import construct_payload

    payload = construct_payload(frames=frames)
    header = "Frames sampled uniformly across the video (in time order):"
    if note:
        header = f"{header}\n{note}"
    return {"role": "user", "content": [
        {"type": "text", "text": header},
        *payload,
    ]}


# Frames per second that focus_clip / grab_frames return on the frames path (the
# e2b and range-proxy backends use 1-2 fps).
TOOL_FRAMES_PER_SECOND = 2


def probe_source_video(path: Optional[str]) -> Optional[dict]:
    """Width, height and frame rate of a local video file via ffprobe, or None."""
    import json
    import os
    import subprocess

    if not path or not os.path.exists(path):
        return None
    try:
        r = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "v:0",
             "-show_entries", "stream=width,height,avg_frame_rate", "-of", "json", path],
            capture_output=True, timeout=30)
        s = json.loads(r.stdout.decode() or "{}").get("streams", [{}])[0]
        num, _, den = str(s.get("avg_frame_rate") or "0/1").partition("/")
        fps = float(num) / float(den or 1) if float(den or 1) else 0.0
        if not s.get("width") or not s.get("height"):
            return None
        return {"width": int(s["width"]), "height": int(s["height"]), "fps": round(fps, 2) or None}
    except Exception:  # noqa: BLE001 - the note just omits the source details
        return None


def frames_context_note(n_frames: int, duration: float, source: Optional[dict] = None,
                        frame_dim: Optional[int] = None) -> Optional[str]:
    """Tell the agent how densely the attached frames cover the video, and so how
    far it can answer from them. Without this the agent treats even near-complete
    coverage as a sparse overview and keeps calling tools that cannot show more
    (and, with the bash tool, crops and upscales frames one image per step)."""
    if n_frames <= 0 or not duration or duration <= 0:
        return None
    gap = duration / n_frames
    rate = n_frames / duration
    src = ""
    at_source_res = False
    if source:
        w, h, fps = source.get("width"), source.get("height"), source.get("fps")
        src = f" The source video is {w}×{h}"
        if fps:
            share = min(100.0, 100.0 * n_frames / (fps * duration))
            src += f" at {fps:g} fps, so these are {share:.0f}% of all its frames"
        src += "."
        at_source_res = bool(frame_dim and w and h and max(w, h) <= frame_dim)
    head = (f"{n_frames} frames over {duration:.1f} s: one every {gap:.2f} s "
            f"({rate:.1f} per second).{src}")
    if rate >= TOOL_FRAMES_PER_SECOND:
        detail = (" No tool shows more than the source's pixels: enlarging, cropping or "
                  "upscaling these frames adds no detail." if at_source_res else "")
        return (f"{head} That is at least as dense as focus_clip and grab_frames return "
                f"(up to {TOOL_FRAMES_PER_SECOND} frames per second).{detail} Answer from "
                "these frames. Use a tool only for something they do not show, such as the "
                "exact moment between two of them.")
    if gap <= 5:
        return (f"{head} Use them to follow the sequence and find where things happen; "
                "check fine timing or small details between them with focus_clip or "
                "grab_frames.")
    return (f"{head} They are sparse: use them as a map to find where things probably "
            "happen, then inspect those windows with focus_clip or grab_frames before "
            "relying on details.")


# --- direct video (self-hosted vLLM samples frames itself) -------------------

async def source_video_url(video_id: str, store, expires_in: int = 7200) -> Optional[str]:
    """Presigned URL of the source video on S3/R2, for sending as a `video_url`.

    Uses the file record's stored key (`r2_key`, set on upload) and falls back to
    `<base>/<video_id>.mp4` (youtube imports are remuxed to mp4). Returns None if
    S3 isn't configured, so the caller can fall back to the frame path.
    """
    from ambient.config import settings
    from ambient.utils.s3 import get_s3_client

    s3 = get_s3_client()
    if not getattr(s3, "bucket", None):
        return None
    key = None
    try:
        rec = await store.get_file(video_id)
        key = (rec or {}).get("r2_key")
    except Exception:  # noqa: BLE001
        pass
    if not key:
        key = f"{settings.s3_video_base_key}/{video_id}.mp4"
    # Only advertise a URL a remote consumer can actually fetch. A YouTube import
    # prepared on the host (sandbox_backend != "e2b") never uploads to R2, so the
    # conventional key 404s — verify existence and return None so the caller falls
    # back to inline frames instead of attaching a dead URL (which 422s the run).
    if not await asyncio.to_thread(s3.object_exists, key):
        return None
    try:
        return s3.get_presigned_url(key, expires_in=expires_in)
    except Exception:  # noqa: BLE001
        return None


def build_fast_video_message(url: str, video_id: str = None) -> dict:
    """A stable user message carrying the whole video as a single `video_url`.
    The vision server samples frames internally per its processor config."""
    return {"role": "user", "content": [
        {"type": "text", "text": f"The full video of video id: {video_id} (frames are sampled across its entire duration):"},
        {"type": "video_url", "video_url": {"url": url}},
    ]}

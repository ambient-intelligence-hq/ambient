"""Media-planning: decide *how* to feed a video span to a vision model.

Two decoupled axes (kept separate on purpose — conflating them is what makes
video sampling confusing):

  1. **Budget** — how densely to sample, per pipeline *step* (context / focus_clip
     / search_clip / grab_frames …). A `SamplingSpec {max_frames, fps}` per step,
     reduced to a single `effective_fps` via "whichever is fewer".

  2. **Delivery** — `video` (one `video_url`, cheap via temporal merge) vs `frames`
     (N `image_url` blocks, deterministic). Decided ONLY by endpoint/model
     capability, never by the step:

       * self-hosted vLLM  -> we control a video's frames by transcoding it to
         exactly `fps x dur` frames (the range-proxy), which vLLM keeps up to its
         `longest_edge` ceiling. So `video` is honored and cheapest.
       * OpenRouter / Vercel -> the provider re-samples any `video_url` to its own
         fixed ~24-frame budget (measured), ignoring our fps. So `frames` is the
         only way to control the count there.
       * image-only models -> `frames`.

`plan_media(step, base_url, model, span_seconds)` combines both into a `MediaPlan`
the range-proxy (or any realizer) executes. Everything here is a pure function of
its inputs + the registries below, so it's trivially extendable:

  * add a step        -> add to `SAMPLING`
  * add an endpoint   -> add to `ENDPOINT_PROFILES` + `classify_service`
  * override per-call -> pass `spec=SamplingSpec(...)` to `plan_media`
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Literal, Optional

# --- 1. Per-step sampling budgets -------------------------------------------

@dataclass(frozen=True)
class SamplingSpec:
    """How densely to sample a span. `effective_fps` reconciles the two knobs:
    use `fps`, but never more than `max_frames` over the span."""
    max_frames: int
    fps: float
    max_dim: int = 768

    def effective_fps(self, span_seconds: float) -> float:
        if span_seconds and span_seconds > 0:
            return min(self.fps, self.max_frames / span_seconds)
        return self.fps

    def frames_for(self, span_seconds: float) -> int:
        return max(1, min(self.max_frames, math.ceil(self.effective_fps(span_seconds) * max(span_seconds, 0))))


# Default budget per pipeline step. Tunable; a caller may also pass a one-off
# `spec=` to plan_media to override for a single call.
SAMPLING: dict[str, SamplingSpec] = {
    # Whole-video context: fast mode's attachment / the ingestion description.
    "context":     SamplingSpec(max_frames=500, fps=1.0),
    # Agent tools that analyze a window with a vision sub-call.
    "focus_clip":  SamplingSpec(max_frames=300, fps=2.0),
    "search_clip": SamplingSpec(max_frames=300, fps=2.0),
    # Raw frames handed to the agent model to inspect directly (always frames).
    "grab_frames": SamplingSpec(max_frames=30,  fps=1.0),
}

# Steps whose output is raw frames the *agent* model looks at itself (not a vision
# sub-call), so they must always be `frames` regardless of endpoint capability.
FRAME_ONLY_STEPS = frozenset({"grab_frames"})


# --- 2. Endpoint capabilities ------------------------------------------------

@dataclass(frozen=True)
class EndpointProfile:
    """What an LLM endpoint lets us do with video/image inputs."""
    service: str
    # Can we control how many frames the model sees from a *video* input?
    # Self-hosted: yes (we transcode exact frames). External proxies: no (they
    # re-sample video_url to a fixed budget), so frames are required for control.
    video_frame_control: bool
    # Max number of image_url blocks a single request reliably accepts.
    image_block_cap: int


ENDPOINT_PROFILES: dict[str, EndpointProfile] = {
    "self_hosted": EndpointProfile("self_hosted", video_frame_control=True,  image_block_cap=1000),
    "openrouter":  EndpointProfile("openrouter",  video_frame_control=False, image_block_cap=200),
    "vercel":      EndpointProfile("vercel",      video_frame_control=False, image_block_cap=200),
}

# Substring -> service. First match wins; anything unmatched is treated as our own
# self-hosted vLLM (the only endpoint where we control video frames).
_SERVICE_PATTERNS: list[tuple[str, str]] = [
    ("openrouter", "openrouter"),
    ("vercel", "vercel"),
]


def classify_service(base_url: Optional[str]) -> str:
    b = (base_url or "").lower()
    for pat, service in _SERVICE_PATTERNS:
        if pat in b:
            return service
    return "self_hosted"


def profile_for(base_url: Optional[str]) -> EndpointProfile:
    return ENDPOINT_PROFILES.get(classify_service(base_url), ENDPOINT_PROFILES["self_hosted"])


def model_modalities(model: str) -> list[str]:
    """Input modalities for `model`, from models.json (via config). Unknown models
    default to full VL — the *endpoint* gate (video_frame_control) does the real
    work, so an over-permissive modality guess is safe."""
    try:
        from ambient.config import get_model_modalities
        mods = get_model_modalities(model)
        if mods:
            # get_model_modalities returns enum members (…IMAGE/…VIDEO); normalize
            # to their string values ("text"/"image"/"video").
            return [str(getattr(m, "value", m)).lower() for m in mods]
    except Exception:  # noqa: BLE001
        pass
    return ["text", "image", "video"]


# --- 3. The plan -------------------------------------------------------------

@dataclass
class MediaPlan:
    """How to realize the media for one step. `controlled=False` flags the
    degraded case (video sent to a provider that ignores our sampling)."""
    method: Literal["video", "frames"]
    fps: float
    max_frames: int
    max_dim: int = 768
    controlled: bool = True
    note: str = ""
    meta: dict = field(default_factory=dict)

    def frames_estimate(self, span_seconds: float) -> int:
        return max(1, min(self.max_frames, math.ceil(self.fps * max(span_seconds, 0)) or 1))


def plan_media(
    step: str,
    base_url: Optional[str],
    model: str,
    span_seconds: float,
    *,
    spec: Optional[SamplingSpec] = None,
    force_frames: bool = False,
) -> MediaPlan:
    """Decide the input type + sampling for one media step.

    step:        key into SAMPLING ("context"/"focus_clip"/"search_clip"/"grab_frames").
    base_url:    the LLM endpoint this media is going to (selects the profile).
    model:       the model id (selects modalities).
    span_seconds: duration of the span (whole video for "context", window for clips).
    spec:        optional per-call override of the step's default budget.
    force_frames: caller forces the frames path (e.g. no S3 to host a video_url).
    """
    spec = spec or SAMPLING.get(step) or SAMPLING["focus_clip"]
    prof = profile_for(base_url)
    mods = model_modalities(model)
    eff_fps = spec.effective_fps(span_seconds)

    # Raw-frame steps (grab_frames) and forced-frames always deliver image frames.
    if force_frames or step in FRAME_ONLY_STEPS:
        cap = min(spec.max_frames, prof.image_block_cap)
        return MediaPlan("frames", fps=min(eff_fps, cap / max(span_seconds, 1)),
                         max_frames=cap, max_dim=spec.max_dim,
                         meta={"service": prof.service, "reason": "frame-only step"})

    # Video: only when the endpoint honors frame control on video AND the model
    # takes video. Cheapest (temporal merge) and exact via transcode.
    if prof.video_frame_control and "video" in mods:
        return MediaPlan("video", fps=eff_fps, max_frames=spec.max_frames, max_dim=spec.max_dim,
                         meta={"service": prof.service, "reason": "video frame-control"})

    # Frames: deterministic on any image-capable model (external proxies,
    # image-only models). Drop fps if the frame count would exceed the block cap.
    if "image" in mods:
        cap = min(spec.max_frames, prof.image_block_cap)
        fps = min(eff_fps, cap / max(span_seconds, 1))
        return MediaPlan("frames", fps=fps, max_frames=cap, max_dim=spec.max_dim,
                         meta={"service": prof.service,
                               "reason": "no video frame-control -> frames"})

    # Video-only model (no image) on a no-control endpoint: coarse, provider-sampled.
    if "video" in mods:
        return MediaPlan("video", fps=eff_fps, max_frames=spec.max_frames, max_dim=spec.max_dim,
                         controlled=False,
                         note="provider re-samples video; fps/max_frames not honored",
                         meta={"service": prof.service, "reason": "video-only, uncontrolled"})

    # Neither image nor video: the model can't take media at all — caller should
    # error / skip. Flagged via meta["error"].
    return MediaPlan("frames", fps=0.0, max_frames=0, max_dim=spec.max_dim, controlled=False,
                     note=f"model '{model}' has no image/video modality",
                     meta={"service": prof.service, "error": "no_media_modality"})

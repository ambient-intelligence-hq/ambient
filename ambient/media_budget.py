"""Per-request media budget: keep a growing agent conversation inside the provider's
request limits, at the least cost to prompt caching and to what the agent knows.

Every completion resends the whole conversation, so tool media (focus_clip frames
and clips) piles up turn after turn until the provider rejects the request. The
limits come in three units, all measured per provider (ambient.capabilities):

  * images        - image blocks per request (Fireworks: 120)
  * videos        - video blocks per request (some deployments: 1). A hard count, so
                    it is exempt from the 85%/60% marks below.
  * frames        - frame equivalents: images + the frames of each video clip. This
                    stands in for decoded pixels / visual context (Together: 1.43 GB
                    decoded, Modal vLLM: VLLM_MAX_MEDIA_REQUEST_DECODE_BYTES ~715 MB)
  * inline bytes  - base64 data URLs in the body (Jarvislabs: ~12 MB)
  * URL bytes     - image bytes the provider downloads from URLs (e2b, S3) in one
                    request (Alibaba: 30 MB). Sizes are measured with a 1-byte ranged
                    GET per URL (`url_sizes`, cached), only where this limit is known.

`fit_media_budget(messages, budget)` edits the conversation in place:

  1. Nothing happens below `HIGH` (85%) of any limit, so most turns leave the
     prompt byte-identical and the cache keeps hitting.
  2. Above it, older media is reduced until the request is under `LOW` (60%), so
     one reduction buys several turns before the next.
  3. Oldest tool media goes first. The opening video context (a stable prefix,
     sized once to `CONTEXT_SHARE` of the budget) and this turn's fresh tool
     results are protected; fresh results are reduced only if they alone would
     break the hard limit, and the newest one is never removed outright: the turn
     would otherwise show the model nothing, so the provider gets to decide.
  4. Media degrades before it disappears: full frames -> every 4th frame -> a
     deterministic note naming the time range and how to view it again. The
     agent's own notes and the tool results' text are never touched.

Reductions are written into the persisted messages, so later turns send the same
reduced content and caching resumes from the edit point.
"""
from __future__ import annotations

import asyncio
import logging
import math
import os
import re
from dataclasses import dataclass, field
from typing import Optional

from ambient.config import settings

log = logging.getLogger(__name__)

HIGH = 0.85          # reduce only when a request would exceed this share of a limit
LOW = 0.60           # ... and then reduce down to this share
CONTEXT_SHARE = 0.5  # the opening video context takes at most this share (agent mode)
THUMB_EVERY = 4      # thumbnail level keeps every Nth frame
_CLIP_FRAMES_DEFAULT = 60  # frame estimate for a clip whose time range isn't labeled
# Size guesses for hosted media whose size couldn't be measured (deliberately high:
# e2b frames averaged ~160 KB in the matrix).
_URL_IMAGE_GUESS = 250_000

_RANGE = re.compile(r"between\s+(\d+(?:\.\d+)?)\s+(?:seconds\s+)?and\s+(\d+(?:\.\d+)?)\s+seconds")
_TOOL = re.compile(r"^Media from (\w+)")
REDUCED_PREFIX = "[Reduced from"
REMOVED_PREFIX = "[Media for"
FRAMES_KEY = "_frames"      # message field: how many frames its video really adds


@dataclass(frozen=True)
class MediaBudget:
    images: int
    frames: int
    inline_bytes: int
    source: str = "default"   # "measured" when any unit came from capabilities.json
    url_bytes: Optional[int] = None   # None = no known download limit
    videos: Optional[int] = None      # None = no known limit (see MAX_CLIPS in the runner)


@dataclass
class Cost:
    images: int = 0
    frames: int = 0
    inline_bytes: int = 0
    url_bytes: int = 0
    videos: int = 0

    def __iadd__(self, o: "Cost") -> "Cost":
        self.images += o.images
        self.frames += o.frames
        self.inline_bytes += o.inline_bytes
        self.url_bytes += o.url_bytes
        self.videos += o.videos
        return self

    def __isub__(self, o: "Cost") -> "Cost":
        self.images -= o.images
        self.frames -= o.frames
        self.inline_bytes -= o.inline_bytes
        self.url_bytes -= o.url_bytes
        self.videos -= o.videos
        return self

    def over(self, b: MediaBudget, share: float) -> bool:
        return (self.images > b.images * share or self.frames > b.frames * share
                or self.inline_bytes > b.inline_bytes * share
                or (b.url_bytes is not None and self.url_bytes > b.url_bytes * share)
                or (b.videos is not None and self.videos > b.videos))


def budget_for(base_url: Optional[str], model: str) -> MediaBudget:
    """The request budget for `model` on the endpoint at `base_url`: limits from
    vlm.fyi where measured (`Capabilities.usable`), else the endpoint profile's image
    cap and conservative defaults."""
    from ambient.capabilities import resolve
    from ambient.media_plan import _legacy_image_block_cap

    caps = resolve(base_url, model)
    measured = bool(caps and (caps.image_cap() or caps.frames_768x432 or caps.inline_bytes))
    images = (caps.image_cap() if caps else None) or _legacy_image_block_cap(base_url, model)
    frames = (caps.usable("frames_768x432") if caps else None) or settings.media_budget_frames
    inline = (caps.usable("inline_bytes") if caps else None) or settings.media_budget_inline_bytes
    url = caps.usable("url_image_bytes") if caps else None
    if caps and caps.video == "yes":
        # The real-size frame limit is about images (it can be as low as 4); video
        # frames on a deployment that takes video are bounded by the default instead.
        frames = max(frames, settings.media_budget_frames)
    images = max(1, min(images, frames, settings.max_frames_per_prompt))
    return MediaBudget(images=images, frames=max(frames, images), inline_bytes=inline,
                       source="measured" if measured else "default", url_bytes=url,
                       videos=caps.videos_per_request if caps else None)


# --- costing -----------------------------------------------------------------

def _url(block: dict, kind: str) -> str:
    return str((block.get(kind) or {}).get("url") or "")


def _inline(url: str) -> int:
    return len(url) if url.startswith("data:") else 0


def _clip_frames(label: Optional[str]) -> int:
    from ambient.media_plan import SAMPLING
    m = _RANGE.search(label or "")
    if not m:
        return _CLIP_FRAMES_DEFAULT
    return SAMPLING["focus_clip"].frames_for(max(0.0, float(m.group(2)) - float(m.group(1))))


def _hosted(url: str) -> bool:
    return url.startswith(("http://", "https://"))


def block_cost(block: dict, label: Optional[str] = None, sizes: Optional[dict] = None) -> Cost:
    sizes = sizes or {}
    t = block.get("type")
    if t == "image_url":
        u = _url(block, "image_url")
        return Cost(1, 1, _inline(u), sizes.get(u, _URL_IMAGE_GUESS) if _hosted(u) else 0)
    if t == "video_url":
        u = _url(block, "video_url")
        # URL video files are capped per file (plan_media's video veto), not by the
        # downloaded-image limit, so they don't count toward url_bytes.
        return Cost(0, _clip_frames(label), _inline(u), videos=1)
    return Cost()


def message_cost(msg: dict, sizes: Optional[dict] = None) -> Cost:
    total, label = Cost(), None
    has_video = False
    for b in _blocks(msg):
        if b.get("type") == "text":
            label = b.get("text")
        else:
            has_video = has_video or b.get("type") == "video_url"
            total += block_cost(b, label, sizes)
    # A video's frame count can't be read off its URL, so whoever attaches it records
    # it here (FRAMES_KEY; the runner strips it before the provider sees the message).
    if has_video and msg.get(FRAMES_KEY) is not None:
        total.frames = int(msg[FRAMES_KEY])
    return total


def hosted_urls(messages: list[dict], skip: int = 0) -> list[str]:
    """Hosted (http) image URLs in `messages`, excluding the first `skip` messages."""
    return [u for m in messages[skip:] for b in _blocks(m)
            if b.get("type") == "image_url" and _hosted(u := _url(b, "image_url"))]


_SIZES: dict[str, int] = {}   # url -> bytes; presigned URLs are unique per object


async def url_sizes(urls: list[str], concurrency: int = 16, timeout: float = 5.0) -> dict[str, int]:
    """Byte size of each hosted media URL, via `Range: bytes=0-0` (works with
    GET-presigned S3/R2 URLs, unlike HEAD). Cached; unmeasurable URLs are left out
    and costed with a conservative guess."""
    todo = [u for u in dict.fromkeys(urls) if u not in _SIZES]
    if todo:
        import aiohttp
        sem = asyncio.Semaphore(concurrency)

        async def one(session, u):
            async with sem:
                try:
                    async with session.get(u, headers={"Range": "bytes=0-0"}) as r:
                        cr = r.headers.get("Content-Range", "")
                        if "/" in cr and cr.rsplit("/", 1)[1].isdigit():
                            _SIZES[u] = int(cr.rsplit("/", 1)[1])
                        elif r.status == 200 and r.content_length:
                            _SIZES[u] = int(r.content_length)
                except Exception as exc:  # noqa: BLE001 - fall back to the guess
                    log.debug("media_budget: size probe failed for a media URL: %s", exc)

        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=timeout)) as session:
            await asyncio.gather(*(one(session, u) for u in todo))
    return {u: _SIZES[u] for u in urls if u in _SIZES}


def _blocks(msg: dict) -> list[dict]:
    c = msg.get("content")
    return [b for b in c if isinstance(b, dict)] if isinstance(c, list) else []


def is_media_message(msg: dict) -> bool:
    return msg.get("role") == "user" and any(
        b.get("type") in ("image_url", "video_url") for b in _blocks(msg))


def context_index(messages: list[dict]) -> Optional[int]:
    """The opening video context: the first media message, attached right after the
    system prompt (see SessionRunner._has_video_context)."""
    for i, m in enumerate(messages[:3]):
        if is_media_message(m):
            return i
    return None


# --- labels and notes ----------------------------------------------------------

def media_label(tool: str, tool_input: dict) -> Optional[str]:
    """Leading text for a tool's media message, so the window it shows can be named
    (and re-opened) if the media is later reduced. None if the call has no range."""
    if tool_span(tool_input) is None:
        return None
    s, e = float(tool_input["start_time"]), float(tool_input["end_time"])
    return f"Media from {tool} between {s:.1f} and {e:.1f} seconds:"


def tool_span(tool_input: dict) -> Optional[float]:
    """Seconds a tool call covers, or None if it names no valid range."""
    try:
        return max(float(tool_input["end_time"]) - float(tool_input["start_time"]), 0.1)
    except (KeyError, TypeError, ValueError):
        return None


def _mmss(sec: float) -> str:
    sec = int(round(sec))
    return f"{sec // 60:02d}:{sec % 60:02d}"


def _describe(msg: dict) -> tuple[Optional[tuple[float, float]], str]:
    """The full time range a media message covers (a window can arrive as several
    consecutive tiles, each with its own label) and the tool that re-opens it."""
    starts, ends, tool = [], [], "focus_clip"
    for b in _blocks(msg):
        if b.get("type") != "text":
            continue
        txt = b.get("text") or ""
        for m in _RANGE.finditer(txt):
            starts.append(float(m.group(1)))
            ends.append(float(m.group(2)))
        if (m := _TOOL.match(txt)):
            tool = m.group(1)
    return ((min(starts), max(ends)) if starts else None), tool


def _reopen(rng: Optional[tuple[float, float]], tool: str) -> str:
    if not rng:
        return f"call {tool} on that range to see it again"
    return f"call {tool}(start_time={rng[0]:g}, end_time={rng[1]:g}) to see it again"


def _thumbnail(msg: dict) -> Optional[dict]:
    blocks = _blocks(msg)
    images = [i for i, b in enumerate(blocks) if b.get("type") == "image_url"]
    if len(images) <= THUMB_EVERY or any(b.get("type") == "video_url" for b in blocks):
        return None   # too few frames to thin, or a clip (which can't be thinned)
    keep = set(images[::THUMB_EVERY])
    rng, tool = _describe(msg)
    span = f" of {_mmss(rng[0])}–{_mmss(rng[1])}" if rng else ""
    note = (f"{REDUCED_PREFIX} {len(images)} to {len(keep)} frames{span} (every {THUMB_EVERY}th kept) "
            f"to fit this provider's request limits; {_reopen(rng, tool)} in full detail.]")
    kept = [b for i, b in enumerate(blocks) if b.get("type") != "image_url" or i in keep]
    return {**msg, "content": [{"type": "text", "text": note}, *kept]}


def _placeholder(msg: dict) -> dict:
    rng, tool = _describe(msg)
    kind = "clip" if any(b.get("type") == "video_url" for b in _blocks(msg)) else "frames"
    span = f"{_mmss(rng[0])}–{_mmss(rng[1])}" if rng else "an earlier range"
    note = (f"{REMOVED_PREFIX} {span} ({kind}) was removed to fit this provider's request "
            f"limits. Your earlier notes on it still stand; {_reopen(rng, tool)}.]")
    return {**msg, "content": [{"type": "text", "text": note}]}


# --- fitting ---------------------------------------------------------------------

@dataclass
class Reduction:
    before: Cost
    after: Cost
    budget: MediaBudget
    thumbnailed: int = 0
    removed: int = 0
    fresh_touched: int = 0
    ranges: list[str] = field(default_factory=list)

    def summary(self) -> str:
        parts = []
        if self.thumbnailed:
            parts.append(f"{self.thumbnailed} thinned to every {THUMB_EVERY}th frame")
        if self.removed:
            parts.append(f"{self.removed} replaced with notes")
        return (f"older media reduced to fit provider limits ({', '.join(parts)}): "
                f"{self.before.images}->{self.after.images} images, "
                f"{self.before.frames}->{self.after.frames} frame-equivalents, "
                f"{self.before.inline_bytes / 1e6:.1f}->{self.after.inline_bytes / 1e6:.1f} MB inline"
                + (f", {self.before.url_bytes / 1e6:.1f}->{self.after.url_bytes / 1e6:.1f} MB by URL"
                   if self.budget.url_bytes is not None else ""))

    def as_dict(self) -> dict:
        return {
            "thumbnailed": self.thumbnailed, "removed": self.removed,
            "fresh_touched": self.fresh_touched, "ranges": self.ranges,
            "before": vars(self.before), "after": vars(self.after),
            "budget": {"images": self.budget.images, "frames": self.budget.frames,
                       "inline_bytes": self.budget.inline_bytes, "url_bytes": self.budget.url_bytes,
                       "source": self.budget.source},
        }


def fit_media_budget(messages: list[dict], budget: MediaBudget,
                     sizes: Optional[dict] = None) -> Optional[Reduction]:
    """Reduce older media in `messages` (in place) so the next request fits `budget`.
    `sizes` maps hosted media URLs to bytes (see `url_sizes`). Returns what was done,
    or None if the request already fits (the usual case)."""
    total = Cost()
    for m in messages:
        total += message_cost(m, sizes)
    if not total.over(budget, HIGH):
        return None

    ctx = context_index(messages)
    last_assistant = max((i for i, m in enumerate(messages) if m.get("role") == "assistant"), default=-1)
    media = [i for i, m in enumerate(messages) if i != ctx and is_media_message(m)]
    older = [i for i in media if i < last_assistant]
    fresh = [i for i in media if i > last_assistant]
    red = Reduction(before=Cost(**vars(total)), after=total, budget=budget)

    def apply(i: int, new: Optional[dict], is_fresh: bool = False) -> None:
        if new is None:
            return
        total.__isub__(message_cost(messages[i], sizes))
        total.__iadd__(message_cost(new, sizes))
        if not is_media_message(new):
            red.removed += 1
            rng, _ = _describe(messages[i])
            red.ranges.append(f"{_mmss(rng[0])}–{_mmss(rng[1])}" if rng else "?")
        else:
            red.thumbnailed += 1
        red.fresh_touched += int(is_fresh)
        messages[i] = new

    for degrade in (_thumbnail, _placeholder):
        for i in older:
            if not total.over(budget, LOW):
                break
            if is_media_message(messages[i]) and not (degrade is _thumbnail and _is_thumb(messages[i])):
                apply(i, degrade(messages[i]))
    # Last resort: this turn's results alone break the hard limit.
    for degrade in (_thumbnail, _placeholder):
        for i in fresh:
            if not total.over(budget, 1.0):
                break
            if degrade is _placeholder and i == fresh[-1]:
                continue    # never leave the turn with nothing to look at
            if is_media_message(messages[i]) and not (degrade is _thumbnail and _is_thumb(messages[i])):
                apply(i, degrade(messages[i]), is_fresh=True)

    if total.over(budget, 1.0):
        log.warning("media_budget: request still over budget after reductions "
                    "(%s images / %s frames / %.1f MB inline / %.1f MB by URL vs %s)", total.images,
                    total.frames, total.inline_bytes / 1e6, total.url_bytes / 1e6, budget)
    return red if (red.thumbnailed or red.removed) else None


def _is_thumb(msg: dict) -> bool:
    b = _blocks(msg)
    return bool(b) and b[0].get("type") == "text" and str(b[0].get("text", "")).startswith(REDUCED_PREFIX)


# --- what a turn can add ------------------------------------------------------------

def free_video_slots(messages: list[dict], budget: MediaBudget) -> Optional[int]:
    """Videos a new turn can add (older clips give way, the opening context can't), or
    None if this endpoint has no known limit."""
    if budget.videos is None:
        return None
    i = context_index(messages)
    return max(0, budget.videos - (message_cost(messages[i]).videos if i is not None else 0))


def video_context_frames(base_url: Optional[str], model: str, plan, duration: float,
                         budget: MediaBudget) -> int:
    """How many frames a whole-video context adds. The provider samples it, so this is
    its measured rate where vlm.fyi has one, else the plan's estimate. It is counted at
    most up to `HIGH` of the budget: the context can't be shrunk, and if it is really
    larger the first request settles that."""
    from ambient.capabilities import resolve

    caps = resolve(base_url, model)
    n = None
    if caps and caps.capped and caps.frames_per_video:
        n = caps.frames_per_video
    elif caps and caps.frames_per_min and duration:
        n = math.ceil(caps.frames_per_min * duration / 60)
    if not n:
        n = plan.frames_estimate(duration) if duration else plan.max_frames
    return min(int(n), math.floor(budget.frames * HIGH))


# --- the opening context -----------------------------------------------------------

def frame_inline_bytes(frame) -> int:
    """Base64 bytes a sampled frame adds to the request: its data URL, or the file it
    will be inlined from; 0 for a hosted URL (e2b uploads)."""
    url = getattr(frame, "frame_url", None)
    if url:
        return _inline(url)
    path = getattr(frame, "frame_file_path", None)
    if path and os.path.exists(path):
        return math.ceil(os.path.getsize(path) * 4 / 3) + 23   # + "data:image/jpeg;base64,"
    return 0


def context_frame_cap(budget: MediaBudget, share: float) -> int:
    return max(1, math.floor(min(budget.images, budget.frames) * share))


def fit_context_frames(frames: list, budget: MediaBudget, share: float,
                       url_sizes: Optional[dict] = None) -> list:
    """Uniformly thin the opening frames so they fit `share` of the budget, counting
    their real encoded size (frame bytes vary ~5x between videos): inline bytes for
    data URLs, downloaded bytes for hosted URLs where the provider limits those."""
    n = min(len(frames), context_frame_cap(budget, share))
    url_sizes = url_sizes or {}
    inline = [frame_inline_bytes(f) for f in frames]
    hosted = [url_sizes.get(u, _URL_IMAGE_GUESS) if _hosted(u := (getattr(f, "frame_url", None) or "")) else 0
              for f in frames]
    url_limit = budget.url_bytes * share if budget.url_bytes is not None else math.inf

    def pick(k: int) -> list[int]:
        if k >= len(frames):
            return list(range(len(frames)))
        if k <= 1:
            return [0]
        return sorted({round(i * (len(frames) - 1) / (k - 1)) for i in range(k)})

    for k in range(n, 0, -1):
        idx = pick(k)
        if (sum(inline[i] for i in idx) <= budget.inline_bytes * share
                and sum(hosted[i] for i in idx) <= url_limit) or k == 1:
            return [frames[i] for i in idx]
    return frames[:1]

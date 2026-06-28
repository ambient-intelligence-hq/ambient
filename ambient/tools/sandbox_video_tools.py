"""E2B-backed VideoFrameTools client.

Mirrors the subset of `ambient.tools.video_tools.VideoFrameTools` that the agent
tools use (`fetch_clip`, `get_overview_frames`, `fetch_frames`, plus the preset
constants and `_duration_sec`), but runs the media work inside an E2B sandbox by
shelling out to `/app/main.py` (extract-frames / fetch-clip --upload-s3). Frames
and clips come back as S3 presigned URLs, so the host never needs the bytes; the
LLM call stays on the host.

The sync E2B `Sandbox` is used (no event-loop affinity), so this works whether
called from a sync method or an async one running in a worker thread.
"""
from __future__ import annotations
from typing import List, Optional, Tuple
from ambient.server.sandbox.interface import MediaSandbox
from ambient.config import settings
from ambient.utils.s3 import get_s3_client
from ambient import Clip, Frame


# Tile manifests are immutable once written for a (video, fps, dim), so cache the
# successful reads to avoid an S3 GET per clip call. Misses (tiling not done yet)
# are not cached, so a later-arriving manifest is picked up.
_MANIFEST_CACHE: dict[str, dict] = {}


def load_tile_manifest(video_id: str) -> Optional[dict]:
    """Return the tile manifest for *video_id* from S3, or None if absent."""
    cached = _MANIFEST_CACHE.get(video_id)
    if cached is not None:
        return cached
    manifest = get_s3_client().read_json(f"{video_id}/tiles/manifest.json")
    if manifest:
        _MANIFEST_CACHE[video_id] = manifest
    return manifest


class SandboxVideoFrameTools:
    """VideoFrameTools surface that runs media ops in an E2B sandbox."""

    # Presets mirror VideoFrameTools so callers that read them keep working.
    FPS = 1
    SKIM_FPS = 1
    SKIM_MAX_FRAMES = 25
    FOCUS_FPS = 2
    FOCUS_MAX_FRAMES = 30
    OVERVIEW_FPS = 2
    OVERVIEW_MAX_FRAMES = 75
    MAX_CLIP_DURATION_SEC = 600

    def __init__(self, video_id: str, max_frame_dimention: Optional[int] = None, *, box: MediaSandbox) -> None:
        self.video_id = video_id
        self.max_frame_dimention = max_frame_dimention
        self._box = box
        self._duration_sec: Optional[float] = None

    # ------------------------------------------------------------------ frames
    def _extract_frames(
        self,
        fps: int,
        start_time_sec: Optional[float] = None,
        end_time_sec: Optional[float] = None,
        max_frames: Optional[int] = None,
    ) -> List[Frame]:
        try:
            argv = ["extract-frames", "--video-id", self.video_id, "--fps", str(int(fps)), "--upload-s3"]
            if start_time_sec:
                argv += ["--start", str(start_time_sec)]
            if end_time_sec is not None:
                argv += ["--end", str(end_time_sec)]
            if max_frames is not None:
                argv += ["--max-frames", str(int(max_frames))]
            if self.max_frame_dimention is not None:
                argv += ["--max-dim", str(int(self.max_frame_dimention))]
            data = self._box.run(argv)
            if data.get("duration") is not None:
                self._duration_sec = data["duration"]
            return [
                Frame(
                    frame_url=f.get("frame_url"),
                    frame_file_path=f["frame_file_path"],
                    timestamp=f["timestamp"],
                    video_id=f["video_id"],
                    id=f["id"],
                )
                for f in data.get("frames", [])
            ]
        except Exception as e:
            print("error:", e)
            raise e

    def fetch_frames(
        self,
        fps: int = FPS,
        start_time_sec: float = 0,
        end_time_sec: Optional[float] = None,
        max_frames: Optional[int] = None,
    ) -> List[Frame]:
        return self._extract_frames(fps, start_time_sec, end_time_sec, max_frames)

    def get_overview_frames(self) -> List[Frame]:
        return self._extract_frames(self.OVERVIEW_FPS, max_frames=self.OVERVIEW_MAX_FRAMES)

    # -------------------------------------------------------------------- clip
    async def fetch_clip(
        self,
        start_time_sec: float,
        end_time_sec: float,
        fps: int = 5,
        crf: Optional[int] = None,
        max_size_mb: Optional[int] = None,
    ) -> Clip:
        # Fast path: if the video has been pre-transcoded into tiles matching the
        # requested quality, assemble the covering tiles (stream-copy, ~1s) instead
        # of transcoding on demand (~minutes). Any miss/mismatch/failure falls
        # through to the on-demand transcode below.
        try:
            fast = self._fetch_clip_from_tiles(start_time_sec, end_time_sec, fps, max_size_mb)
            if fast is not None:
                return fast
        except Exception as e:  # noqa: BLE001 - fall back to on-demand transcode
            print(f"[fetch_clip] tile fast path failed, falling back to transcode: {e}")

        argv = [
            "fetch-clip", "--video-id", self.video_id,
            "--start", str(start_time_sec), "--end", str(end_time_sec),
            "--fps", str(int(fps)), "--upload-s3",
        ]
        if crf is not None:
            argv += ["--crf", str(int(crf))]
        if max_size_mb is not None:
            argv += ["--max-size-mb", str(int(max_size_mb))]
        if self.max_frame_dimention is not None:
            argv += ["--max-dim", str(int(self.max_frame_dimention))]
        # Blocking call; we are already on a dedicated worker thread (the
        # dispatcher runs each tool in its own thread + loop).
        data = self._box.run(argv)
        c = data["clip"]
        return Clip(
            clip_url=c.get("clip_url"),
            clip_file_path=c.get("clip_file_path"),
            start_time=c["start_time"],
            end_time=c["end_time"],
            video_id=c["video_id"],
            id=c["id"],
        )

    async def fetch_clips(
        self,
        start_time_sec: float,
        end_time_sec: float,
        fps: int = 5,
        crf: Optional[int] = None,
        max_size_mb: Optional[int] = None,
    ) -> Tuple[List[Clip], float]:
        """Return ``(clips, global_offset)`` for a window.

        When a provider size cap (``max_size_mb``) is set and the video is tiled,
        the covering tiles are returned as separate clips — each is already under
        the cap, so a multi-minute window that a single re-encode could only fit by
        crushing quality is delivered at full tile quality. The clips are labelled
        on a single continuous 0-based timeline, so the model treats them as one
        clip and cites ``mm:ss`` within the window; ``global_offset`` (the window's
        true start) maps those local citations back to absolute video time.

        Otherwise (no cap, or not tiled / tiles too big) a single assembled or
        transcoded clip is returned, with its global start as the offset.
        """
        if max_size_mb is not None:
            covering = self._select_covering_tiles(start_time_sec, end_time_sec, fps)
            if covering and self._tiles_fit_cap(covering, max_size_mb):
                return self._tiles_as_local_clips(covering), covering[0]["start"]
        clip = await self.fetch_clip(start_time_sec, end_time_sec, fps, crf, max_size_mb)
        offset = clip.start_time if clip.start_time is not None else start_time_sec
        return [clip], offset

    def _select_covering_tiles(
        self, start_time_sec: float, end_time_sec: float, fps: int
    ) -> Optional[List[dict]]:
        """Tiles overlapping ``[start, end)``, or None if the fast path can't serve.

        None means: tiling disabled, no manifest yet, manifest quality (fps/dim)
        doesn't match the request, or no tile covers the window — in every case the
        caller falls back to an on-demand transcode.
        """
        if not settings.tiling_enabled:
            return None
        manifest = load_tile_manifest(self.video_id)
        if not manifest:
            return None
        # Tiles are baked at a fixed (fps, dim); only serve requests that match.
        if manifest.get("fps") != fps or manifest.get("max_dim") != self.max_frame_dimention:
            return None
        covering = [
            t for t in manifest.get("tiles", [])
            if t["end"] > start_time_sec and t["start"] < end_time_sec
        ]
        return covering or None

    @staticmethod
    def _tiles_fit_cap(covering: List[dict], max_size_mb: int) -> bool:
        """True if every tile fits the cap (each is sent as its own video_url)."""
        cap_bytes = max_size_mb * 1024 * 1024
        return all((t.get("size_bytes") or 0) <= cap_bytes for t in covering)

    def _tiles_as_local_clips(self, covering: List[dict]) -> List[Clip]:
        """Presign the covering tiles as clips on a continuous 0-based timeline.

        No box round-trip and no ffmpeg: each tile is already an S3 object, so we
        just presign it. start/end are window-local (tile0 0..d0, tile1 d0..d0+d1,
        …) so the model reads the sequence as one clip.
        """
        s3 = get_s3_client()
        clips: List[Clip] = []
        local = 0.0
        for t in covering:
            dur = t["end"] - t["start"]
            clips.append(Clip(
                clip_url=s3.get_presigned_url(t["key"], expires_in=7200),
                clip_file_path=None,
                start_time=local,
                end_time=local + dur,
                video_id=self.video_id,
                id=f"{self.video_id}_tile_{t['index']:05d}",
            ))
            local += dur
        return clips

    def _fetch_clip_from_tiles(
        self,
        start_time_sec: float,
        end_time_sec: float,
        fps: int,
        max_size_mb: Optional[int] = None,
    ) -> Optional[Clip]:
        """Assemble a single clip from pre-transcoded tiles, or None if not usable.

        The single-clip fast path (used when there's no size cap): stream-copy
        concat of the covering tiles in the box (~1s, no re-decode). Returns None
        (caller falls back to on-demand transcode) when the fast path can't serve,
        or when the concatenated tiles would exceed ``max_size_mb`` — a stream copy
        can't shrink, so only an on-demand re-encode can fit the cap. (The cap'd
        case is normally handled earlier by `fetch_clips`' multi-tile return.) On a
        hit the clip's start/end are snapped to tile boundaries (a superset of the
        requested window), which the caller uses as the citation offset.
        """
        covering = self._select_covering_tiles(start_time_sec, end_time_sec, fps)
        if not covering:
            return None

        if max_size_mb is not None:
            total_mb = sum((t.get("size_bytes") or 0) for t in covering) / 1024 / 1024
            if total_mb > max_size_mb:
                return None

        data = self._box.run([
            "concat-tiles", "--video-id", self.video_id,
            "--keys", ",".join(t["key"] for t in covering),
            "--start", str(covering[0]["start"]),
            "--end", str(covering[-1]["end"]),
            "--upload-s3",
        ])
        c = data["clip"]
        return Clip(
            clip_url=c.get("clip_url"),
            clip_file_path=c.get("clip_file_path"),
            start_time=c["start_time"],
            end_time=c["end_time"],
            video_id=c["video_id"],
            id=c["id"],
        )

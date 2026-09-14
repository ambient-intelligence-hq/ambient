"""Range-proxy media production for the agent tools (clips + frames + annotate).

Instead of e2b + pre-transcoded tiles, this produces media by running ffmpeg on
the host directly against the S3/R2 source (HTTP range reads — it fetches only the
requested window, not the whole file) and inlines the result as base64. No
sandbox, no tiling, no S3 media storage.

This is the host-local media backend (selected whenever no E2B sandbox is
active — see make_video_tools). It covers the full tool surface: `fetch_clips`
(search_clip/focus_clip), `fetch_frames` (grab_frames), `annotate_frame`
(draw_bounding_box/draw_point) and `get_overview_frames` (ingestion description).
"""
from __future__ import annotations

import asyncio
import base64
import concurrent.futures
import glob
import io
import os
import shutil
import subprocess
import tempfile
import time
from typing import List, Optional, Tuple

from ambient import Clip, Frame
from ambient.config import settings
from ambient.utils.s3 import get_s3_client

_RECONNECT = ["-reconnect", "1", "-reconnect_streamed", "1", "-reconnect_delay_max", "2"]


class RangeProxyVideoFrameTools:
    FPS = 2
    OVERVIEW_FPS = 2
    OVERVIEW_MAX_FRAMES = 75
    MAX_CLIP_DURATION_SEC = 600
    # Whole-video / sparse sampling: when the target frames are more than this many
    # seconds apart, seek to each instead of decoding the whole span end-to-end (cost
    # then scales with the frame COUNT, not the span duration).
    SEEK_GAP_SEC = 2.0
    SEEK_WORKERS = min(16, (os.cpu_count() or 4) + 4)

    _LOCAL_EXTS = (".mp4", ".mov", ".mkv", ".webm", ".m4v", ".avi")

    def __init__(self, video_id: str, max_frame_dimention: Optional[int] = None,
                 duration: Optional[float] = None) -> None:
        self.video_id = video_id
        self.max_dim = int(max_frame_dimention or settings.analysis_max_dim or 768)
        # Duration is probed + persisted at ingest; when the caller passes it we skip
        # the probe. None -> probe_duration() resolves it lazily (local/faststart).
        self._duration_sec: Optional[float] = duration

    def _local_source(self) -> Optional[str]:
        """Locally-cached source file if present on this host (store_video writes it).
        Reading from disk avoids HTTP entirely and keeps seeks fast even moov-at-end."""
        for ext in self._LOCAL_EXTS:
            p = os.path.join(settings.video_folder, f"{self.video_id}{ext}")
            if os.path.exists(p) and os.path.getsize(p) > 0:
                return p
        return None

    def _source_url(self) -> str:
        # Uploads (mp4) and youtube imports (remuxed to mp4) live here. Non-mp4
        # uploads would need an extension probe; not handled yet.
        key = f"{settings.s3_video_base_key}/{self.video_id}.mp4"
        return get_s3_client().get_presigned_url(key, expires_in=settings.source_url_ttl)

    def _source_input(self) -> str:
        """ffmpeg input for this video: the local cached file when present (fast, no
        HTTP), else the R2 presigned URL (faststart-normalized at ingest, so its range
        reads are cheap too)."""
        return self._local_source() or self._source_url()

    def _input_spec(self) -> Tuple[List[str], str]:
        """(reconnect_flags, input) for an ffmpeg command. The `-reconnect*` options
        are http(s)-only — passing them with a local-file input aborts ffmpeg with
        'Option reconnect not found' — so they're included only for the R2 URL."""
        local = self._local_source()
        if local:
            return [], local
        return list(_RECONNECT), self._source_url()

    # ------------------------------------------------------ seek-based sampling
    def _grab_one(self, ts: float, out_path: str) -> bool:
        """Extract a single frame at `ts` via a fast input-seek (`-ss` before `-i`
        jumps by the moov index, so it doesn't decode the whole file). One thread —
        many of these run concurrently."""
        reconnect, src = self._input_spec()
        cmd = ["ffmpeg", "-nostdin", "-y", "-loglevel", "error", *reconnect,
               "-ss", f"{ts:.3f}", "-i", src, "-frames:v", "1", "-threads", "1",
               "-vf", f"scale={self.max_dim}:-2", "-pix_fmt", "yuvj420p", "-q:v", "3",
               out_path]
        r = subprocess.run(cmd, capture_output=True)
        return r.returncode == 0 and os.path.exists(out_path) and os.path.getsize(out_path) > 0

    def _fetch_frames_seek(self, start: float, end: float, n: int) -> List[Frame]:
        """Sample `n` frames uniformly across [start, end] by seeking to each target
        time concurrently — decode cost scales with `n`, not the span duration. Used
        for sparse whole-video sampling; short dense windows use the single-decode
        path (contiguous, already fast)."""
        n = max(1, n)
        # Sample at the CENTER of n equal segments: gives exactly n points, all
        # strictly inside (start, end) — so no seek lands on EOF (which yields no
        # frame) and none coincide, regardless of container-vs-stream duration slack.
        span = end - start
        times = [start + (i + 0.5) * span / n for i in range(n)]
        tmpdir = tempfile.mkdtemp(prefix="rpseek_")
        try:
            def _work(item):
                i, ts = item
                out = os.path.join(tmpdir, f"s_{i:04d}.jpg")
                return (i, ts, out) if self._grab_one(ts, out) else None

            got: List[tuple] = []
            with concurrent.futures.ThreadPoolExecutor(max_workers=self.SEEK_WORKERS) as ex:
                for res in ex.map(_work, enumerate(times)):
                    if res:
                        got.append(res)
            got.sort()
            frames: List[Frame] = []
            for out_i, (_i, ts, out) in enumerate(got):
                with open(out, "rb") as fh:
                    data_url = "data:image/jpeg;base64," + base64.b64encode(fh.read()).decode()
                frames.append(Frame(frame_url=data_url, frame_file_path="",
                                    timestamp=round(ts, 3), video_id=self.video_id,
                                    id=f"{self.video_id}_rpf_{int(ts)}_{out_i}"))
            return frames
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

    def probe_duration(self) -> Optional[float]:
        """Total source duration (seconds) via a cheap ffprobe on the R2 URL,
        cached. Needed so whole-video sampling (fast mode's `end=None`) covers the
        entire clip instead of falling back to a tiny default window."""
        _start_time = time.time()
        if self._duration_sec is not None:
            return self._duration_sec
        try:
            r = subprocess.run(
                ["ffprobe", "-v", "error", "-show_entries", "format=duration",
                 "-of", "default=nw=1:nk=1", self._source_input()],
                capture_output=True, timeout=60)
            if r.returncode == 0:
                self._duration_sec = float(r.stdout.decode().strip())
        except Exception:  # noqa: BLE001
            pass
        print(f"[probe_duration] duration: {self._duration_sec}, time taken: {time.time() - _start_time}")
        return self._duration_sec

    # ------------------------------------------------------------------- clips
    def _produce_clip(self, start: float, end: float, fps: int, crf: Optional[int]) -> Tuple[List[Clip], float]:
        _start_time = time.time()
        dur = min(end - start, self.MAX_CLIP_DURATION_SEC)
        reconnect, src = self._input_spec()
        cmd = ["ffmpeg", "-nostdin", "-y", "-loglevel", "error", *reconnect,
               "-ss", str(start), "-i", src, "-t", str(dur),
               "-vf", f"scale={self.max_dim}:-2,fps={fps}",
               "-c:v", "libx264", "-crf", str(crf if crf is not None else 28), "-an",
               "-movflags", "frag_keyframe+empty_moov+default_base_moof", "-f", "mp4", "pipe:1"]
        proc = subprocess.run(cmd, capture_output=True)
        if proc.returncode != 0 or not proc.stdout:
            raise RuntimeError(f"range-proxy clip ffmpeg failed [{start},{end}]: "
                               f"{proc.stderr.decode('utf-8', 'replace')[:300]}")
        data_url = "data:video/mp4;base64," + base64.b64encode(proc.stdout).decode()
        clip = Clip(clip_url=data_url, clip_file_path=None, start_time=0.0, end_time=float(dur),
                    video_id=self.video_id, id=f"{self.video_id}_rp_{int(start)}_{int(end)}")
        print(f"[produce_clip] clip: {clip}, time taken: {time.time() - _start_time}")
        return [clip], float(start)

    async def fetch_clips(self, start_time_sec: float, end_time_sec: float,
                          fps: int = 5, crf: Optional[int] = None,
                          max_size_mb: Optional[int] = None) -> Tuple[List[Clip], float]:
        fps = int(fps or settings.analysis_fps or 2)
        _start_time = time.time()
        clips, start = await asyncio.to_thread(self._produce_clip, start_time_sec, end_time_sec, fps, crf)
        print(f"[fetch_clips] clips: {len(clips)}, time taken: {time.time() - _start_time}")
        return clips, start

    # ------------------------------------------------------------------ frames
    def fetch_frames(self, fps: Optional[int] = None, start_time_sec: float = 0,
                     end_time_sec: Optional[float] = None, max_frames: Optional[int] = None) -> List[Frame]:
        """Extract frames of [start,end] from the source (base64 data-URL frames).
        Called synchronously (grab_frames uses asyncio.to_thread)."""
        fps = int(fps or self.FPS)
        start = float(start_time_sec or 0)
        _start_time = time.time()
        if end_time_sec is not None:
            end = float(end_time_sec)
        else:
            # Whole-video request (e.g. fast-mode sampling): use the real duration.
            dur = self.probe_duration()
            end = float(dur) if dur else start + 5.0
        dur = max(0.1, end - start)

        # Sparse sampling over a long span (whole-video context): seek to each target
        # time instead of decoding the whole span and discarding most frames. Decode
        # cost then scales with the frame count, not the span duration.
        if max_frames and max_frames >= 2 and dur / max_frames > self.SEEK_GAP_SEC:
            frames = self._fetch_frames_seek(start, end, max_frames)
            print(f"[fetch_frames] frames: {len(frames)} (seek), time taken: {time.time() - _start_time}")
            return frames

        tmpdir = tempfile.mkdtemp(prefix="rpframes_")
        try:
            reconnect, src = self._input_spec()
            cmd = ["ffmpeg", "-nostdin", "-y", "-loglevel", "error", *reconnect,
                   "-ss", str(start), "-i", src, "-t", str(dur),
                   "-vf", f"scale={self.max_dim}:-2,fps={fps}",
                   # yuvj420p = JPEG full-range; without it, full-range-YUV sources
                   # make the mjpeg encoder abort ("Non full-range YUV is non-standard").
                   "-pix_fmt", "yuvj420p", "-q:v", "3",
                   os.path.join(tmpdir, "f_%04d.jpg")]
            r = subprocess.run(cmd, capture_output=True)
            if r.returncode != 0:
                raise RuntimeError(f"range-proxy frames ffmpeg failed: {r.stderr.decode('utf-8', 'replace')[:200]}")
            paths = sorted(glob.glob(os.path.join(tmpdir, "f_*.jpg")))
            if not paths:
                return []
            if max_frames and len(paths) > max_frames:
                sel = [0] if max_frames == 1 else [round(i * (len(paths) - 1) / (max_frames - 1))
                                                   for i in range(max_frames)]
            else:
                sel = list(range(len(paths)))
            frames: List[Frame] = []
            for out_i, k in enumerate(sel):
                with open(paths[k], "rb") as fh:
                    data_url = "data:image/jpeg;base64," + base64.b64encode(fh.read()).decode()
                frames.append(Frame(frame_url=data_url, frame_file_path="",
                                    timestamp=round(start + k / float(fps), 3),
                                    video_id=self.video_id, id=f"{self.video_id}_rpf_{int(start)}_{out_i}"))
            print(f"[fetch_frames] frames: {len(frames)}, time taken: {time.time() - _start_time}")
            return frames
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

    # ---------------------------------------------------------------- annotate
    def annotate_frame(self, timestamp: float, annotations: List[dict],
                       coord_scale: float = 1000.0, max_dim: Optional[int] = None) -> Optional[dict]:
        """Draw bounding boxes on the frame at `timestamp` (host PIL) and return
        {annotated_url, width, height}. Points are pre-converted to boxes upstream."""
        from PIL import Image, ImageDraw
        _start_time = time.time()
        boxed = [a for a in (annotations or []) if a.get("bounding_box")]
        if not boxed:
            return None
        dim = int(max_dim or self.max_dim)
        tmpdir = tempfile.mkdtemp(prefix="rpann_")
        try:
            out = os.path.join(tmpdir, "f.jpg")
            reconnect, src = self._input_spec()
            cmd = ["ffmpeg", "-nostdin", "-y", "-loglevel", "error", *reconnect,
                   "-ss", str(timestamp), "-i", src, "-frames:v", "1",
                   "-vf", f"scale={dim}:-2", "-pix_fmt", "yuvj420p", "-q:v", "2", out]
            r = subprocess.run(cmd, capture_output=True)
            if r.returncode != 0 or not os.path.exists(out):
                raise RuntimeError(f"range-proxy annotate ffmpeg failed: {r.stderr.decode('utf-8', 'replace')[:200]}")
            img = Image.open(out).convert("RGB")
            w, h = img.size
            draw = ImageDraw.Draw(img)
            for a in boxed:
                bb = a["bounding_box"]
                if len(bb) != 4:
                    continue
                y_min, x_min, y_max, x_max = bb
                box = [x_min / coord_scale * w, y_min / coord_scale * h,
                       x_max / coord_scale * w, y_max / coord_scale * h]
                draw.rectangle(box, outline=(255, 45, 60), width=3)
                lbl = a.get("label")
                if lbl:
                    draw.text((box[0] + 2, max(0, box[1] - 12)), str(lbl)[:48], fill=(255, 45, 60))
            buf = io.BytesIO()
            img.save(buf, format="JPEG", quality=85)
            data_url = "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()
            print(f"[annotate_frame] annotated_url: {data_url}, time taken: {time.time() - _start_time}")
            return {"annotated_url": data_url, "width": w, "height": h}
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

    def get_overview_frames(self) -> List[Frame]:
        """Ingestion overview: sample `OVERVIEW_MAX_FRAMES` uniformly across the
        whole video (same contract as the sandbox backend). `video_description`
        overrides `OVERVIEW_MAX_FRAMES` on the instance before calling; it also
        reads `self._duration_sec` afterwards, which `probe_duration()` populates."""
        dur = self.probe_duration()
        return self.fetch_frames(fps=self.OVERVIEW_FPS, start_time_sec=0,
                                 end_time_sec=dur, max_frames=self.OVERVIEW_MAX_FRAMES)

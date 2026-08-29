"""Range-proxy media production for the agent tools (clips + frames + annotate).

Instead of e2b + pre-transcoded tiles, this produces media by running ffmpeg on
the host directly against the S3/R2 source (HTTP range reads — it fetches only the
requested window, not the whole file) and inlines the result as base64. No
sandbox, no tiling, no S3 media storage.

Enabled with `settings.clip_backend == "rangeproxy"` (see make_video_tools).
Covers `fetch_clips` (search_clip/focus_clip), `fetch_frames` (grab_frames) and
`annotate_frame` (draw_bounding_box/draw_point). `get_overview_frames` (ingestion
description) is unchanged and still uses the e2b/inprocess path.
"""
from __future__ import annotations

import asyncio
import base64
import glob
import io
import os
import shutil
import subprocess
import tempfile
from typing import List, Optional, Tuple

from ambient import Clip, Frame
from ambient.config import settings
from ambient.utils.s3 import get_s3_client

_RECONNECT = ["-reconnect", "1", "-reconnect_streamed", "1", "-reconnect_delay_max", "2"]


class RangeProxyVideoFrameTools:
    FPS = 2
    MAX_CLIP_DURATION_SEC = 600

    def __init__(self, video_id: str, max_frame_dimention: Optional[int] = None) -> None:
        self.video_id = video_id
        self.max_dim = int(max_frame_dimention or settings.analysis_max_dim or 768)
        self._duration_sec: Optional[float] = None

    def _source_url(self) -> str:
        # Uploads (mp4) and youtube imports (remuxed to mp4) live here. Non-mp4
        # uploads would need an extension probe; not handled yet.
        key = f"{settings.s3_video_base_key}/{self.video_id}.mp4"
        return get_s3_client().get_presigned_url(key, expires_in=settings.source_url_ttl)

    # ------------------------------------------------------------------- clips
    def _produce_clip(self, start: float, end: float, fps: int, crf: Optional[int]) -> Tuple[List[Clip], float]:
        dur = min(end - start, self.MAX_CLIP_DURATION_SEC)
        cmd = ["ffmpeg", "-nostdin", "-y", "-loglevel", "error", *_RECONNECT,
               "-ss", str(start), "-i", self._source_url(), "-t", str(dur),
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
        return [clip], float(start)

    async def fetch_clips(self, start_time_sec: float, end_time_sec: float,
                          fps: int = 5, crf: Optional[int] = None,
                          max_size_mb: Optional[int] = None) -> Tuple[List[Clip], float]:
        fps = int(fps or settings.analysis_fps or 2)
        return await asyncio.to_thread(self._produce_clip, start_time_sec, end_time_sec, fps, crf)

    # ------------------------------------------------------------------ frames
    def fetch_frames(self, fps: Optional[int] = None, start_time_sec: float = 0,
                     end_time_sec: Optional[float] = None, max_frames: Optional[int] = None) -> List[Frame]:
        """Extract frames of [start,end] from the source (base64 data-URL frames).
        Called synchronously (grab_frames uses asyncio.to_thread)."""
        fps = int(fps or self.FPS)
        start = float(start_time_sec or 0)
        end = float(end_time_sec) if end_time_sec is not None else start + 5.0
        dur = max(0.1, end - start)
        tmpdir = tempfile.mkdtemp(prefix="rpframes_")
        try:
            cmd = ["ffmpeg", "-nostdin", "-y", "-loglevel", "error", *_RECONNECT,
                   "-ss", str(start), "-i", self._source_url(), "-t", str(dur),
                   "-vf", f"scale={self.max_dim}:-2,fps={fps}", "-q:v", "3",
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
            return frames
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

    # ---------------------------------------------------------------- annotate
    def annotate_frame(self, timestamp: float, annotations: List[dict],
                       coord_scale: float = 1000.0, max_dim: Optional[int] = None) -> Optional[dict]:
        """Draw bounding boxes on the frame at `timestamp` (host PIL) and return
        {annotated_url, width, height}. Points are pre-converted to boxes upstream."""
        from PIL import Image, ImageDraw

        boxed = [a for a in (annotations or []) if a.get("bounding_box")]
        if not boxed:
            return None
        dim = int(max_dim or self.max_dim)
        tmpdir = tempfile.mkdtemp(prefix="rpann_")
        try:
            out = os.path.join(tmpdir, "f.jpg")
            cmd = ["ffmpeg", "-nostdin", "-y", "-loglevel", "error", *_RECONNECT,
                   "-ss", str(timestamp), "-i", self._source_url(), "-frames:v", "1",
                   "-vf", f"scale={dim}:-2", "-q:v", "2", out]
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
            return {"annotated_url": data_url, "width": w, "height": h}
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

    def get_overview_frames(self, *a, **k):
        raise NotImplementedError("range-proxy backend: overview/description still uses e2b/inprocess")

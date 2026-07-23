#!/usr/bin/env python3
"""
Video media CLI – self-contained entry point for the E2B sandbox.

This sandbox does media work only: frame extraction and clip extraction, with
optional S3 upload. LLM analysis runs on the host, never here, so no LLM key is
ever plumbed into the box.

Output is a single JSON envelope fenced by RESULT_SENTINEL on stdout so the host
can parse it deterministically. Progress / debug messages go to stderr.

Settings are read from environment variables (matching ambient/config.py naming):
  VIDEO_FOLDER          – local directory where videos are cached  (default: /videos)
  S3_ENDPOINT           – S3 / R2 endpoint URL
  S3_BUCKET             – bucket name
  S3_VIDEO_BASE_KEY     – S3 key prefix under which videos live    (default: videos)
  AWS_ACCESS_KEY_ID
  AWS_SECRET_ACCESS_KEY

Video resolution:
  Each command takes --video-id instead of a raw file path.
  The CLI first checks VIDEO_FOLDER for a file matching <video-id>.<ext>.
  If nothing is found locally it downloads   S3_BUCKET/<S3_VIDEO_BASE_KEY>/<video-id>.mp4
  and saves it to VIDEO_FOLDER/<video-id>/<video-id>.mp4 before proceeding.

Usage examples:
  python main.py extract-frames --video-id myvideo --fps 1 --max-frames 50 --upload-s3
  python main.py fetch-clip     --video-id myvideo --start 10 --end 40 --upload-s3
"""

import argparse
import concurrent.futures
import glob
import importlib
import json
import logging
import math
import os
import shutil
import subprocess
import sys
from typing import Any, List, Optional, Tuple
from urllib.parse import urlparse
import time
import boto3
from botocore.config import Config
from pydantic import BaseModel

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Settings (env-var driven, mirrors ambient/config.py fields used by video_tools)
# ---------------------------------------------------------------------------
VIDEO_FOLDER = os.environ.get("VIDEO_FOLDER", "/videos")

S3_ENDPOINT = os.environ.get("S3_ENDPOINT")
S3_BUCKET = os.environ.get("S3_BUCKET")
S3_VIDEO_BASE_KEY = os.environ.get("S3_VIDEO_BASE_KEY", "videos")
AWS_ACCESS_KEY_ID = os.environ.get("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.environ.get("AWS_SECRET_ACCESS_KEY")

# Videos longer than this (seconds) use the ffmpeg backend unconditionally.
# decord's VideoReader builds a full seek index in RAM; for a 40-min 1080p video
# that index alone can exceed available memory and trigger an OOM kill.
DECORD_MAX_DURATION_SECS = int(os.environ.get("DECORD_MAX_DURATION_SECS", "300"))

# Parallelism for S3 frame uploads (frames dominate overview latency).
UPLOAD_CONCURRENCY = int(os.environ.get("UPLOAD_CONCURRENCY", "16"))

VIDEO_EXTENSIONS = (".mp4", ".mov", ".avi", ".mkv", ".webm", ".flv", ".wmv", ".mpeg", ".mpg", ".m4v", ".3gp", ".3g2", ".mj2")

# --- Source-video streaming (video I/O optimization) -----------------------
# Read the source video via a presigned URL with HTTP range reads instead of
# downloading the whole file: clips range-read only their window, the overview
# samples N frames via independent seeks. See docs/video-io-optimizations-spec.md.
STREAM_SOURCE_VIDEO = os.environ.get("STREAM_SOURCE_VIDEO", "true").lower() == "true"
SOURCE_URL_TTL = int(os.environ.get("SOURCE_URL_TTL", "7200"))
# Below this size a single download beats N range-reads (per-request overhead).
STREAM_MIN_BYTES = int(os.environ.get("STREAM_MIN_BYTES", str(100 * 1024 * 1024)))
# Parallel seeks for the overview sampler (independent range reads).
OVERVIEW_SEEK_CONCURRENCY = int(os.environ.get("OVERVIEW_SEEK_CONCURRENCY", "16"))

# --- YouTube imports -------------------------------------------------------
YOUTUBE_MAX_HEIGHT = int(os.environ.get("YOUTUBE_MAX_HEIGHT", "720"))
YOUTUBE_MAX_DURATION_SECONDS = int(os.environ.get("YOUTUBE_MAX_DURATION_SECONDS", str(3 * 60 * 60)))
YOUTUBE_MAX_SIZE_BYTES = int(os.environ.get("YOUTUBE_MAX_SIZE_BYTES", str(5 * 1024 * 1024 * 1024)))
YOUTUBE_DOWNLOAD_TIMEOUT_SECONDS = int(os.environ.get("YOUTUBE_DOWNLOAD_TIMEOUT_SECONDS", "900"))
YOUTUBE_HOSTS = {"youtube.com", "www.youtube.com", "m.youtube.com", "youtu.be"}

# --- Pre-transcoded clip tiles ---------------------------------------------
# At ingestion the whole video is transcoded once into fixed, GOP-aligned tiles
# at the target provider quality and uploaded to S3 (with a manifest). fetch_clip
# then assembles the covering tiles instead of transcoding on demand. See the
# `transcode-tiles` / `concat-tiles` sub-commands below.
TILE_SECONDS = int(os.environ.get("TILE_SECONDS", "45"))
TILE_FPS = int(os.environ.get("TILE_FPS", "2"))
TILE_MAX_DIM = int(os.environ.get("TILE_MAX_DIM", "768"))
TILE_WORKERS = int(os.environ.get("TILE_WORKERS", str(os.cpu_count() or 4)))


def _scale_even_filter(dim: int) -> str:
    """ffmpeg -vf scale that fits within dim x dim (aspect preserved) AND forces
    even width/height.

    `force_original_aspect_ratio=decrease` alone can yield an odd side (e.g.
    573x768), which libx264 rejects ("width not divisible by 2"). The trailing
    ``scale=trunc(iw/2)*2:trunc(ih/2)*2`` snaps both sides down to even. Written as
    a second scale (not `force_divisible_by`, which needs ffmpeg >= 4.4) so it
    works on any ffmpeg build in the sandbox base image.
    """
    return (
        f"scale={dim}:{dim}:force_original_aspect_ratio=decrease,"
        "scale=trunc(iw/2)*2:trunc(ih/2)*2"
    )


# ---------------------------------------------------------------------------
# S3 client (self-contained port of ambient/utils/s3.py)
# ---------------------------------------------------------------------------

class S3Client:
    def __init__(
        self,
        endpoint: Optional[str] = None,
        bucket: Optional[str] = None,
        access_key: Optional[str] = None,
        secret_key: Optional[str] = None,
    ):
        self.bucket = bucket or S3_BUCKET
        if not self.bucket:
            raise ValueError("S3_BUCKET is required (set env var or pass --s3-bucket)")
        self._s3 = boto3.client(
            "s3",
            endpoint_url=endpoint or S3_ENDPOINT,
            aws_access_key_id=access_key or AWS_ACCESS_KEY_ID,
            aws_secret_access_key=secret_key or AWS_SECRET_ACCESS_KEY,
            config=Config(signature_version="s3v4"),
        )
    
    def exists(self, key: str) -> bool:
        return self._s3.head_object(Bucket=self.bucket, Key=key) is not None

    def head(self, key: str) -> Optional[dict]:
        """Return the object's HEAD metadata, or None if it doesn't exist."""
        try:
            return self._s3.head_object(Bucket=self.bucket, Key=key)
        except Exception:  # 404s (and any access error) -> treat as absent
            return None

    def read_head_bytes(self, key: str, nbytes: int) -> bytes:
        """Read the first nbytes of an object (cheap container-format probe)."""
        resp = self._s3.get_object(Bucket=self.bucket, Key=key, Range=f"bytes=0-{nbytes - 1}")
        return resp["Body"].read()

    def upload_file(self, local_path: str, key: str, content_type: str = "application/octet-stream") -> str:
        self._s3.upload_file(
            local_path,
            self.bucket,
            key,
            ExtraArgs={"ContentType": content_type, "Metadata": {"source": "e2b-sandbox"}},
        )
        return f"s3://{self.bucket}/{key}"

    def get_presigned_url(self, key: str, expires_in: int = 3600) -> str:
        return self._s3.generate_presigned_url(
            "get_object",
            Params={"Bucket": self.bucket, "Key": key},
            ExpiresIn=expires_in,
        )

    def upload_and_presign(self, local_path: str, key: str, content_type: str, expires_in: int = 3600) -> tuple[str, str]:
        s3_uri = self.upload_file(local_path, key, content_type)
        url = self.get_presigned_url(key, expires_in)
        return s3_uri, url

    def download_file(self, key: str, local_path: str) -> None:
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        print(f"[s3] downloading s3://{self.bucket}/{key} → {local_path}", file=sys.stderr)
        self._s3.download_file(self.bucket, key, local_path)

    def put_json(self, key: str, obj: dict) -> None:
        self._s3.put_object(
            Bucket=self.bucket,
            Key=key,
            Body=json.dumps(obj).encode("utf-8"),
            ContentType="application/json",
        )

    def read_json(self, key: str) -> Optional[dict]:
        try:
            resp = self._s3.get_object(Bucket=self.bucket, Key=key)
            return json.loads(resp["Body"].read())
        except Exception:  # missing key / access error -> treat as absent
            return None


def _make_s3_client() -> Optional["S3Client"]:
    """Return an S3Client if all required env vars are present, else None."""
    if S3_BUCKET and (S3_ENDPOINT or AWS_ACCESS_KEY_ID):
        return S3Client()
    return None


# ---------------------------------------------------------------------------
# Video resolution: local cache → S3 fallback
# ---------------------------------------------------------------------------

def _probe_mp4_layout(s3: "S3Client", key: str, probe_bytes: int = 2 * 1024 * 1024) -> Tuple[bool, bool]:
    """Return ``(fragmented, faststart)`` for an mp4/mov by reading its head.

    * **fragmented** — a `moof` box near the front (fMP4/DASH). It has no sample
      table in `moov`; seeking/duration require the `mfra` index at the end, which
      over HTTP degrades to reading ~the whole file. Must be downloaded.
    * **faststart** — `moov` precedes `mdat` (index at the front). Only faststart
      files HTTP-seek cheaply: the overview opens the file 50× (one per seek), and
      with `moov` at the *end* each open re-downloads the (large) tail index — so
      moov-at-end streaming is pathologically slow and must be downloaded too.
    """
    try:
        head = s3.read_head_bytes(key, probe_bytes)
    except Exception:
        return (True, False)  # can't probe -> treat as not-streamable (download)
    fragmented = b"moof" in head
    moov = head.find(b"moov")
    mdat = head.find(b"mdat")
    faststart = (moov != -1) and (mdat == -1 or moov < mdat)
    return (fragmented, faststart)


def _probe_local_mp4_layout(path: str, probe_bytes: int = 2 * 1024 * 1024) -> Tuple[bool, bool]:
    """Return ``(fragmented, faststart)`` for a local mp4/mov by reading its head."""
    with open(path, "rb") as fh:
        head = fh.read(probe_bytes)
    fragmented = b"moof" in head
    moov = head.find(b"moov")
    mdat = head.find(b"mdat")
    faststart = (moov != -1) and (mdat == -1 or moov < mdat)
    return (fragmented, faststart)


def _is_youtube_url(url: str) -> bool:
    parsed = urlparse(url or "")
    host = (parsed.hostname or "").lower()
    return parsed.scheme in {"http", "https"} and host in YOUTUBE_HOSTS


def _probe_source_key(s3: "S3Client", video_id: str) -> Tuple[str, int]:
    """Find the source object key for *video_id* and its size in bytes.

    Probes known extensions (mp4 first) under <S3_VIDEO_BASE_KEY>/. Raises
    FileNotFoundError if no object exists.
    """
    for ext in VIDEO_EXTENSIONS:
        key = f"{S3_VIDEO_BASE_KEY}/{video_id}{ext}"
        head = s3.head(key)
        if head is not None:
            return key, int(head.get("ContentLength", 0))
    raise FileNotFoundError(
        f"Video '{video_id}' not found: no object under "
        f"s3://{s3.bucket}/{S3_VIDEO_BASE_KEY}/{video_id}.* "
        f"(tried {len(VIDEO_EXTENSIONS)} extensions)"
    )


def resolve_video_source(video_id: str) -> Tuple[str, bool]:
    """Return ``(source, is_url)`` for ffmpeg/ffprobe to read.

    Resolution order:
      1. Local cache file ``<video_id>.<ext>`` in VIDEO_FOLDER[/<video_id>/].
      2. S3 object ``<S3_VIDEO_BASE_KEY>/<video_id>.<ext>``:
           * if STREAM_SOURCE_VIDEO and the object is >= STREAM_MIN_BYTES, return
             a presigned GET URL — ffmpeg range-reads it, no download;
           * otherwise download to the local cache and return the path.

    ``is_url=True`` tells callers to force the ffmpeg backend (decord can't
    range-read HTTP) and to add reconnect flags to ffmpeg/ffprobe.
    """
    print("[resolve_video_source] starting resolve_video_source", file=sys.stderr)
    start_time = time.time()
    # 1. Local cache: an actual *file* named <video_id>.<ext>. (Returning a
    #    directory here would feed the folder to ffmpeg as -i.)
    search_dirs = [VIDEO_FOLDER, os.path.join(VIDEO_FOLDER, video_id)]
    for directory in search_dirs:
        if not os.path.isdir(directory):
            continue
        for ext in VIDEO_EXTENSIONS:
            candidate = os.path.join(directory, f"{video_id}{ext}")
            if os.path.isfile(candidate):
                print(f"[video] found locally: {candidate}", file=sys.stderr)
                return candidate, False
    print(f"[resolve_video_source] local cache search time: {time.time() - start_time} seconds", file=sys.stderr)
    # 2. S3: probe extensions (mp4 first) so non-mp4 uploads resolve too.
    s3 = _make_s3_client()
    if s3 is None:
        raise FileNotFoundError(
            f"Video '{video_id}' not found in {VIDEO_FOLDER!r} and S3 is not configured "
            f"(set S3_BUCKET + S3_ENDPOINT / AWS_ACCESS_KEY_ID)."
        )
    start_probe_time = time.time()
    key, size = _probe_source_key(s3, video_id)
    size_mb = size / 1024 / 1024
    ext = os.path.splitext(key)[1].lower()
    print(f"[resolve_video_source] probe time: {time.time() - start_probe_time} seconds", file=sys.stderr)
    start_streamable_time = time.time()
    if STREAM_SOURCE_VIDEO and size >= STREAM_MIN_BYTES:
        # Stream only a faststart, non-fragmented MP4 — the only layout that
        # HTTP-seeks cheaply across the 50-seek overview. Fragmented or
        # moov-at-end files fall through to a one-time download, after which the
        # overview seek-samples the *local* file (instant disk seeks) and the
        # download is cached in the box for any later clip ops.
        if ext in (".mp4", ".mov", ".m4v"):
            fragmented, faststart = _probe_mp4_layout(s3, key)
            streamable = (not fragmented) and faststart
            not_streamable_reason = "fragmented" if fragmented else "moov-at-end (not faststart)"
        else:
            streamable, not_streamable_reason = False, f"{ext or 'unknown'} container"
        if streamable:
            url = s3.get_presigned_url(key, expires_in=SOURCE_URL_TTL)
            print(
                f"[video] streaming source via presigned URL "
                f"(s3://{s3.bucket}/{key}, {size_mb:.1f} MB)",
                file=sys.stderr,
            )
            print(f"[resolve_video_source] streamable time: {time.time() - start_streamable_time} seconds", file=sys.stderr)
            return url, True
        print(
            f"[video] {not_streamable_reason} — HTTP multi-seek is slow; "
            f"downloading for local seek ({size_mb:.1f} MB)",
            file=sys.stderr,
        )
    elif not STREAM_SOURCE_VIDEO:
        print(f"[video] streaming disabled; downloading source ({size_mb:.1f} MB)", file=sys.stderr)
    else:
        print(f"[video] below stream threshold; downloading source ({size_mb:.1f} MB)", file=sys.stderr)

    local_path = os.path.join(VIDEO_FOLDER, video_id, f"{video_id}{ext}")
    start_download_time = time.time()
    s3.download_file(key, local_path)
    print(f"[resolve_video_source] [s3] s3 download time: {time.time() - start_download_time} seconds", file=sys.stderr)
    print(f"[resolve_video_source] Total time taken: {time.time() - start_time} seconds", file=sys.stderr)
    return local_path, False


# ---------------------------------------------------------------------------
# Lightweight data classes (mirrors ambient/__init__.py models)
# ---------------------------------------------------------------------------

class Frame:
    def __init__(self, frame_file_path: str, timestamp: float, video_id: str, id: str):
        self.frame_file_path = frame_file_path
        self.timestamp = timestamp
        self.video_id = video_id
        self.id = id
        self.frame_url: Optional[str] = None   # populated after S3 upload

    def to_dict(self) -> dict:
        d = {
            "id": self.id,
            "video_id": self.video_id,
            "timestamp": self.timestamp,
            "frame_file_path": self.frame_file_path,
        }
        if self.frame_url:
            d["frame_url"] = self.frame_url
        return d


class Clip:
    def __init__(self, clip_file_path: str, start_time: float, end_time: float, video_id: str, id: str):
        self.clip_file_path = clip_file_path
        self.start_time = start_time
        self.end_time = end_time
        self.video_id = video_id
        self.id = id
        self.clip_url: Optional[str] = None    # populated after S3 upload

    def to_dict(self) -> dict:
        d = {
            "id": self.id,
            "video_id": self.video_id,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "clip_file_path": self.clip_file_path,
        }
        if self.clip_url:
            d["clip_url"] = self.clip_url
        return d


# ---------------------------------------------------------------------------
# Structured output envelope
#
# stdout is the control plane between this CLI and the host caller. Everything
# the host needs is emitted as a single JSON envelope fenced by RESULT_SENTINEL,
# so the host can extract it deterministically even if libraries print stray
# lines to stdout. All progress / debug logging goes to stderr.
# ---------------------------------------------------------------------------

RESULT_SENTINEL = "===AMBIENT_RESULT==="


class FrameOut(BaseModel):
    id: str
    video_id: str
    timestamp: float
    frame_file_path: str
    frame_url: Optional[str] = None


class ClipOut(BaseModel):
    id: str
    video_id: str
    start_time: float
    end_time: float
    clip_file_path: str
    clip_url: Optional[str] = None


class FramesResult(BaseModel):
    frames: List[FrameOut]
    duration: Optional[float] = None


class ClipResult(BaseModel):
    clip: ClipOut


class TileEntry(BaseModel):
    index: int
    start: float
    end: float
    key: str
    size_bytes: Optional[int] = None


class TilesResult(BaseModel):
    video_id: str
    fps: int
    max_dim: int
    tile_seconds: int
    duration: float
    tiles: List[TileEntry]


class PrepareYoutubeResult(BaseModel):
    video_id: str
    source_file_path: str
    r2_key: Optional[str] = None
    s3_uri: Optional[str] = None
    size_bytes: int
    duration: Optional[float] = None
    width: Optional[int] = None
    height: Optional[int] = None
    source_kind: str = "youtube"
    download_info: dict[str, Any]


class ProbeYoutubeResult(BaseModel):
    video_id: str
    title: Optional[str] = None
    duration: Optional[float] = None
    width: Optional[int] = None
    height: Optional[int] = None
    webpage_url: Optional[str] = None
    extractor: Optional[str] = None


class UploadSourceResult(BaseModel):
    video_id: str
    r2_key: str
    s3_uri: str
    size_bytes: int


class ResultError(BaseModel):
    code: str
    message: str


class ResultEnvelope(BaseModel):
    ok: bool
    data: Optional[dict] = None
    error: Optional[ResultError] = None


def emit(data: Optional[BaseModel] = None, error: Optional[ResultError] = None) -> None:
    """Write the single result envelope to stdout and exit.

    Success exits 0, failure exits 1. The payload is fenced by RESULT_SENTINEL
    on both sides so the host can recover it regardless of surrounding stdout.
    """
    envelope = ResultEnvelope(
        ok=error is None,
        data=data.model_dump() if data is not None else None,
        error=error,
    )
    sys.stdout.write(f"{RESULT_SENTINEL}{envelope.model_dump_json()}{RESULT_SENTINEL}\n")
    sys.stdout.flush()
    sys.exit(0 if error is None else 1)


# ---------------------------------------------------------------------------
# VideoFrameTools (self-contained port of ambient/tools/video_tools.py)
# ---------------------------------------------------------------------------

class VideoFrameTools:
    FPS = 1
    SKIM_FPS = 1
    SKIM_MAX_FRAMES = 25
    FOCUS_FPS = 2
    FOCUS_MAX_FRAMES = 30
    OVERVIEW_FPS = 2
    OVERVIEW_MAX_FRAMES = 75
    MAX_CLIP_DURATION_SEC = 600

    def __init__(self, video_id: str, max_frame_dimension: Optional[int] = None):
        self.video_id = video_id
        # `video_path` is a local file path or a presigned URL; `is_url` selects
        # the ffmpeg backend + reconnect flags for remote (range-read) sources.
        self.video_path, self.is_url = resolve_video_source(video_id)
        # Mirror ambient/tools/video_tools.py: VIDEO_FOLDER/<video_id>/frames/
        self.frame_dir = os.path.join(VIDEO_FOLDER, video_id, "frames")
        os.makedirs(self.frame_dir, exist_ok=True)
        self.max_frame_dimension = max_frame_dimension
        self._vr = None
        self._backend: Optional[str] = None
        self._avg_fps: Optional[float] = None
        self._duration_sec: Optional[float] = None

    # ------------------------------------------------------------------
    # Backend initialisation (decord preferred, ffmpeg fallback)
    # ------------------------------------------------------------------

    def _reconnect_flags(self) -> List[str]:
        """ffmpeg/ffprobe input options to survive transient drops on a remote
        (presigned-URL) source. Must precede ``-i``. Empty for local files."""
        if not self.is_url:
            return []
        return ["-reconnect", "1", "-reconnect_streamed", "1", "-reconnect_delay_max", "5"]

    def _probe_video_info(self) -> tuple[float, float]:
        cmd = [
            "ffprobe", "-v", "error", "-select_streams", "v:0",
            "-show_entries", "stream=avg_frame_rate,duration:format=duration",
            "-of", "default=noprint_wrappers=1",
            *self._reconnect_flags(),
            self.video_path,
        ]
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        avg_frame_rate = ""
        duration = 0.0
        for line in result.stdout.splitlines():
            if line.startswith("avg_frame_rate="):
                avg_frame_rate = line.split("=", 1)[1].strip()
            elif line.startswith("duration="):
                try:
                    duration = float(line.split("=", 1)[1].strip())
                except ValueError:
                    pass
        fps = 1.0
        if avg_frame_rate and avg_frame_rate != "0/0":
            if "/" in avg_frame_rate:
                num, den = avg_frame_rate.split("/", 1)
                fps = float(num) / float(den) if float(den) != 0 else 1.0
            else:
                fps = float(avg_frame_rate)
        if fps <= 0:
            fps = 1.0
        return duration, fps

    def _ensure_backend(self) -> None:
        if self._backend is not None:
            return

        # Always probe duration cheaply via ffprobe first so we can decide
        # whether decord is safe to use before loading it (decord builds a full
        # seek index in RAM; on long videos that alone causes an OOM kill).
        duration, fps = self._probe_video_info()

        # Remote sources must use ffmpeg: decord does random-access reads to build
        # its seek index, which over HTTP is a storm of tiny range GETs.
        if self.is_url or duration > DECORD_MAX_DURATION_SECS:
            skip_reason = (
                "remote source" if self.is_url
                else f"video > {DECORD_MAX_DURATION_SECS}s threshold"
            )
            print(
                f"[backend] ffmpeg  fps={fps:.2f}  duration={duration:.2f}s"
                f"  (decord skipped: {skip_reason})",
                file=sys.stderr,
            )
            self._duration_sec = duration
            self._avg_fps = fps
            self._backend = "ffmpeg"
            return

        try:
            decord = importlib.import_module("decord")
            self._vr = decord.VideoReader(self.video_path, ctx=decord.cpu(0))
            self._avg_fps = float(self._vr.get_avg_fps()) or 1.0
            self._duration_sec = len(self._vr) / self._avg_fps
            self._backend = "decord"
            print(f"[backend] decord  fps={self._avg_fps:.2f}  duration={self._duration_sec:.2f}s", file=sys.stderr)
        except ModuleNotFoundError:
            self._duration_sec = duration
            self._avg_fps = fps
            self._backend = "ffmpeg"
            print(f"[backend] ffmpeg  fps={fps:.2f}  duration={duration:.2f}s", file=sys.stderr)

    def _video_duration_seconds(self) -> float:
        self._ensure_backend()
        return self._duration_sec

    # ------------------------------------------------------------------
    # Frame-cache helpers
    # ------------------------------------------------------------------

    def _fps_cache_dir(
        self,
        fps: int,
        start_time: Optional[float],
        duration_sec: Optional[float],
        max_frames: Optional[int] = None,
    ) -> str:
        dim_label = str(self.max_frame_dimension) if self.max_frame_dimension is not None else "orig"
        folder = f"fps_{fps}_dim_{dim_label}"
        if start_time is not None and duration_sec is not None:
            folder += f"_start_{start_time}_duration_{duration_sec}"
        if max_frames is not None:
            folder += f"_max{max_frames}"
        cache_dir = os.path.join(self.frame_dir, folder)
        os.makedirs(cache_dir, exist_ok=True)
        return cache_dir

    @staticmethod
    def _uniform_sample_indices(values: List[int], target_size: int) -> List[int]:
        if target_size <= 0:
            return []
        if target_size >= len(values):
            return values
        if target_size == 1:
            return [values[0]]
        last = len(values) - 1
        return [values[int(i * last / (target_size - 1))] for i in range(target_size)]

    def _ensure_frames_cache(
        self,
        fps: int,
        start_time: Optional[float] = None,
        duration_sec: Optional[float] = None,
        max_frames: Optional[int] = None,
    ) -> List[Tuple[str, float]]:
        """Return [(frame_path, timestamp_sec), ...] for the requested window.

        Timestamps are the frames' true positions in the source video. When the
        window is capped to `max_frames`, frames are sampled *uniformly across the
        whole window* (not truncated to the first N), so the timestamps are
        persisted alongside the PNGs (frames are renumbered sequentially and can't
        be recovered from filenames alone).
        """
        self._ensure_backend()
        cache_dir = self._fps_cache_dir(fps, start_time, duration_sec, max_frames)
        done_marker = os.path.join(cache_dir, ".done")
        times_path = os.path.join(cache_dir, "times.json")

        if os.path.exists(done_marker):
            cached = sorted(glob.glob(os.path.join(cache_dir, "frame_*.png")))
            if cached:
                return list(zip(cached, self._load_times(times_path, cached, fps, start_time)))

        for existing in glob.glob(os.path.join(cache_dir, "frame_*.png")):
            try:
                os.remove(existing)
            except OSError:
                pass
        if os.path.exists(done_marker):
            os.remove(done_marker)

        if max_frames is not None and self._backend == "ffmpeg":
            # Bounded sample (overview) on the ffmpeg backend — i.e. a streamed
            # URL or a long (>DECORD_MAX_DURATION_SECS) local file. Seek-sample
            # instead of a single full-timeline `-vf fps` pass: over HTTP each
            # seek is a range read; on a local file each `-ss` is an instant disk
            # seek. Either way it avoids demuxing the whole video.
            timestamps = self._extract_frames_seek(fps, start_time, duration_sec, cache_dir, max_frames)
        elif self._backend == "decord" and self._vr is not None:
            timestamps = self._extract_frames_decord(fps, start_time, duration_sec, cache_dir, max_frames)
        else:
            timestamps = self._extract_frames_ffmpeg(fps, start_time, duration_sec, cache_dir, max_frames)

        paths = sorted(glob.glob(os.path.join(cache_dir, "frame_*.png")))
        # Guard against any count drift (e.g. ffmpeg emitting +/-1 frame).
        if len(timestamps) != len(paths):
            base = start_time or 0.0
            timestamps = [base + i / float(fps) for i in range(len(paths))]
        with open(times_path, "w", encoding="utf-8") as fh:
            json.dump(timestamps, fh)
        with open(done_marker, "w", encoding="utf-8") as fh:
            fh.write("ok")
        return list(zip(paths, timestamps))

    @staticmethod
    def _load_times(
        times_path: str, paths: List[str], fps: int, start_time: Optional[float]
    ) -> List[float]:
        try:
            with open(times_path, encoding="utf-8") as fh:
                times = json.load(fh)
            if len(times) == len(paths):
                return times
        except (OSError, ValueError):
            pass
        base = start_time or 0.0
        return [base + i / float(fps) for i in range(len(paths))]

    # Maximum frames decoded and held in RAM at once by the decord backend.
    # At 1080p a single RGB frame is ~6 MB; 32 frames ≈ 200 MB — well within budget.
    DECORD_BATCH_SIZE = 32

    def _extract_frames_decord(
        self,
        fps: int,
        start_time: Optional[float],
        duration_sec: Optional[float],
        cache_dir: str,
        max_frames: Optional[int] = None,
    ) -> List[float]:
        print(
            f"[decord] extracting  fps={fps}  start={start_time}  duration={duration_sec}"
            + (f"  max_frames={max_frames}" if max_frames else ""),
            file=sys.stderr,
        )
        native_fps = self._avg_fps
        total_frames = len(self._vr)
        if total_frames == 0:
            return []
        step_sec = 1.0 / float(fps)
        t = start_time if start_time is not None else 0.0
        end_sec = (t + duration_sec) if duration_sec is not None else self._duration_sec
        # Keep each sample's true time paired with its frame index, so capping
        # below preserves accurate timestamps.
        seen: set = set()
        pairs: List[Tuple[float, int]] = []
        while t < end_sec:
            idx = int(round(t * native_fps))
            idx = max(0, min(idx, total_frames - 1))
            if idx not in seen:
                seen.add(idx)
                pairs.append((t, idx))
            t += step_sec

        # Apply max_frames early (uniformly across the window) so we never
        # allocate more than needed.
        if max_frames is not None and len(pairs) > max_frames:
            pairs = self._uniform_sample_indices(pairs, max_frames)

        timestamps = [p[0] for p in pairs]
        frame_indices = [p[1] for p in pairs]

        print(f"[decord] {len(frame_indices)} frames to decode", file=sys.stderr)

        PIL_Image = importlib.import_module("PIL.Image")
        out_idx = 0
        # Decode in small batches to keep peak memory bounded.
        for batch_start in range(0, len(frame_indices), self.DECORD_BATCH_SIZE):
            batch_indices = frame_indices[batch_start: batch_start + self.DECORD_BATCH_SIZE]
            batch = self._vr.get_batch(batch_indices).asnumpy()
            for rgb_frame in batch:
                image = PIL_Image.fromarray(rgb_frame)
                if self.max_frame_dimension is not None:
                    image.thumbnail(
                        (self.max_frame_dimension, self.max_frame_dimension),
                        PIL_Image.Resampling.LANCZOS,
                    )
                image.save(os.path.join(cache_dir, f"frame_{out_idx:09d}.png"), format="PNG")
                out_idx += 1
            del batch  # release numpy array before the next batch
        return timestamps

    def _extract_frames_seek(
        self,
        fps: int,
        start_time: Optional[float],
        duration_sec: Optional[float],
        cache_dir: str,
        max_frames: int,
    ) -> List[float]:
        """Sample ``max_frames`` frames across the span via independent ``-ss`` seeks.

        For a remote (presigned-URL) source a single ``-vf fps=N`` pass would
        demux the whole span; instead we issue one HTTP-range seek per sample
        point (each reads ~one GOP) and run them in parallel. Timestamps are the
        *requested* sample times — the decoded frame is the nearest keyframe, so
        they're approximate, which is fine for an overview.
        """
        self._ensure_backend()  # ensure self._duration_sec for open-ended spans
        base = start_time or 0.0
        if duration_sec is not None and duration_sec > 0:
            span = duration_sec
        elif self._duration_sec is not None:
            span = max(0.0, self._duration_sec - base)
        else:
            span = 0.0
        if span <= 0 or max_frames <= 0:
            return []

        # Sample at window midpoints so no seek lands exactly on EOF (which would
        # decode no frame), while still covering the span uniformly.
        timestamps = [base + span * (i + 0.5) / max_frames for i in range(max_frames)]
        workers = min(OVERVIEW_SEEK_CONCURRENCY, len(timestamps))
        print(
            f"[seek] sampling {len(timestamps)} frames over [{base:.1f}, {base + span:.1f}]s "
            f"from {'remote URL' if self.is_url else 'local file'} ({workers} workers)",
            file=sys.stderr,
        )

        def _grab(item: Tuple[int, float]) -> None:
            idx, t = item
            out_path = os.path.join(cache_dir, f"frame_{idx:09d}.png")
            cmd = [
                "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
                "-ss", str(t), *self._reconnect_flags(),
                "-i", self.video_path, "-frames:v", "1",
            ]
            if self.max_frame_dimension is not None:
                cmd += [
                    "-vf",
                    _scale_even_filter(self.max_frame_dimension),
                ]
            cmd.append(out_path)
            try:
                subprocess.run(cmd, check=True, capture_output=True, text=True)
            except subprocess.CalledProcessError as exc:
                # A single failed seek shouldn't sink the whole overview; the
                # missing frame is simply dropped below.
                tail = (exc.stderr or "")[-200:]
                print(f"[seek] frame at {t:.1f}s failed: {tail}", file=sys.stderr)

        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
            list(pool.map(_grab, enumerate(timestamps)))

        # Keep timestamps whose frame was actually written, in index order, so
        # they align with sorted(glob('frame_*.png')) in the caller.
        return [
            t
            for idx, t in enumerate(timestamps)
            if os.path.isfile(os.path.join(cache_dir, f"frame_{idx:09d}.png"))
        ]

    def _extract_frames_ffmpeg(
        self,
        fps: int,
        start_time: Optional[float],
        duration_sec: Optional[float],
        cache_dir: str,
        max_frames: Optional[int] = None,
    ) -> List[float]:
        base = start_time or 0.0
        # Window length, used both to cap frames uniformly and to time them.
        if duration_sec is not None and duration_sec > 0:
            window = duration_sec
        elif self._duration_sec is not None:
            window = max(0.0, self._duration_sec - base)
        else:
            window = None

        # Cap by *lowering the sampling fps* so frames stay uniform across the
        # whole window, instead of `-frames:v N` which truncates to the first N.
        effective_fps = float(fps)
        if max_frames is not None and window and window > 0:
            effective_fps = min(effective_fps, max_frames / window)
            if effective_fps <= 0:
                effective_fps = float(fps)

        print(
            f"[ffmpeg] extracting  fps={effective_fps:.4f} (requested {fps})  "
            f"start={start_time}  duration={duration_sec}"
            + (f"  max_frames={max_frames}" if max_frames else ""),
            file=sys.stderr,
        )
        output_pattern = os.path.join(cache_dir, "frame_%09d.png")
        cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y"]
        if start_time is not None and start_time > 0:
            cmd += ["-ss", str(start_time)]
        if duration_sec is not None and duration_sec > 0:
            cmd += ["-t", str(duration_sec)]
        cmd += [*self._reconnect_flags(), "-i", self.video_path]
        vf_parts = [f"fps={effective_fps}"]
        if self.max_frame_dimension is not None:
            vf_parts.append(
                _scale_even_filter(self.max_frame_dimension)
            )
        cmd += ["-vf", ",".join(vf_parts)]
        cmd.append(output_pattern)
        subprocess.run(cmd, check=True, capture_output=True, text=True)

        paths = sorted(glob.glob(os.path.join(cache_dir, "frame_*.png")))
        return [base + i / effective_fps for i in range(len(paths))]

    # ------------------------------------------------------------------
    # Public frame API
    # ------------------------------------------------------------------

    def fetch_frames(
        self,
        fps: int = FPS,
        start_time_sec: float = 0.0,
        end_time_sec: Optional[float] = None,
        max_frames: Optional[int] = None,
    ) -> List[Frame]:
        if fps <= 0:
            raise ValueError("fps must be > 0")
        duration_sec = (end_time_sec - start_time_sec) if end_time_sec is not None else None
        # Pass max_frames into the cache layer so extraction is capped at the source,
        # preventing OOM on long videos (e.g. 2400s @ 1fps with no limit = ~2400 frames).
        # Returns (path, true_timestamp) so capped frames keep accurate times.
        frame_items = self._ensure_frames_cache(fps, start_time_sec, duration_sec, max_frames)
        if not frame_items:
            return []
        frames = [
            Frame(
                frame_file_path=p,
                timestamp=timestamp,
                video_id=self.video_id,
                id=f"{self.video_id}_fps{fps}_frame_{idx}",
            )
            for idx, (p, timestamp) in enumerate(frame_items)
        ]
        return frames

    def get_overview_frames(self) -> List[Frame]:
        return self.fetch_frames(fps=self.OVERVIEW_FPS, max_frames=self.OVERVIEW_MAX_FRAMES)

    # ------------------------------------------------------------------
    # Public clip API
    # ------------------------------------------------------------------

    def fetch_clip(
        self,
        start_time: float,
        end_time: float,
        fps: int = 5,
        crf: Optional[int] = None,
        max_size_mb: Optional[int] = None,
    ) -> Clip:
        print("[fetch-clip] starting fetch-clip", file=sys.stderr)
        duration_sec = min(end_time - start_time, self.MAX_CLIP_DURATION_SEC)
        clip = self._fetch_clip(start_time, duration_sec, fps=fps, crf=crf)
        size_mb = os.path.getsize(clip.clip_file_path) / 1024 / 1024
        print(f"[clip] size={size_mb:.2f}MB", file=sys.stderr)
        if max_size_mb is not None and size_mb > max_size_mb:
            print(f"[clip] too large ({size_mb:.2f}MB > {max_size_mb}MB), re-encoding at lower quality", file=sys.stderr)
            self.max_frame_dimension = 480
            clip = self._fetch_clip(start_time, duration_sec, fps=1, crf=28)
            size_mb = os.path.getsize(clip.clip_file_path) / 1024 / 1024
            if size_mb > max_size_mb:
                raise ValueError(f"Clip still too large after re-encode: {size_mb:.2f}MB > {max_size_mb}MB")
        return clip

    def _fetch_clip(self, start_time: float, duration_sec: float, fps: int = 5, crf: Optional[int] = None) -> Clip:
        _start_time = time.time()
        clip_path = os.path.join(self.frame_dir, f"clip_{start_time}_{duration_sec}secs.mp4")
        cmd = [
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
            "-ss", str(start_time), "-t", str(duration_sec),
            *self._reconnect_flags(),
            "-i", self.video_path,
        ]
        if crf is not None:
            cmd += ["-c:v", "libx264", "-crf", str(crf), "-preset", "fast"]
        vf_parts = []
        if self.max_frame_dimension is not None:
            vf_parts.append(
                _scale_even_filter(self.max_frame_dimension)
            )
        if fps is not None:
            vf_parts.append(f"fps={fps}")
        if vf_parts:
            cmd += ["-vf", ",".join(vf_parts)]
        cmd.append(clip_path)
        subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(f"[fetch-clip-ffmpeg] Total time taken: {time.time() - _start_time} seconds", file=sys.stderr)
        return Clip(
            clip_file_path=clip_path,
            start_time=start_time,
            end_time=start_time + duration_sec,
            video_id=self.video_id,
            id=f"{self.video_id}_clip_{start_time}_{duration_sec}secs",
        )


# ---------------------------------------------------------------------------
# CLI sub-commands
# ---------------------------------------------------------------------------

def _upload_frames(frames: List[Frame], s3: "S3Client", expires_in: int) -> None:
    """Upload frame PNGs to S3 in parallel and set frame.frame_url on each.

    Frame uploads dominate overview latency (dozens of small PNGs), so they run
    on a thread pool. The boto3 client is thread-safe for concurrent requests.
    """
    if not frames:
        return

    def _one(frame: Frame) -> None:
        key = f"{frame.video_id}/frames/{frame.id}.png"
        _, url = s3.upload_and_presign(frame.frame_file_path, key, "image/png", expires_in)
        frame.frame_url = url

    workers = min(UPLOAD_CONCURRENCY, len(frames))
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        # map() re-raises the first exception when the results are consumed.
        list(pool.map(_one, frames))
    print(f"[s3] uploaded {len(frames)} frames ({workers} workers)", file=sys.stderr)


def _upload_clip(clip: Clip, s3: "S3Client", expires_in: int) -> None:
    """Upload a clip MP4 to S3 and set clip.clip_url to the presigned URL."""
    key = f"{clip.video_id}/clips/{clip.id}.mp4"
    _, url = s3.upload_and_presign(clip.clip_file_path, key, "video/mp4", expires_in)
    clip.clip_url = url
    print(f"[s3] uploaded clip {clip.id} → {url}", file=sys.stderr)


def _run_json(cmd: List[str], timeout: int) -> dict[str, Any]:
    result = subprocess.run(cmd, check=True, capture_output=True, text=True, timeout=timeout)
    return json.loads(result.stdout)


def _ffprobe_media_info(path: str) -> dict[str, Any]:
    cmd = [
        "ffprobe", "-v", "error",
        "-show_entries", "format=duration:stream=width,height",
        "-of", "json",
        path,
    ]
    try:
        info = _run_json(cmd, timeout=60)
    except Exception:
        return {}
    duration: Optional[float] = None
    try:
        duration = float((info.get("format") or {}).get("duration"))
    except (TypeError, ValueError):
        pass
    width = None
    height = None
    for stream in info.get("streams") or []:
        if stream.get("width") and stream.get("height"):
            width = int(stream["width"])
            height = int(stream["height"])
            break
    return {"duration": duration, "width": width, "height": height}


def _normalize_faststart(path: str) -> bool:
    """Best-effort stream-copy remux to put moov at the front."""
    tmp = path + ".fix.mp4"
    cmd = [
        "ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error",
        "-y", "-i", path, "-c", "copy", "-movflags", "+faststart", tmp,
    ]
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True, timeout=YOUTUBE_DOWNLOAD_TIMEOUT_SECONDS)
        os.replace(tmp, path)
        return True
    except Exception as exc:  # noqa: BLE001
        try:
            if os.path.exists(tmp):
                os.remove(tmp)
        except OSError:
            pass
        print(f"[youtube] faststart normalization skipped: {exc}", file=sys.stderr)
        return False


def _canonicalize_youtube_file(video_id: str, work_dir: str) -> str:
    canonical = os.path.join(work_dir, f"{video_id}.mp4")
    candidates = [
        p
        for p in glob.glob(os.path.join(work_dir, f"{video_id}.*"))
        if os.path.isfile(p) and not p.endswith((".json", ".part", ".ytdl"))
    ]
    if not candidates:
        raise FileNotFoundError(f"yt-dlp produced no media file for {video_id}")
    if canonical in candidates:
        return canonical

    source = max(candidates, key=lambda p: os.path.getsize(p))
    ext = os.path.splitext(source)[1].lower()
    if ext in (".mp4", ".m4v", ".mov"):
        os.replace(source, canonical)
        return canonical

    cmd = [
        "ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error",
        "-y", "-i", source, "-c", "copy", "-movflags", "+faststart", canonical,
    ]
    subprocess.run(cmd, check=True, capture_output=True, text=True, timeout=YOUTUBE_DOWNLOAD_TIMEOUT_SECONDS)
    try:
        os.remove(source)
    except OSError:
        pass
    return canonical


def _youtube_format_selector(max_height: int) -> str:
    """Build the yt-dlp -f selector, optionally capping height.

    Downstream never uses more than 768px (tiles, description frames, provider
    quality), so a 720p proxy downloads 3-10x less while losing nothing. Prefers
    H.264 (avc1) over AV1/VP9 (YouTube's AV1 is SABR-gated and 403s mid-download,
    and H.264 decodes faster downstream). The final "/b" stays unfiltered so odd
    videos with no height-capped format still import. max_height <= 0 = uncapped.
    """
    h = f"[height<={max_height}]" if max_height > 0 else ""
    return (
        f"bv*[vcodec^=avc1]{h}[ext=mp4]+ba[ext=m4a]"
        f"/bv*{h}[ext=mp4]+ba[ext=m4a]"
        f"/b{h}[ext=mp4]"
        f"/bv*{h}+ba"
        "/b"
    )


def _probe_youtube_info(url: str) -> dict[str, Any]:
    """Fetch yt-dlp metadata (no download) and enforce the duration limit."""
    if not _is_youtube_url(url):
        raise ValueError("url must be a YouTube URL")
    if shutil.which("yt-dlp") is None:
        raise RuntimeError("yt-dlp is not installed in the sandbox template")
    info = _run_json(["yt-dlp", "--dump-single-json", "--no-playlist", url], timeout=120)
    duration = info.get("duration")
    if duration is not None and float(duration) > YOUTUBE_MAX_DURATION_SECONDS:
        raise ValueError(
            f"YouTube video duration {duration}s exceeds limit "
            f"{YOUTUBE_MAX_DURATION_SECONDS}s"
        )
    return info


def _download_youtube(video_id: str, url: str) -> tuple[str, dict[str, Any]]:
    info = _probe_youtube_info(url)

    work_dir = os.path.join(VIDEO_FOLDER, video_id)
    os.makedirs(work_dir, exist_ok=True)
    for existing in glob.glob(os.path.join(work_dir, f"{video_id}.*")):
        if os.path.isfile(existing):
            try:
                os.remove(existing)
            except OSError:
                pass

    fmt = _youtube_format_selector(YOUTUBE_MAX_HEIGHT)
    cmd = [
        "yt-dlp",
        "--no-playlist",
        "--restrict-filenames",
        # Prefer H.264 (avc1) over AV1/VP9: the AV1 formats YouTube serves are
        # SABR-gated and 403 mid-download, and H.264 also decodes faster downstream.
        "-f", fmt,
        # Survive transient 403s / format drops within this one invocation instead
        # of relying on the job-level retry (which re-boots an entire sandbox).
        "--retries", "10",
        "--fragment-retries", "10",
        "--extractor-retries", "5",
        "--retry-sleep", "http:exp=1:30",
        "--merge-output-format", "mp4",
        "--postprocessor-args", "Merger:-movflags +faststart",
        "--paths", work_dir,
        "--output", f"{video_id}.%(ext)s",
        url,
    ]
    print(f"[youtube] downloading {url}", file=sys.stderr)
    subprocess.run(
        cmd,
        check=True,
        text=True,
        stdout=sys.stderr,
        stderr=sys.stderr,
        timeout=YOUTUBE_DOWNLOAD_TIMEOUT_SECONDS,
    )

    local_path = _canonicalize_youtube_file(video_id, work_dir)
    size = os.path.getsize(local_path)
    if size > YOUTUBE_MAX_SIZE_BYTES:
        raise ValueError(
            f"YouTube download size {size} bytes exceeds limit "
            f"{YOUTUBE_MAX_SIZE_BYTES} bytes"
        )

    fragmented, faststart = _probe_local_mp4_layout(local_path)
    if fragmented or not faststart:
        reason = "fragmented" if fragmented else "moov-at-end"
        print(f"[youtube] {reason}; attempting faststart normalization", file=sys.stderr)
        _normalize_faststart(local_path)

    return local_path, info


def cmd_probe_youtube(args: argparse.Namespace) -> None:
    """Metadata-only probe (no download): title/duration/dimensions, in ~seconds.

    Lets the host seed a session with title + duration immediately while the full
    download proceeds in the background (see docs/session-startup-latency-plan.md).
    """
    start = time.time()
    info = _probe_youtube_info(args.url)
    print(f"[youtube] probe-youtube time: {time.time() - start} seconds", file=sys.stderr)
    emit(data=ProbeYoutubeResult(
        video_id=args.video_id,
        title=info.get("title"),
        duration=info.get("duration"),
        width=info.get("width"),
        height=info.get("height"),
        webpage_url=info.get("webpage_url") or args.url,
        extractor=info.get("extractor"),
    ))


def cmd_upload_source(args: argparse.Namespace) -> None:
    """Upload the already-downloaded local source MP4 to S3.

    Split out of prepare-youtube so the (slow) S3 upload runs concurrently with
    description + tiling instead of gating them. Resolves the local file the same
    way the tiling path does.
    """
    start = time.time()
    s3 = _make_s3_client()
    if s3 is None:
        emit(error=ResultError(code="S3NotConfigured", message="S3 is required for upload-source"))
        return
    local_path = _download_source_local(s3, args.video_id)  # local cache hit (no re-download)
    r2_key = f"{S3_VIDEO_BASE_KEY}/{args.video_id}.mp4"
    s3_uri = s3.upload_file(local_path, r2_key, "video/mp4")
    print(f"[s3] uploaded source {s3_uri} in {time.time() - start:.1f}s", file=sys.stderr)
    emit(data=UploadSourceResult(
        video_id=args.video_id,
        r2_key=r2_key,
        s3_uri=s3_uri,
        size_bytes=os.path.getsize(local_path),
    ))


def cmd_prepare_youtube(args: argparse.Namespace) -> None:
    start_download_time = time.time()
    print(f"[youtube] preparing {args.url}", file=sys.stderr)
    local_path, info = _download_youtube(args.video_id, args.url)
    print(f"[youtube] download time: {time.time() - start_download_time} seconds", file=sys.stderr)
    start_probe_time = time.time()
    probe = _ffprobe_media_info(local_path)
    print(f"[youtube] probe time: {time.time() - start_probe_time} seconds", file=sys.stderr)
    r2_key: Optional[str] = None
    s3_uri: Optional[str] = None
    if args.upload_s3:
        start_s3_upload_time = time.time()
        s3 = S3Client()
        r2_key = f"{S3_VIDEO_BASE_KEY}/{args.video_id}.mp4"
        s3_uri = s3.upload_file(local_path, r2_key, "video/mp4")
        print(f"[s3] uploaded source {s3_uri}", file=sys.stderr)
        print(f"[s3] upload time: {time.time() - start_s3_upload_time} seconds", file=sys.stderr)
    
    duration = probe.get("duration")
    if duration is None:
        duration = info.get("duration")
    width = probe.get("width") or info.get("width")
    height = probe.get("height") or info.get("height")
    print(f"[youtube] Total time taken for prepare-youtube: {time.time() - start_download_time} seconds", file=sys.stderr)
    emit(data=PrepareYoutubeResult(
        video_id=args.video_id,
        source_file_path=local_path,
        r2_key=r2_key,
        s3_uri=s3_uri,
        size_bytes=os.path.getsize(local_path),
        duration=duration,
        width=width,
        height=height,
        download_info={
            "extractor": info.get("extractor"),
            "webpage_url": info.get("webpage_url") or args.url,
            "title": info.get("title"),
            "format_id": info.get("format_id") or info.get("format"),
        },
    ))


def cmd_extract_frames(args: argparse.Namespace) -> None:
    start_time = time.time()
    tools = VideoFrameTools(args.video_id, max_frame_dimension=args.max_dim)
    frames = tools.fetch_frames(
        fps=args.fps,
        start_time_sec=args.start,
        end_time_sec=args.end,
        max_frames=args.max_frames,
    )
    print(f"[extract-frames] Total time taken: {time.time() - start_time} seconds", file=sys.stderr)
    if args.upload_s3:
        start_s3_upload_time = time.time()
        s3 = S3Client()
        _upload_frames(frames, s3, args.s3_presigned_expires)
        print(f"[extract-frames] [s3] upload time: {time.time() - start_s3_upload_time} seconds", file=sys.stderr)
    emit(data=FramesResult(
        frames=[FrameOut(**f.to_dict()) for f in frames],
        duration=tools._video_duration_seconds(),
    ))


def cmd_fetch_clip(args: argparse.Namespace) -> None:
    start_time = time.time()
    tools = VideoFrameTools(args.video_id, max_frame_dimension=args.max_dim)
    clip = tools.fetch_clip(
        start_time=args.start,
        end_time=args.end,
        fps=args.fps,
        crf=args.crf,
        max_size_mb=args.max_size_mb,
    )
    print(f"[fetch-clip] Total time taken: {time.time() - start_time} seconds", file=sys.stderr)
    if args.upload_s3:
        start_s3_upload_time = time.time()
        s3 = S3Client()
        _upload_clip(clip, s3, args.s3_presigned_expires)
        print(f"[fetch-clip] [s3] upload time: {time.time() - start_s3_upload_time} seconds", file=sys.stderr)
    emit(data=ClipResult(clip=ClipOut(**clip.to_dict())))


# ---------------------------------------------------------------------------
# Tile transcoding (ingestion-time) + concat assembly (fetch_clip fast path)
# ---------------------------------------------------------------------------

def _manifest_key(video_id: str) -> str:
    return f"{video_id}/tiles/manifest.json"


def _load_manifest(s3: "S3Client", video_id: str) -> Optional[dict]:
    return s3.read_json(_manifest_key(video_id))


def _download_source_local(s3: "S3Client", video_id: str) -> str:
    """Resolve *video_id* to a LOCAL file path, downloading from S3 if needed.

    Tiling reads the whole video, so we always want it on local disk (the N
    parallel range decodes then hit disk, not HTTP). One sequential full-file
    download is far more network-efficient than many windowed range reads.
    """
    search_dirs = [VIDEO_FOLDER, os.path.join(VIDEO_FOLDER, video_id)]
    for directory in search_dirs:
        if not os.path.isdir(directory):
            continue
        for ext in VIDEO_EXTENSIONS:
            candidate = os.path.join(directory, f"{video_id}{ext}")
            if os.path.isfile(candidate):
                print(f"[tiles] source found locally: {candidate}", file=sys.stderr)
                return candidate
    key, _size = _probe_source_key(s3, video_id)
    ext = os.path.splitext(key)[1].lower()
    local_path = os.path.join(VIDEO_FOLDER, video_id, f"{video_id}{ext}")
    s3.download_file(key, local_path)
    return local_path


def _parse_seglist(seglist_path: str, out_dir: str, offset: float) -> List[Tuple[str, float, float]]:
    """Parse the ffmpeg segment-muxer CSV rows: ``filename,start,end``.

    Times are relative to the worker's ``-ss`` range start, so we add *offset* to
    recover true positions in the source video. Returns rows in file (time) order.
    """
    rows: List[Tuple[str, float, float]] = []
    with open(seglist_path, encoding="utf-8") as fh:
        for line in fh:
            parts = line.strip().split(",")
            if len(parts) < 3:
                continue
            try:
                rows.append((os.path.join(out_dir, parts[0]), float(parts[1]) + offset, float(parts[2]) + offset))
            except ValueError:
                continue
    return rows


def _upload_tiles_parallel(tiles: List[dict], s3: "S3Client") -> None:
    """Upload tile MP4s to S3 in parallel and stamp each dict's ``size_bytes``."""
    def _one(t: dict) -> None:
        s3.upload_file(t["local_path"], t["key"], "video/mp4")
        t["size_bytes"] = os.path.getsize(t["local_path"])

    workers = min(UPLOAD_CONCURRENCY, len(tiles)) or 1
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        list(pool.map(_one, tiles))
    print(f"[tiles] uploaded {len(tiles)} tiles ({workers} workers)", file=sys.stderr)


def cmd_transcode_tiles(args: argparse.Namespace) -> None:
    """Transcode the whole video once into fixed, GOP-aligned target-quality tiles.

    The video is split into worker time-ranges (each a whole number of tiles so
    cuts stay aligned) and transcoded in parallel; each range emits a segment
    list with exact per-tile boundaries. Tiles + a manifest are uploaded to S3.
    Idempotent: a complete matching manifest short-circuits the work.
    """
    overall_start = time.time()
    s3 = _make_s3_client()
    if s3 is None:
        emit(error=ResultError(code="S3NotConfigured", message="S3 is required for tiling"))
        return

    existing = _load_manifest(s3, args.video_id)
    if (
        existing
        and existing.get("fps") == args.fps
        and existing.get("max_dim") == args.max_dim
        and existing.get("tile_seconds") == args.tile_seconds
        and existing.get("tiles")
    ):
        print(f"[tiles] manifest exists ({len(existing['tiles'])} tiles); skipping", file=sys.stderr)
        emit(data=TilesResult(**existing))
        return

    src = _download_source_local(s3, args.video_id)
    info = _ffprobe_media_info(src)
    duration = info.get("duration")
    if not duration or duration <= 0:
        emit(error=ResultError(code="ProbeFailed", message=f"could not determine duration for {args.video_id}"))
        return

    # Each worker handles a whole number of tiles so range boundaries land on
    # tile_seconds multiples (only the final tile is short).
    tiles_per_worker = max(1, math.ceil(duration / args.tile_seconds / max(1, args.workers)))
    chunk_dur = tiles_per_worker * args.tile_seconds
    ranges: List[Tuple[int, float, float]] = []
    rstart = 0.0
    while rstart < duration:
        ranges.append((len(ranges), rstart, min(chunk_dur, duration - rstart)))
        rstart += chunk_dur
    tiles_root = os.path.join(VIDEO_FOLDER, args.video_id, "tiles")

    def _tile_range(item: Tuple[int, float, float]) -> List[Tuple[str, float, float]]:
        widx, r_start, r_len = item
        out_dir = os.path.join(tiles_root, f"w{widx}")
        os.makedirs(out_dir, exist_ok=True)
        seglist = os.path.join(out_dir, "list.csv")
        cmd = [
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
            "-ss", str(r_start), "-t", str(r_len), "-i", src,
            "-vf", f"{_scale_even_filter(args.max_dim)},fps={args.fps}",
            "-c:v", "libx264", "-preset", "veryfast",
            # Force a keyframe at every tile boundary so tiles are independently
            # decodable AND concat-able with `-c copy` later.
            "-force_key_frames", f"expr:gte(t,n_forced*{args.tile_seconds})",
            "-f", "segment", "-segment_time", str(args.tile_seconds),
            "-reset_timestamps", "1",
            "-segment_list", seglist, "-segment_list_type", "csv",
            os.path.join(out_dir, "seg_%05d.mp4"),
        ]
        subprocess.run(cmd, check=True, capture_output=True, text=True)
        return _parse_seglist(seglist, out_dir, offset=r_start)

    workers = min(max(1, args.workers), len(ranges))
    print(
        f"[tiles] transcoding {args.video_id} dur={duration:.1f}s into {args.tile_seconds}s tiles "
        f"(fps={args.fps} dim={args.max_dim}) across {workers} workers",
        file=sys.stderr,
    )
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        per_worker = list(pool.map(_tile_range, ranges))

    # Flatten in range (time) order, assign global indices.
    tiles: List[dict] = []
    for chunk in per_worker:
        for local_path, start, end in chunk:
            idx = len(tiles)
            tiles.append({
                "index": idx,
                "start": start,
                "end": end,
                "key": f"{args.video_id}/tiles/tile_{idx:05d}.mp4",
                "local_path": local_path,
            })
    if not tiles:
        emit(error=ResultError(code="NoTiles", message="tiling produced no segments"))
        return

    _upload_tiles_parallel(tiles, s3)

    manifest = {
        "video_id": args.video_id,
        "fps": args.fps,
        "max_dim": args.max_dim,
        "tile_seconds": args.tile_seconds,
        "duration": duration,
        "tiles": [
            {"index": t["index"], "start": t["start"], "end": t["end"], "key": t["key"], "size_bytes": t.get("size_bytes")}
            for t in tiles
        ],
    }
    s3.put_json(_manifest_key(args.video_id), manifest)
    print(
        f"[tiles] wrote manifest {_manifest_key(args.video_id)} "
        f"({len(tiles)} tiles, {time.time() - overall_start:.1f}s)",
        file=sys.stderr,
    )
    emit(data=TilesResult(**manifest))


def cmd_concat_tiles(args: argparse.Namespace) -> None:
    """Assemble a clip by stream-copy concatenating pre-transcoded tiles.

    Reads only the tile bytes (no re-decode): builds an ffmpeg concat list of
    presigned tile URLs and remuxes with `-c copy`. This is the fetch_clip fast
    path. ``--start``/``--end`` carry the true (tile-boundary) span so the caller
    can map citations.
    """
    s3 = _make_s3_client()
    if s3 is None:
        emit(error=ResultError(code="S3NotConfigured", message="S3 is required for concat-tiles"))
        return
    keys = [k for k in args.keys.split(",") if k]
    if not keys:
        emit(error=ResultError(code="NoKeys", message="no tile keys provided"))
        return

    work_dir = os.path.join(VIDEO_FOLDER, args.video_id, "assembled")
    os.makedirs(work_dir, exist_ok=True)
    list_path = os.path.join(work_dir, "concat.txt")
    with open(list_path, "w", encoding="utf-8") as fh:
        for k in keys:
            url = s3.get_presigned_url(k, expires_in=SOURCE_URL_TTL)
            # concat demuxer: wrap in single quotes, escaping any embedded quote.
            fh.write("file '%s'\n" % url.replace("'", "'\\''"))

    out_id = f"{args.video_id}_tiles_{os.path.splitext(os.path.basename(keys[0]))[0]}_{len(keys)}"
    out_path = os.path.join(work_dir, f"{out_id}.mp4")
    cmd = [
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
        "-protocol_whitelist", "file,http,https,tcp,tls",
        "-f", "concat", "-safe", "0", "-i", list_path,
        "-c", "copy", out_path,
    ]
    subprocess.run(cmd, check=True, capture_output=True, text=True)
    clip = Clip(
        clip_file_path=out_path,
        start_time=args.start,
        end_time=args.end,
        video_id=args.video_id,
        id=out_id,
    )
    if args.upload_s3:
        _upload_clip(clip, s3, args.s3_presigned_expires)
    emit(data=ClipResult(clip=ClipOut(**clip.to_dict())))


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="main.py",
        description="Video analysis CLI for the E2B sandbox.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # ── extract-frames ───────────────────────────────────────────────────────
    p_ef = sub.add_parser("extract-frames", help="Extract frames from a video file and write them to disk")
    p_ef.add_argument("--video-id", required=True, help="Video ID; resolved from VIDEO_FOLDER or downloaded from S3")
    p_ef.add_argument("--fps", type=int, default=1, help="Frames per second to extract (default: 1)")
    p_ef.add_argument("--start", type=float, default=0.0, help="Start time in seconds (default: 0)")
    p_ef.add_argument("--end", type=float, default=None, help="End time in seconds (default: full video)")
    p_ef.add_argument("--max-frames", type=int, default=None, help="Maximum number of frames to return")
    p_ef.add_argument("--max-dim", type=int, default=None, help="Resize frames so longest edge ≤ this value")
    p_ef.add_argument("--upload-s3", action="store_true", help="Upload extracted frames to S3 and include presigned URLs in output")
    p_ef.add_argument("--s3-presigned-expires", type=int, default=3600, metavar="SECS", help="Presigned URL TTL in seconds (default: 3600)")
    p_ef.set_defaults(func=cmd_extract_frames)

    # ── fetch-clip ────────────────────────────────────────────────────────────
    p_fc = sub.add_parser("fetch-clip", help="Extract a video clip between two timestamps")
    p_fc.add_argument("--video-id", required=True, help="Video ID; resolved from VIDEO_FOLDER or downloaded from S3")
    p_fc.add_argument("--start", type=float, required=True, help="Start time in seconds")
    p_fc.add_argument("--end", type=float, required=True, help="End time in seconds")
    p_fc.add_argument("--fps", type=int, default=5, help="Output clip fps (default: 5)")
    p_fc.add_argument("--crf", type=int, default=None, help="H.264 CRF quality (lower = better; e.g. 23)")
    p_fc.add_argument("--max-dim", type=int, default=None, help="Resize frames so longest edge ≤ this value")
    p_fc.add_argument("--max-size-mb", type=int, default=None, help="Auto-reduce quality if clip exceeds this size in MB")
    p_fc.add_argument("--upload-s3", action="store_true", help="Upload the clip to S3 and include a presigned URL in output")
    p_fc.add_argument("--s3-presigned-expires", type=int, default=3600, metavar="SECS", help="Presigned URL TTL in seconds (default: 3600)")
    p_fc.set_defaults(func=cmd_fetch_clip)

    # ── prepare-youtube ──────────────────────────────────────────────────────
    p_yt = sub.add_parser("prepare-youtube", help="Download a YouTube URL into the sandbox video cache")
    p_yt.add_argument("--video-id", required=True, help="Video ID to materialize into VIDEO_FOLDER")
    p_yt.add_argument("--url", required=True, help="YouTube URL to download")
    p_yt.add_argument("--upload-s3", action="store_true", help="Upload the prepared source MP4 to S3/R2")
    p_yt.set_defaults(func=cmd_prepare_youtube)

    # ── probe-youtube ─────────────────────────────────────────────────────────
    p_py = sub.add_parser("probe-youtube", help="Fetch YouTube metadata only (title/duration), no download")
    p_py.add_argument("--video-id", required=True, help="Video ID the metadata belongs to")
    p_py.add_argument("--url", required=True, help="YouTube URL to probe")
    p_py.set_defaults(func=cmd_probe_youtube)

    # ── upload-source ─────────────────────────────────────────────────────────
    p_us = sub.add_parser("upload-source", help="Upload the already-downloaded local source MP4 to S3")
    p_us.add_argument("--video-id", required=True, help="Video ID whose local source to upload")
    p_us.set_defaults(func=cmd_upload_source)

    # ── transcode-tiles ──────────────────────────────────────────────────────
    p_tt = sub.add_parser("transcode-tiles", help="Transcode the whole video into fixed target-quality tiles + manifest (ingestion)")
    p_tt.add_argument("--video-id", required=True, help="Video ID; resolved from VIDEO_FOLDER or downloaded from S3")
    p_tt.add_argument("--tile-seconds", type=int, default=TILE_SECONDS, help=f"Tile length / keyframe interval (default: {TILE_SECONDS})")
    p_tt.add_argument("--fps", type=int, default=TILE_FPS, help=f"Tile fps; must match the provider quality (default: {TILE_FPS})")
    p_tt.add_argument("--max-dim", type=int, default=TILE_MAX_DIM, help=f"Tile longest-edge dimension (default: {TILE_MAX_DIM})")
    p_tt.add_argument("--workers", type=int, default=TILE_WORKERS, help=f"Parallel decode ranges; set to box vCPUs (default: {TILE_WORKERS})")
    p_tt.add_argument("--upload-s3", action="store_true", help="Upload tiles + manifest to S3")
    p_tt.set_defaults(func=cmd_transcode_tiles)

    # ── concat-tiles ─────────────────────────────────────────────────────────
    # TODO(fetch-clip fast path): wire SandboxVideoFrameTools.fetch_clip to read
    # the tile manifest, select the covering tiles, and call this command instead
    # of on-demand `fetch-clip`. The command is implemented and ready; only the
    # caller side is not wired yet.
    p_ct = sub.add_parser("concat-tiles", help="Stream-copy concat pre-transcoded tiles into one clip (fetch_clip fast path)")
    p_ct.add_argument("--video-id", required=True, help="Video ID the tiles belong to")
    p_ct.add_argument("--keys", required=True, help="Comma-separated S3 tile keys in time order")
    p_ct.add_argument("--start", type=float, default=0.0, help="True start time of the first tile (for citations)")
    p_ct.add_argument("--end", type=float, default=0.0, help="True end time of the last tile")
    p_ct.add_argument("--upload-s3", action="store_true", help="Upload the assembled clip to S3 and include a presigned URL")
    p_ct.add_argument("--s3-presigned-expires", type=int, default=3600, metavar="SECS", help="Presigned URL TTL in seconds (default: 3600)")
    p_ct.set_defaults(func=cmd_concat_tiles)

    return parser


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING, stream=sys.stderr)
    parser = _build_parser()
    args = parser.parse_args()
    try:
        args.func(args)
    except Exception as exc:  # noqa: BLE001 - top-level guard: all failures become an envelope
        logger.exception("command failed")
        emit(error=ResultError(code=type(exc).__name__, message=str(exc)))

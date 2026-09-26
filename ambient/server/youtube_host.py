"""Host-side YouTube ingestion (no e2b sandbox).

Used when ``settings.sandbox_backend != "e2b"``: download + remux a YouTube
source straight into ``settings.video_folder`` as ``<video_id>.mp4`` so the
inprocess (range-proxy) tools, the ``/files/{id}/content`` endpoint, and host
description generation all read it locally — no S3, no sandbox.

Mirrors the sandbox's ``prepare-youtube`` logic (format selector, faststart)
but shells out to the host ``yt-dlp`` / ``ffmpeg`` CLIs.
"""

from __future__ import annotations

import glob
import json
import os
import shutil
import subprocess
from typing import Any

from ambient.config import settings
from ambient.server.video_store import is_youtube_url


def _format_selector(max_height: int) -> str:
    """yt-dlp ``-f`` selector, capped to ``max_height`` and preferring H.264.

    Matches the sandbox: downstream never uses more than 768px, and YouTube's
    AV1 is SABR-gated (403s mid-download), so a height-capped avc1 proxy is both
    smaller and more reliable. The final unfiltered ``/b`` keeps odd videos
    importable. ``max_height <= 0`` = uncapped.
    """
    h = f"[height<={max_height}]" if max_height > 0 else ""
    return (
        f"bv*[vcodec^=avc1]{h}[ext=mp4]+ba[ext=m4a]"
        f"/bv*{h}[ext=mp4]+ba[ext=m4a]"
        f"/b{h}[ext=mp4]"
        f"/bv*{h}+ba"
        "/b"
    )


def probe_youtube(url: str) -> dict[str, Any]:
    """yt-dlp metadata only (no download); enforces the duration cap."""
    if not is_youtube_url(url):
        raise ValueError("url must be a YouTube URL")
    if shutil.which("yt-dlp") is None:
        raise RuntimeError(
            "yt-dlp is not installed on the host — install it "
            "(uv pip install yt-dlp) to import YouTube URLs without e2b"
        )
    out = subprocess.run(
        ["yt-dlp", "--dump-single-json", "--no-playlist", url],
        capture_output=True,
        check=True,
        text=True,
        timeout=120,
    )
    info = json.loads(out.stdout)
    duration = info.get("duration")
    if duration is not None and float(duration) > settings.youtube_max_duration_seconds:
        raise ValueError(
            f"YouTube video duration {duration}s exceeds limit "
            f"{settings.youtube_max_duration_seconds}s"
        )
    return info


def download_youtube(video_id: str, url: str) -> tuple[str, dict[str, Any]]:
    """Download + remux into ``settings.video_folder/<video_id>.mp4``.

    Returns ``(local_path, info)``. Raises on probe/download/size-cap failure so
    the ingest worker can record ``source_error`` and re-enqueue.
    """
    info = probe_youtube(url)

    video_folder = settings.video_folder
    os.makedirs(video_folder, exist_ok=True)
    work_dir = os.path.join(video_folder, f".{video_id}.ytwork")
    shutil.rmtree(work_dir, ignore_errors=True)
    os.makedirs(work_dir, exist_ok=True)

    try:
        cmd = [
            "yt-dlp",
            "--no-playlist",
            "--restrict-filenames",
            "-f",
            _format_selector(settings.youtube_max_height),
            # Survive transient 403s / format drops within this one invocation.
            "--retries",
            "10",
            "--fragment-retries",
            "10",
            "--extractor-retries",
            "5",
            "--retry-sleep",
            "http:exp=1:30",
            "--merge-output-format",
            "mp4",
            "--postprocessor-args",
            "Merger:-movflags +faststart",
            "--paths",
            work_dir,
            "--output",
            f"{video_id}.%(ext)s",
            url,
        ]
        subprocess.run(
            cmd,
            capture_output=True,
            check=True,
            text=True,
            timeout=settings.youtube_download_timeout_seconds,
        )

        candidates = [
            p
            for p in glob.glob(os.path.join(work_dir, f"{video_id}.*"))
            if os.path.isfile(p) and not p.endswith((".json", ".part", ".ytdl"))
        ]
        if not candidates:
            raise FileNotFoundError(f"yt-dlp produced no media file for {video_id}")

        source = max(candidates, key=os.path.getsize)
        final = os.path.join(video_folder, f"{video_id}.mp4")
        ext = os.path.splitext(source)[1].lower()
        if ext in (".mp4", ".m4v", ".mov"):
            os.replace(source, final)
        else:
            # Remux odd containers (webm/mkv) to a faststart MP4 without re-encoding.
            subprocess.run(
                [
                    "ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-y",
                    "-i", source, "-c", "copy", "-movflags", "+faststart", final,
                ],
                capture_output=True,
                check=True,
                text=True,
                timeout=settings.youtube_download_timeout_seconds,
            )
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)

    size = os.path.getsize(final)
    if size > settings.youtube_max_size_bytes:
        os.remove(final)
        raise ValueError(
            f"YouTube download size {size} bytes exceeds limit "
            f"{settings.youtube_max_size_bytes} bytes"
        )
    return final, info

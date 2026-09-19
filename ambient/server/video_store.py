"""Video ingestion: persist an uploaded video and hand back a `video_id`.

The agent's tools (`VideoFrameTools`) resolve a `video_id` by globbing
`settings.video_folder` for `<video_id>*` and reading the file locally with
ffmpeg. So an uploaded video must land in that folder. We also push a copy to
R2/S3 (best-effort) for durability and for any future remote sandbox.

`store_video()` returns the metadata both API surfaces echo back; the canonical
identifier is `video_id` (used verbatim as the Files API file id too).
"""
from __future__ import annotations

import hashlib
import logging
import mimetypes
import os
import subprocess
from urllib.parse import parse_qs, urlparse

from ambient.config import settings
from ambient.utils.s3 import get_s3_client

log = logging.getLogger(__name__)

VIDEO_FOLDER = settings.video_folder
_VIDEO_EXTS = {".mp4", ".mov", ".mkv", ".webm", ".avi", ".m4v", ".mpeg", ".mpg"}
# Containers where moov placement matters and `+faststart` is a valid stream-copy.
_FASTSTART_EXTS = {".mp4", ".mov", ".m4v"}
_YOUTUBE_HOSTS = {"youtube.com", "www.youtube.com", "m.youtube.com", "youtu.be"}


def _faststart_in_place(path: str) -> None:
    """Stream-copy remux so the mp4 `moov` atom sits at the FRONT (faststart).

    Uploads commonly ship moov-at-end, which forces any HTTP reader (the
    range-proxy) to pull most of the file before it can probe duration or seek —
    turning a 1-frame extract into a whole-file download. `-c copy` keeps it cheap
    (no re-encode). Best-effort: leaves the original untouched on any failure.
    """
    # Keep the source extension on the temp file so ffmpeg infers the output
    # container (a `.tmp` suffix -> "Unable to choose an output format").
    root, ext = os.path.splitext(path)
    tmp = f"{root}.faststart{ext}"
    try:
        r = subprocess.run(
            ["ffmpeg", "-nostdin", "-y", "-loglevel", "error", "-i", path,
             "-c", "copy", "-movflags", "+faststart", tmp],
            capture_output=True, timeout=300)
        if r.returncode == 0 and os.path.exists(tmp) and os.path.getsize(tmp) > 0:
            os.replace(tmp, path)
        else:
            log.warning("faststart remux skipped for %s: %s", path,
                        r.stderr.decode("utf-8", "replace")[:200])
    except Exception as exc:  # noqa: BLE001 - best-effort; original file still works
        log.warning("faststart remux failed for %s: %s", path, exc)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)


def _probe_duration_local(path: str) -> "float | None":
    """Duration (seconds) via a fast LOCAL ffprobe. None on failure. Probed at
    ingest and persisted so tools never probe duration over the network."""
    try:
        r = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "default=nw=1:nk=1", path],
            capture_output=True, timeout=30)
        if r.returncode == 0:
            return float(r.stdout.decode().strip())
    except Exception:  # noqa: BLE001
        pass
    return None


def content_video_id(data: bytes) -> str:
    """Content-addressed id for an uploaded video.

    Deriving the id from the bytes means identical content always maps to the
    same `video_id` — and therefore the same `files` row, cached description,
    R2 object, and local file — so re-uploads reuse the ingested description
    instead of regenerating it. The description depends only on the video
    content (frames + transcript), never the filename, so sharing is safe.
    """
    return f"vid_{hashlib.sha256(data).hexdigest()[:16]}"


def _youtube_source_key(url: str) -> str:
    """Stable key for a YouTube URL, used to content-address the id.

    Normalizes to the 11-char video id when we can recognize the URL shape
    (`youtu.be/<id>`, `watch?v=<id>`, `/shorts/<id>`, `/embed/<id>`) so that
    differing query params / hosts for the same video collapse to one id.
    Falls back to the lowercased URL when the shape is unfamiliar.
    """
    parsed = urlparse(url or "")
    host = (parsed.hostname or "").lower()
    path = parsed.path or ""
    if host == "youtu.be":
        vid = path.lstrip("/").split("/")[0]
        if vid:
            return vid
    if host in _YOUTUBE_HOSTS:
        qs = parse_qs(parsed.query or "")
        if qs.get("v"):
            return qs["v"][0]
        for prefix in ("/shorts/", "/embed/", "/v/"):
            if path.startswith(prefix):
                vid = path[len(prefix):].split("/")[0]
                if vid:
                    return vid
    return (url or "").strip().lower()


def _youtube_video_id(url: str) -> str:
    """Content-addressed id for a YouTube-backed video (keyed on the URL)."""
    key = _youtube_source_key(url)
    return f"vid_{hashlib.sha256(key.encode()).hexdigest()[:16]}"


def _pick_extension(filename: str | None, mime_type: str | None) -> str:
    ext = os.path.splitext(filename or "")[1].lower()
    if ext in _VIDEO_EXTS:
        return ext
    if mime_type:
        guessed = mimetypes.guess_extension(mime_type)
        if guessed:
            return guessed
    return ".mp4"


def store_video(data: bytes, filename: str | None, mime_type: str | None) -> dict:
    """Persist an uploaded video locally (for the tools) and to R2 (durability).

    Returns a metadata dict: video_id, filename, mime_type, size_bytes,
    local_path, r2_key (None if the R2 upload was skipped/failed), created_at.
    """
    video_id = content_video_id(data)
    ext = _pick_extension(filename, mime_type)
    os.makedirs(VIDEO_FOLDER, exist_ok=True)
    local_path = os.path.join(VIDEO_FOLDER, f"{video_id}{ext}")
    with open(local_path, "wb") as f:
        f.write(data)

    # Normalize moov -> front (faststart) so the range-proxy's HTTP seeks/probes are
    # cheap for any worker without the local cache, then probe duration once locally
    # (instant) and persist it so tools never probe over the network. Both best-effort.
    if ext in _FASTSTART_EXTS:
        _faststart_in_place(local_path)
    duration = _probe_duration_local(local_path)

    r2_key: str | None = None
    try:
        client = get_s3_client()
        if client.bucket:
            # Must match where the agent's media backend reads it from:
            # the e2b sandbox resolves `<s3_video_base_key>/<video_id>.mp4`
            # (ambient/sandboxes/e2b/templates/video-analysis-v1/main.py).
            r2_key = f"{settings.s3_video_base_key}/{video_id}{ext}"
            client.upload_file(
                local_path,
                r2_key,
                extra_args={"ContentType": mime_type or f"video/{ext.lstrip('.')}"},
            )
    except Exception as exc:  # best-effort — local cache is enough for local sandbox
        log.warning("R2 upload failed for %s: %s", video_id, exc)
        r2_key = None

    from ambient.server.store import _now

    return {
        "video_id": video_id,
        "filename": filename or f"{video_id}{ext}",
        "mime_type": mime_type or f"video/{ext.lstrip('.')}",
        "size_bytes": len(data),
        "local_path": local_path,
        "r2_key": r2_key,
        "duration": duration,
        "source_type": "upload",
        "source_status": "ready",
        "source_error": None,
        "source_attempts": 0,
        "created_at": _now(),
    }


def is_youtube_url(url: str) -> bool:
    parsed = urlparse(url or "")
    host = (parsed.hostname or "").lower()
    return parsed.scheme in {"http", "https"} and host in _YOUTUBE_HOSTS


def create_youtube_video(url: str, filename: str | None = None) -> dict:
    """Create metadata for a YouTube-backed file.

    The source is not downloaded on the API host. A background ingest sandbox
    materializes it, uploads the canonical MP4 to R2/S3, then writes these fields
    through with source_status="ready".
    """
    if not is_youtube_url(url):
        raise ValueError("url must be a YouTube URL")

    from ambient.server.store import _now

    now = _now()
    video_id = _youtube_video_id(url)
    return {
        "video_id": video_id,
        "filename": filename or f"{video_id}.mp4",
        "mime_type": "video/mp4",
        "size_bytes": None,
        "local_path": None,
        "r2_key": None,
        "source_type": "youtube",
        "source_url": url,
        "source_status": "pending",
        "source_error": None,
        "source_attempts": 0,
        "source_updated_at": now,
        "youtube": {
            "webpage_url": None,
            "title": None,
            "extractor": None,
            "duration": None,
            "width": None,
            "height": None,
            "format_id": None,
        },
        "description": None,
        "description_status": "pending",
        "description_error": None,
        "description_attempts": 0,
        "created_at": now,
    }

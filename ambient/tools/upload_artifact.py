"""upload_artifact: publish a file the agent produced to S3 for the user.

The agent often creates files while working — a rendered chart, an extracted
audio track, a processed clip, a report — via the bash tool. On the e2b backend
those live inside the ephemeral sandbox and are otherwise unreachable; on the
host they sit in the workspace. This tool uploads such a file to S3 and returns
its `s3://` path plus a presigned download URL the user can open.

Backend-agnostic (selected the usual way via `current_media_box`): an active e2b
box runs the in-box `upload-artifact` command; otherwise the host uploads the
file directly. Registered only when S3 is configured (`settings.s3_bucket`).
Objects are keyed under `artifacts/<video_id>/`.
"""
from __future__ import annotations

import asyncio
import mimetypes
import os
import re
import time
from typing import Dict, List, Optional, Tuple

from pydantic import BaseModel, Field

from ambient.config import settings
from ambient.tools.video_backend import current_media_box, validate_external_media_path
from ambient.utils.s3 import get_s3_client

_NAME_RE = re.compile(r"[^A-Za-z0-9._-]+")


class UploadArtifactTool(BaseModel):
    video_id: str = Field(description="The id of the session video (used only to group the uploaded file; supplied automatically).")
    path: str = Field(description="Absolute path to the file to upload (a file you produced, e.g. via the bash tool). On the e2b backend this is a path inside the sandbox; on the host, a path in the workspace.")
    name: Optional[str] = Field(default=None, description="Optional friendly name for the uploaded object (defaults to the file's own name).")


def _artifact_key(video_id: str, path: str, name: Optional[str]) -> str:
    base = os.path.basename(path)
    stem, ext = os.path.splitext(name or base)
    stem = _NAME_RE.sub("_", stem).strip("_") or "artifact"
    if not ext:
        ext = os.path.splitext(base)[1]
    vid = _NAME_RE.sub("_", video_id or "session").strip("_") or "session"
    return f"artifacts/{vid}/{stem}_{int(time.time())}{ext}"


def _content_type(path: str) -> str:
    return mimetypes.guess_type(path)[0] or "application/octet-stream"


def _upload_host(path: str, key: str, ttl: int) -> Tuple[str, str, int]:
    """Upload a validated host file. Returns (s3_uri, presigned_url, size_bytes)."""
    s3 = get_s3_client()
    size = os.path.getsize(path)
    s3_uri = s3.upload_file(path, key, extra_args={
        "ContentType": _content_type(path), "Metadata": {"source": "upload_artifact"}})
    url = s3.get_presigned_url(key, expires_in=ttl)
    return s3_uri, url, size


async def upload_artifact(video_id: str, path: str, name: str = None) -> Tuple[str, List[Dict]]:
    if not getattr(get_s3_client(), "bucket", None):
        return "Error: S3 is not configured, so upload_artifact is unavailable.", []

    box = current_media_box.get()
    try:
        resolved, is_box = validate_external_media_path(path)
    except ValueError as e:
        return f"Error uploading artifact: {e}", []

    key = _artifact_key(video_id, resolved, name)
    ttl = int(settings.artifact_url_ttl_seconds)
    cap_mb = int(settings.artifact_max_size_mb)

    if is_box and box is not None:
        try:
            data = await asyncio.to_thread(box.run, [
                "upload-artifact", "--path", resolved, "--key", key,
                "--content-type", _content_type(resolved),
                "--max-size-mb", str(cap_mb),
                "--s3-presigned-expires", str(ttl),
            ])
        except Exception as e:  # noqa: BLE001
            return f"Error uploading artifact {path!r} from sandbox: {e}", []
        s3_uri, url, size = (data or {}).get("s3_uri"), (data or {}).get("url"), (data or {}).get("size_bytes")
        if not url:
            return f"Error uploading artifact {path!r}: sandbox returned no download url", []
    else:
        # Host size cap up front (the box command enforces its own).
        try:
            size = os.path.getsize(resolved)
        except OSError as e:
            return f"Error uploading artifact {path!r}: {e}", []
        if size > cap_mb * 1024 * 1024:
            return (f"Error: artifact is {size / 1024 / 1024:.1f} MB, over the "
                    f"{cap_mb} MB limit.", [])
        try:
            s3_uri, url, size = await asyncio.to_thread(_upload_host, resolved, key, ttl)
        except Exception as e:  # noqa: BLE001
            return f"Error uploading artifact {path!r}: {e}", []

    mb = (size or 0) / 1024 / 1024
    hours = ttl // 3600
    text = (f"Uploaded {path} ({mb:.1f} MB) to {s3_uri}\n"
            f"Download URL (valid ~{hours}h): {url}")
    return text, []

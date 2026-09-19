"""read_image: hand a standalone image file to the agent for direct inspection.

Like grab_frames, but the source is an image on disk (a screenshot, a chart, a
frame the agent saved via the bash tool) rather than a video frame. Selects the
backend the same way the video tools do: an active e2b media box runs the in-box
`read-image` command (resize + upload -> presigned URL); otherwise the host reads,
downscales, and inlines the image as a base64 data URL. Either way the image comes
back as an `image_url` block the agent model reads directly.
"""
from __future__ import annotations

import asyncio
import base64
import io
from typing import Dict, List, Optional, Tuple

from pydantic import BaseModel, Field

from ambient.config import settings, get_provider_quality_settings
from ambient.tools.video_backend import current_media_box, validate_external_media_path


class ReadImageTool(BaseModel):
    image_path: str = Field(description="Absolute path to a local image file to read and inspect (e.g. a screenshot or frame you produced via the bash tool). For video frames use grab_frames instead.")
    note: Optional[str] = Field(default=None, description="Optional note on why you're reading this image / what to look for.")


def _agent_max_dim() -> int:
    q = get_provider_quality_settings(settings.agent_model)
    return int((q.max_dimentions if q else None) or settings.analysis_max_dim or 768)


def _read_host_image(path: str, max_dim: int) -> str:
    """Open, downscale to `max_dim` (longest edge, never upscale), return a JPEG
    base64 data URL."""
    from PIL import Image

    img = Image.open(path)
    img = img.convert("RGB")
    w, h = img.size
    longest = max(w, h)
    if longest > max_dim and longest > 0:
        scale = float(max_dim) / longest
        img = img.resize((max(1, round(w * scale)), max(1, round(h * scale))))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


async def read_image(image_path: str, note: str = None) -> Tuple[str, List[Dict]]:
    box = current_media_box.get()
    try:
        resolved, is_box = validate_external_media_path(image_path)
    except ValueError as e:
        return f"Error reading image: {e}", []

    max_dim = _agent_max_dim()

    if is_box and box is not None:
        try:
            data = await asyncio.to_thread(
                box.run, ["read-image", "--path", resolved, "--max-dim", str(max_dim), "--upload-s3"]
            )
        except Exception as e:  # noqa: BLE001
            return f"Error reading image {image_path!r} in sandbox: {e}", []
        url = (data or {}).get("image_url")
        if not url:
            return f"Error reading image {image_path!r}: sandbox returned no image url", []
    else:
        try:
            url = await asyncio.to_thread(_read_host_image, resolved, max_dim)
        except Exception as e:  # noqa: BLE001
            return f"Error reading image {image_path!r}: {e}", []

    prompt = f"Image at {image_path}." + (f" Note: {note}" if note else "") + " Carefully inspect it."
    return prompt, [{"type": "image_url", "image_url": {"url": url}}]

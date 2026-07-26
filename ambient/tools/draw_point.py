"""draw_point: visualize a single point / click coordinate on a video frame.

Mirror of ``draw_bounding_box``: the agent predicts a click location, this tool
draws a marker there on the real (full-res) frame and hands the annotated image
back so the agent can verify the point lands on the intended element.

Drawing reuses the sandbox ``annotate-frame`` path (the only place that has the
frame): the point is converted to a small square box centered on it, so no
sandbox/template change is needed. Alongside the image the tool returns the
click coordinate computed deterministically in code (center of the marker mapped
to the frame), so downstream consumers get an exact point instead of the model's
mental arithmetic.

Input convention: ``point=[x, y]`` on a 0-1000 grid (x horizontal, y vertical) —
the same ``coord_scale`` ``draw_bounding_box`` uses for boxes. Note the box tool
is y-first ([y_min, x_min, ...]) while this point is x,y.
"""
import tenacity
from typing import List, Dict, Optional
from ambient.config import settings, get_provider_quality_settings
from ambient.tools.video_backend import make_video_tools
from pydantic import BaseModel, Field

# Half-size of the marker square, in 0-1000 grid units (~2.4% of the frame edge).
_MARKER_HALF = 12.0
_COORD_SCALE = 1000.0


class DrawPointTool(BaseModel):
    video_id: str = Field(description="The id of the video containing the frame.")
    frame_timestamp: float = Field(
        description="The global timestamp of the frame to mark, in seconds."
    )
    point: List[float] = Field(
        description=(
            "The point / click coordinate to visualize, as [x, y] on a 0-1000 grid "
            "where (0,0) is the top-left and (1000,1000) the bottom-right of the frame "
            "(x is horizontal, y is vertical)."
        )
    )
    label: str = Field(description="A short description of what the point marks.")


def _mmss(t: float) -> str:
    t = max(0, int(round(t)))
    return f"{t // 60:02d}:{t % 60:02d}"


@tenacity.retry(
    stop=tenacity.stop_after_attempt(3),
    wait=tenacity.wait_exponential(multiplier=1, min=4, max=10),
    reraise=True,
)
async def draw_point(
    video_id: str,
    frame_timestamp: float,
    point: List[float],
    label: str,
) -> tuple[dict, List[Dict]]:
    if not point or len(point) != 2:
        return (
            {"error": f"point must be [x, y] on a 0-1000 grid, got {point!r}"},
            [],
        )
    x, y = float(point[0]), float(point[1])

    # Point -> small square box centered on it, clamped to the grid, in the box
    # tool's [y_min, x_min, y_max, x_max] order at coord_scale=1000.
    def _clamp(v: float) -> float:
        return max(0.0, min(_COORD_SCALE, v))

    box = [
        _clamp(y - _MARKER_HALF), _clamp(x - _MARKER_HALF),
        _clamp(y + _MARKER_HALF), _clamp(x + _MARKER_HALF),
    ]

    provider_quality_settings = get_provider_quality_settings(settings.llm_model)
    video_tools = make_video_tools(
        video_id, max_frame_dimention=provider_quality_settings.max_dimentions
    )

    annotated: Optional[dict] = None
    if hasattr(video_tools, "annotate_frame"):
        try:
            annotated = video_tools.annotate_frame(
                timestamp=frame_timestamp,
                annotations=[{"bounding_box": box, "label": label}],
                coord_scale=_COORD_SCALE,
            )
        except Exception as e:  # noqa: BLE001 - drawing is best-effort
            print(f"[draw_point] annotate_frame failed at {frame_timestamp}s: {e}")

    # Deterministic click coordinate: the marker centre (== the input point)
    # mapped from the 0-1000 grid to the frame the marker was drawn on.
    nx, ny = round(x / _COORD_SCALE, 4), round(y / _COORD_SCALE, 4)
    result: dict = {
        "clicked_element": label,
        "frame_timestamp": frame_timestamp,
        "citation": _mmss(frame_timestamp),
        "point_grid": [x, y],
        "normalized_coordinates": {"x": nx, "y": ny},
    }
    width = (annotated or {}).get("width")
    height = (annotated or {}).get("height")
    if width and height:
        result["frame_size"] = {"width": width, "height": height}
        result["coordinates"] = {"x": round(nx * width), "y": round(ny * height)}
    if annotated and annotated.get("annotated_url"):
        result["annotated_url"] = annotated["annotated_url"]

    verify_prompt = (
        f"A marker for '{label}' was drawn at the point {[x, y]} (0-1000 grid) on the "
        f"frame at {frame_timestamp} seconds. Carefully inspect the frame and verify the "
        f"marker sits exactly on the intended element. If it is off, retry with corrected "
        f"coordinates."
    )

    user_message_contents: List[Dict] = []
    if annotated and annotated.get("annotated_url"):
        user_message_contents.append({"type": "text", "text": verify_prompt})
        user_message_contents.append(
            {"type": "image_url", "image_url": {"url": annotated["annotated_url"]}}
        )

    return result, user_message_contents


if __name__ == "__main__":
    import asyncio
    import dotenv

    dotenv.load_dotenv()
    result, user_message_contents = asyncio.run(
        draw_point("Seattle_bad_driver_accident", 240, [778, 475], "Police Car")
    )
    print(result)

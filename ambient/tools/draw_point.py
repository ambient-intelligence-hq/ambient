"""draw_point: visualize one or more point / click coordinates on a video frame.

Mirror of ``draw_bounding_box``: the agent predicts click location(s), this tool
draws a marker at each on the real (full-res) frame and hands the annotated image
back so the agent can verify the points land on the intended elements.

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
from ambient.tools.video_backend import resolve_video_tools
from pydantic import BaseModel, Field

# Half-size of the marker square, in 0-1000 grid units (~2.4% of the frame edge).
_MARKER_HALF = 12.0
_COORD_SCALE = 1000.0


class Point(BaseModel):
    point: List[float] = Field(
        description=(
            "The point / click coordinate to visualize, as [x, y] on a 0-1000 grid "
            "where (0,0) is the top-left and (1000,1000) the bottom-right of the frame "
            "(x is horizontal, y is vertical)."
        )
    )
    label: str = Field(description="A short description of what this point marks.")


class DrawPointTool(BaseModel):
    video_id: str = Field(description="The id of the video containing the frame.")
    video_path: Optional[str] = Field(default=None, description="Absolute path to a local video file whose frame to mark instead of the session's initial video (e.g. one you downloaded via the bash tool into the workspace). Leave empty to use the initial video.")
    frame_timestamp: float = Field(
        description="The global timestamp of the frame to mark, in seconds."
    )
    points: List[Point] = Field(
        description="One or more points to visualize on the frame, each with its own [x, y] coordinate and label."
    )


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
    points: List[Dict],
    video_path: Optional[str] = None,
) -> tuple[List[dict], List[Dict]]:
    if not points:
        return ([{"error": "points must be a non-empty list of {point, label}"}], [])

    # Normalize each point: tool args arrive as dicts (from func(**tool_input)),
    # but tolerate Point instances too.
    normalized = [
        p.model_dump() if isinstance(p, Point) else dict(p) for p in points
    ]

    def _clamp(v: float) -> float:
        return max(0.0, min(_COORD_SCALE, v))

    # Point -> small square box centered on it, clamped to the grid, in the box
    # tool's [y_min, x_min, y_max, x_max] order at coord_scale=1000.
    valid: List[Dict] = []  # {x, y, label}
    errors: List[dict] = []
    annotations: List[Dict] = []
    for p in normalized:
        pt = p.get("point")
        label = p.get("label")
        if not pt or len(pt) != 2:
            errors.append({"error": f"point must be [x, y] on a 0-1000 grid, got {pt!r}"})
            continue
        x, y = float(pt[0]), float(pt[1])
        box = [
            _clamp(y - _MARKER_HALF), _clamp(x - _MARKER_HALF),
            _clamp(y + _MARKER_HALF), _clamp(x + _MARKER_HALF),
        ]
        annotations.append({"bounding_box": box, "label": label})
        valid.append({"x": x, "y": y, "label": label})

    if not valid:
        return (errors, [])

    provider_quality_settings = get_provider_quality_settings(settings.llm_model)
    video_tools = resolve_video_tools(
        video_id, video_path, max_frame_dimention=provider_quality_settings.max_dimentions
    )

    annotated: Optional[dict] = None
    if hasattr(video_tools, "annotate_frame"):
        try:
            annotated = video_tools.annotate_frame(
                timestamp=frame_timestamp,
                annotations=annotations,
                coord_scale=_COORD_SCALE,
            )
        except Exception as e:  # noqa: BLE001 - drawing is best-effort
            print(f"[draw_point] annotate_frame failed at {frame_timestamp}s: {e}")

    width = (annotated or {}).get("width")
    height = (annotated or {}).get("height")

    # Deterministic click coordinate per point: the marker centre (== the input
    # point) mapped from the 0-1000 grid to the frame the marker was drawn on.
    results: List[dict] = []
    for v in valid:
        x, y, label = v["x"], v["y"], v["label"]
        nx, ny = round(x / _COORD_SCALE, 4), round(y / _COORD_SCALE, 4)
        result: dict = {
            "clicked_element": label,
            "frame_timestamp": frame_timestamp,
            "citation": _mmss(frame_timestamp),
            "point_grid": [x, y],
            "normalized_coordinates": {"x": nx, "y": ny},
        }
        if width and height:
            result["frame_size"] = {"width": width, "height": height}
            result["coordinates"] = {"x": round(nx * width), "y": round(ny * height)}
        results.append(result)
    # Surface any malformed points alongside the drawn ones.
    results.extend(errors)

    annotated_url = (annotated or {}).get("annotated_url")
    # if annotated_url:
    #     for r in results:
    #         if "error" not in r:
    #             r["annotated_url"] = annotated_url

    points_desc = "; ".join(f"'{v['label']}' at {[v['x'], v['y']]}" for v in valid)
    verify_prompt = (
        f"{len(valid)} marker(s) were drawn on the frame at {frame_timestamp} seconds (0-1000 grid) — "
        f"{points_desc}. Carefully inspect the frame and verify each marker sits exactly on its intended "
        f"element. If any is off, retry with corrected coordinates."
    )

    user_message_contents: List[Dict] = []
    if annotated_url:
        user_message_contents.append({"type": "text", "text": verify_prompt})
        user_message_contents.append(
            {"type": "image_url", "image_url": {"url": annotated_url}}
        )

    return results, user_message_contents


if __name__ == "__main__":
    import asyncio
    import dotenv

    dotenv.load_dotenv()
    points = [
        {"point": [778, 475], "label": "Police Car"},
        {"point": [300, 600], "label": "Pedestrian"},
    ]
    result, user_message_contents = asyncio.run(
        draw_point("Seattle_bad_driver_accident", 240, points)
    )
    print(result)

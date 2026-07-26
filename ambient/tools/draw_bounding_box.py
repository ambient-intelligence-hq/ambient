import tenacity
import os
from typing import List, Dict
from ambient.config import settings, get_provider_quality_settings
from ambient.tools.video_backend import make_video_tools
from ambient.utils.s3 import get_s3_client
import asyncio
from pydantic import BaseModel, Field

s3_client = get_s3_client()
ENABLE_RETURN_CITATION_IMAGES = False


class DrawBoundingBoxTool(BaseModel):
    video_id: str = Field(description="The id of the video to search the clip from.")
    frame_timestamp: float = Field(
        description="The global timestamp of the frame to draw the bounding box on in seconds."
    )
    bounding_box: List[float] = Field(
        description="The bounding box coordinates to draw on the frame. The bounding box coordinates should be in the format [y_min, x_min,y_max, x_max]."
    )
    label: str = Field(description="The label of the bounding box.")


@tenacity.retry(
    stop=tenacity.stop_after_attempt(3),
    wait=tenacity.wait_exponential(multiplier=1, min=4, max=10),
    reraise=True,
)
async def draw_bounding_box(
    video_id: str,
    frame_timestamp: float,
    bounding_box: List[float],
    label: str,
) -> tuple[str, List[Dict]]:
    provider_quality_settings = get_provider_quality_settings(settings.llm_model)
    video_tools = make_video_tools(
        video_id, max_frame_dimention=provider_quality_settings.max_dimentions
    )
    user_message_contents = []
    annotated = None

    try:
        annotated = video_tools.annotate_frame(
            timestamp=frame_timestamp,
            annotations=[{"bounding_box": bounding_box, "label": label}],
        )
    except Exception as e:  # noqa: BLE001 - annotation is non-essential
        print(f"[draw_bounding_box] annotate_frame failed at {frame_timestamp}s: {e}")

    RESULT_PROMPT = (
        f"The bounding box has been drawn on the frame at {frame_timestamp} seconds with the label {label}."
        f"The bounding box coordinates are {bounding_box}."
        "Carefully inspect the frame and verify if your bounding box prediction is correct. If not, retry again with the correct bounding box."
    )

    if annotated and annotated.get("annotated_url"):
        user_message_contents.append({"type": "text", "text": RESULT_PROMPT})
        user_message_contents.append(
            {"type": "image_url", "image_url": {"url": annotated.get("annotated_url")}}
        )

    return annotated, user_message_contents


if __name__ == "__main__":
    import asyncio
    import dotenv

    dotenv.load_dotenv()
    frame_timestamp = 240
    bounding_box = [0.1, 0.1, 0.2, 0.2]
    label = "Police Car"
    result, user_message_contents = asyncio.run(
        draw_bounding_box(
            "Seattle_bad_driver_accident", frame_timestamp, bounding_box, label
        )
    )
    print(result)

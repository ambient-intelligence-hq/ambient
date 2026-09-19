import tenacity
import os
from typing import List, Dict, Optional
from ambient.config import settings, get_provider_quality_settings
from ambient.tools.video_backend import resolve_video_tools
from ambient.utils.s3 import get_s3_client
from ambient.llm import video_to_data_url
import asyncio
from pydantic import BaseModel, Field
from ambient.config import model_modalities, get_model_modalities


s3_client = get_s3_client()
ENABLE_RETURN_CITATION_IMAGES = False


class GrabFramesTool(BaseModel):
    video_id: str = Field(description="The id of the video to search the clip from.")
    video_path: Optional[str] = Field(default=None, description="Absolute path to a local video file to grab frames from instead of the session's initial video (e.g. one you downloaded via the bash tool into the workspace). Leave empty to use the initial video.")
    start_time: float = Field(description="The start time of the clip in seconds.")
    end_time: float = Field(
        description="The end time of the clip to search in seconds. The end time should be within 5 mins from the start_time."
    )


@tenacity.retry(
    stop=tenacity.stop_after_attempt(3),
    wait=tenacity.wait_exponential(multiplier=1, min=4, max=10),
    reraise=True,
)
async def grab_frames(
    video_id: str,
    start_time: float,
    end_time: float,
    video_path: Optional[str] = None,
) -> tuple[str, List[Dict]]:
    provider_quality_settings = get_provider_quality_settings(settings.llm_model)
    video_tools = resolve_video_tools(
        video_id, video_path, max_frame_dimention=provider_quality_settings.max_dimentions
    )
    user_message_contents = []

    print(f"[Tool call] grab_frames: video_id: {video_id} , start_time: {start_time} , end_time: {end_time}")

    try:
        # Returns one or more clips on a single continuous 0-based timeline plus the
        # window's global start (clip_start). With a size cap on a tiled video the
        # covering tiles come back as separate (already-small) clips.
        frames = await asyncio.to_thread(video_tools.fetch_frames,
            video_tools.FPS,start_time,end_time,6
        )

    except Exception as e:
        print(f"[annotate_frames] Error fetching frames: {e}")
        return f"Error fetching frames: {e} , please try again", []

    agent_modalities = get_model_modalities(settings.agent_model)

    result_prompt = (f"The following are the frames between the {start_time} and {end_time} seconds."
                     "You should carefully inspect the frames. If you are using the frames to predict coordinates,"
                     "You must verify your prediction with an appropriate verification tool (draw_bounding_box or draw_point) before committing to your predictions.")

    if agent_modalities and model_modalities.IMAGE in agent_modalities:
        for frame in frames:
            # Need at least one source for the image: the uploaded URL (e2b) or a
            # local file to inline as a data URL (inprocess). Skip only if both are
            # missing — the previous `or` guard dropped every frame lacking a URL.
            if not frame.frame_url and not frame.frame_file_path:
                continue
            frame_url = frame.frame_url or video_to_data_url(frame.frame_file_path, "image/jpeg")
            user_message_contents.append({"type": "image_url", "image_url": {"url": frame_url }})
        # print(f"[search_clip] LLM Response: {llm_response}")
    return result_prompt, user_message_contents


if __name__ == "__main__":
    import asyncio
    import dotenv

    dotenv.load_dotenv()
    start_time = 240
    end_time = 360
    query = (
        "Find the moment when the police car arrived at the scene after the accident."
    )
    result, user_message_contents = asyncio.run(
        grab_frames("Seattle_bad_driver_accident", query, start_time, end_time)
    )
    print(result)

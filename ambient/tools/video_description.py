
from ambient.config import settings
from ambient.tools.video_backend import make_video_tools
from ambient.llm import llm_call
from pydantic import BaseModel, Field
from ambient.prompt import VIDEO_DESCRIPTION_TOOL_PROMPT as prompt
import logging
import time
log = logging.getLogger(__name__)

_TRANSCRIPT = None
MAX_CLIPS = 5
MAX_FRAMES = 30

class VideoDescriptionTool(BaseModel):
    video_id: str = Field(description="The id of the video to get the description of.")


# @tenacity.retry(
#     stop=tenacity.stop_after_attempt(3),
#     wait=tenacity.wait_exponential(multiplier=1, min=4, max=10),
#     reraise=True,
# )
async def get_video_description(video_id: str) -> str:
    # audio_url = s3_client.get_presigned_url(f"{video_id}/audio.mp3")
    # transcription = await get_audio_transcription(audio_url)
    transcription = _TRANSCRIPT
    # print(f"Transcription: {transcription}")

    start_time = time.time()
    video_tools = make_video_tools(video_id, max_frame_dimention=768)
    video_tools.OVERVIEW_MAX_FRAMES = 50
    frames = video_tools.get_overview_frames()
    end_time = time.time()
    log.info(f"Time taken to get overview frames: {end_time - start_time} seconds")
    video_duration = video_tools._duration_sec

    start_time = time.time()
    log.info(f"Requesting LLM with {len(frames)} frames")
    # create a list of clips with presigned urls
    llm_response = await llm_call(
        prompt=prompt,
        query="Provide the description of the video now:",
        model=settings.llm_model,
        base_url=settings.llm_base_url,
        api_key=settings.llm_api_key,
        video_frames=frames,
    )   
    end_time = time.time()
    log.info(f"Time taken to get LLM response: {end_time - start_time} seconds")
    log.info(f"Received LLM response for video {video_id}")
    description = llm_response["choices"][0]["message"]["content"]
    
    if "<video_description>" in description:
        description = (
            description.split("<video_description>")[1]
            .split("</video_description>")[0]
            .strip()
        )

    description = description + f"\n\n Full Video Duration: {video_duration} seconds"

    if transcription is not None:
        result = f"Highlevel approximate description of the video: {description}\n\n Full Video Transcription / Subtitles (noisy): {transcription}"
    else:
        result = f"Highlevel approximate description of the video: {description}"
    return result , []

import tenacity
from typing import List, Dict
from ambient.config import settings, get_provider_quality_settings
from ambient.tools.video_backend import make_video_tools
from ambient.utils.s3 import get_s3_client
from ambient.llm import llm_call
from ambient.tools.citations import (
    _extract_temporal_citations,
    _build_user_message_contents_from_citations,
    _replace_citations_with_global_video_timestamps
)
from pydantic import BaseModel, Field
# from ambient.tools.video_description import get_video_description
from ambient.prompt import FOCUS_CLIP_TOOL_PROMPT

s3_client = get_s3_client()
ENABLE_RETURN_CITATION_IMAGES = False


class FocusClipTool(BaseModel):
    video_id: str = Field(description="The id of the video to focus the clip from.")
    start_time: float = Field(
        description="The start time of the clip to focus in seconds."
    )
    end_time: float = Field(
        description="The end time of the clip to focus in seconds. The end time should be within 5 mins from the start_time."
    )


@tenacity.retry(
    stop=tenacity.stop_after_attempt(3),
    wait=tenacity.wait_exponential(multiplier=1, min=4, max=10),
    reraise=True,
)
async def focus_clip(
    video_id: str, start_time: float, end_time: float, video_description: str = None
) -> tuple[str, List[Dict]]:
    provider_quality_settings = get_provider_quality_settings(settings.llm_model)
    video_tools = make_video_tools(
        video_id, max_frame_dimention=provider_quality_settings.max_dimentions
    )

    # Returns one or more clips on a single continuous 0-based timeline plus the
    # window's global start (clip_start). When a size cap is set and the video is
    # tiled, the covering tiles come back as separate (already-small) clips; the
    # model still reads them as one clip and cites mm:ss within the window.
    clips, clip_start = await video_tools.fetch_clips(
        start_time,
        end_time,
        fps=provider_quality_settings.fps,
        crf=provider_quality_settings.crf,
        max_size_mb=provider_quality_settings.max_size_mb,
    )
    # The e2b backend already uploaded clips and set clip_url; only the local
    # backend returns local files that still need uploading.
    for clip in clips:
        if clip.clip_url is None:
            if settings.inline_clips:
                # No S3: leave clip_url unset so construct_payload embeds the clip
                # as a base64 data URL instead.
                continue
            key = f"{clip.video_id}/clips/{clip.id}.mp4"
            s3_client.upload_file(clip.clip_file_path, key)
            clip.clip_url = s3_client.get_presigned_url(key, expires_in=7200)
    user_message_contents = []

    PROMPT = FOCUS_CLIP_TOOL_PROMPT
    if video_description:
        PROMPT = PROMPT + f"\n\n Highlevel overview of the Video:\n{video_description}"

    llm_response = await llm_call(
        prompt=PROMPT,
        query="Provide the description of the video now:",
        model=settings.llm_model,
        base_url=settings.llm_base_url,
        api_key=settings.llm_api_key,
        video_clips=clips,
    )
    # print(f"[focus_clip] LLM Response: {llm_response}")
    description = llm_response["choices"][0]["message"]["content"]
    try:
        description = (
            description.split("<video_description>")[1]
            .split("</video_description>")[0]
            .strip()
        )
    except Exception:
        description = description.strip()

    citations = _extract_temporal_citations(description)
    # clip_start is the window's global start (clips are labelled window-local, and
    # may be snapped to tile boundaries), so the model's local mm:ss citations map
    # to absolute video time by adding it.
    if ENABLE_RETURN_CITATION_IMAGES:
        user_message_contents = _build_user_message_contents_from_citations(
            video_tools, citations, clip_start, end_time
        )

    description = _replace_citations_with_global_video_timestamps(description, citations, clip_start)
    return description, user_message_contents


if __name__ == "__main__":
    import asyncio
    import dotenv
    dotenv.load_dotenv()
    start_time = 240
    end_time = 360
    result, user_message_contents = asyncio.run(focus_clip("Seattle_bad_driver_accident", start_time, end_time))
    print(result)
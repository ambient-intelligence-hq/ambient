import logging
import asyncio
import aiohttp
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
log = logging.getLogger(__name__)

class FocusClipTool(BaseModel):
    video_id: str = Field(description="The id of the video to focus the clip from.")
    start_time: float = Field(
        description="The start time of the clip to focus in seconds."
    )
    end_time: float = Field(
        description="The end time of the clip to focus in seconds. The end time should be within 5 mins from the start_time."
    )


async def focus_clip(
    video_id: str, start_time: float, end_time: float, video_description: str = None
) -> tuple[str, List[Dict]]:
    provider_quality_settings = get_provider_quality_settings(settings.llm_model)
    video_tools = make_video_tools(
        video_id, max_frame_dimention=provider_quality_settings.max_dimentions
    )

    from ambient.tools.clip_media import produce_window_media

    # media_plan decides video-clips vs image-frames for this window (frames only
    # where the endpoint can't control a video's frame count). Returns the window's
    # global start (clip_start) for mapping the sub-model's citations back. Clips
    # ride a single 0-based timeline; the model reads them as one clip and cites
    # mm:ss within the window.
    kind, media, clip_start, _plan = await produce_window_media(
        video_tools, "focus_clip", start_time, end_time)
    clips = media if kind == "clips" else None
    frames = media if kind == "frames" else None

    print(f"[Tool call] focus_clip: kind: {kind} , media_count: {len(media)} , clip_start: {clip_start} , _plan: {_plan}")

    # The e2b backend already uploaded clips and set clip_url; only the local
    # backend returns local files that still need uploading. (Frames come back
    # already inlined/uploaded by the backend.)
    for clip in (clips or []):
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

    try:
        llm_response = await llm_call(
            prompt=PROMPT,
            query="Provide the description of the video now:",
            model=settings.llm_model,
            base_url=settings.llm_base_url,
            api_key=settings.llm_api_key,
            video_clips=clips,
            video_frames=frames,
            timeout=120,
        )

    except (aiohttp.ClientResponseError, asyncio.TimeoutError) as exc:
        status = getattr(exc, "status", None)
        log.error("[focus_clip] LLM request failed for %s (%s-%ss): %s", video_id, start_time, end_time, exc)
        return (
            f"Error analyzing clip: LLM request failed"
            f"{f' (HTTP {status})' if status else ''}. "
            "The analysis server may be overloaded or the clip may be too large. "
            "Try a smaller time window or retry later.",
            [],
        )
    except Exception as exc:
        log.exception("[focus_clip] unexpected error for %s", video_id)
        return f"Error analyzing clip: {type(exc).__name__}: {exc}", []

    choices = llm_response.get("choices") or []
    if not choices:
        log.error("[focus_clip] No choices in LLM response: %s", llm_response)
        return "Error analyzing clip: LLM returned an empty response. Please try again.", []

    description = (choices[0].get("message") or {}).get("content") or ""
    if not description:
        return "Error analyzing clip: LLM returned no content. Please try again.", []

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
    # Clips are labelled window-local (and may be snapped to tile boundaries), so
    # their citations need clip_start added. Frames are labelled with absolute
    # video timestamps already -> no offset.
    citation_offset = 0.0 if kind == "frames" else clip_start
    if ENABLE_RETURN_CITATION_IMAGES:
        user_message_contents = _build_user_message_contents_from_citations(
            video_tools, citations, citation_offset, end_time
        )

    description = _replace_citations_with_global_video_timestamps(description, citations, citation_offset)
    return description, user_message_contents


if __name__ == "__main__":
    import asyncio
    import dotenv
    dotenv.load_dotenv()
    start_time = 240
    end_time = 360
    result, user_message_contents = asyncio.run(focus_clip("Seattle_bad_driver_accident", start_time, end_time))
    print(result)
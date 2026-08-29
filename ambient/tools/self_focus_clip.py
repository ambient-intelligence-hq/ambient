import logging
from typing import List, Dict
from ambient.config import settings, get_provider_quality_settings
from ambient.tools.video_backend import make_video_tools
from ambient.utils.s3 import get_s3_client
from pydantic import BaseModel, Field
# from ambient.tools.video_description import get_video_description
from ambient.prompt import FOCUS_CLIP_TOOL_PROMPT

s3_client = get_s3_client()
ENABLE_RETURN_CITATION_IMAGES = False
log = logging.getLogger(__name__)

class SelfFocusClipTool(BaseModel):
    video_id: str = Field(description="The id of the video to focus the clip from.")
    start_time: float = Field(
        description="The start time of the clip to focus in seconds."
    )
    end_time: float = Field(
        description="The end time of the clip to focus in seconds. The end time should be within 5 mins from the start_time."
    )


async def self_focus_clip(
    video_id: str, start_time: float, end_time: float, video_description: str = None
) -> tuple[str, List[Dict]]:

    user_message_contents = []
    provider_quality_settings = get_provider_quality_settings(settings.llm_model)
    video_tools = make_video_tools(
        video_id, max_frame_dimention=provider_quality_settings.max_dimentions
    )

    from ambient.tools.clip_media import produce_window_media

    # media_plan decides video-clips vs image-frames for this window: self-hosted
    # vLLM gets the video (it samples internally, cheapest via temporal merge);
    # endpoints that can't control a video's frame count get a fixed number of
    # frames instead. Clips ride a single 0-based timeline; the model reads them as
    # one clip and cites mm:ss within the window.
    kind, media, _clip_start, _plan = await produce_window_media(
        video_tools, "focus_clip", start_time, end_time)

    print(f"[Tool call] self_focus_clip: kind: {kind} , media_count: {len(media)} , _clip_start: {_clip_start} , _plan: {_plan} ")

    if kind == "frames":
        from ambient.llm import video_to_data_url
        for frame in media:
            # Need at least one source: the uploaded URL (e2b) or a local file to
            # inline (inprocess); rangeproxy already sets a data URL.
            if not frame.frame_url and not frame.frame_file_path:
                continue
            frame_url = frame.frame_url or video_to_data_url(frame.frame_file_path, "image/jpeg")
            user_message_contents.append({"type": "image_url", "image_url": {"url": frame_url}})
    else:
        # The e2b backend already uploaded clips and set clip_url; only the local
        # backend returns local files that still need uploading.
        for clip in media:
            if clip.clip_url is None:
                if settings.inline_clips:
                    # No S3: leave clip_url unset so construct_payload embeds the clip
                    # as a base64 data URL instead.
                    continue
                key = f"{clip.video_id}/clips/{clip.id}.mp4"
                s3_client.upload_file(clip.clip_file_path, key)
                clip.clip_url = s3_client.get_presigned_url(key, expires_in=7200)

            user_message_contents.extend([
                {
                    "type": "text",
                    "text": f"Clip of the video between {clip.start_time} seconds and {clip.end_time} seconds",
                },
                {
                    "type": "video_url",
                    "video_url": {"url": clip.clip_url},
                }
            ])

    PROMPT = FOCUS_CLIP_TOOL_PROMPT
    if video_description:
        PROMPT = PROMPT + f"\n\n Highlevel overview of the Video:\n{video_description}"
    
    
    return f"Here is the clip of the video between {start_time} seconds and {end_time} seconds, carefully review and decide next actions", user_message_contents
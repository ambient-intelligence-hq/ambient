import logging
from typing import List, Dict, Optional
from ambient.config import settings, get_provider_quality_settings
from ambient.tools.video_backend import resolve_video_tools
from ambient.utils.s3 import get_s3_client
from pydantic import BaseModel, Field
# from ambient.tools.video_description import get_video_description
from ambient.prompt import FOCUS_CLIP_TOOL_PROMPT

s3_client = get_s3_client()
ENABLE_RETURN_CITATION_IMAGES = False
log = logging.getLogger(__name__)

class SelfFocusClipTool(BaseModel):
    video_id: str = Field(description="The id of the video to focus the clip from.")
    video_path: Optional[str] = Field(default=None, description="Absolute path to a local video file to focus instead of the session's initial video (e.g. one you downloaded via the bash tool into the workspace). Leave empty to use the initial video.")
    start_time: float = Field(
        description="The start time of the clip to focus in seconds."
    )
    end_time: float = Field(
        description="The end time of the clip to focus in seconds. The end time should be within 5 mins from the start_time."
    )


async def self_focus_clip(
    video_id: str, start_time: float, end_time: float, video_description: str = None,
    video_path: Optional[str] = None,
) -> tuple[str, List[Dict]]:

    user_message_contents = []
    provider_quality_settings = get_provider_quality_settings(settings.llm_model)
    video_tools = resolve_video_tools(
        video_id, video_path, max_frame_dimention=provider_quality_settings.max_dimentions
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
        # Label every clip in ABSOLUTE video time. A window may come back as several
        # consecutive tiles whose own start/end ride a 0-based per-window timeline
        # (see sandbox_video_tools._tiles_as_local_clips), so `clip.start_time` is
        # NOT absolute and repeats window-to-window. Anchor on the window's true
        # global offset (`_clip_start`) and walk each segment's duration instead —
        # duration is convention-independent, so this is correct for both the tiled
        # and single-clip paths and never emits misleading/duplicated 0-based labels.
        base = float(_clip_start) if _clip_start is not None else float(start_time)
        local = 0.0
        n = len(media)
        # The e2b backend already uploaded clips and set clip_url; only the local
        # backend returns local files that still need uploading.
        for idx, clip in enumerate(media):
            if clip.clip_url is None:
                if settings.inline_clips:
                    # No S3: leave clip_url unset so construct_payload embeds the clip
                    # as a base64 data URL instead.
                    continue
                key = f"{clip.video_id}/clips/{clip.id}.mp4"
                s3_client.upload_file(clip.clip_file_path, key)
                clip.clip_url = s3_client.get_presigned_url(key, expires_in=7200)

            dur = None
            if clip.start_time is not None and clip.end_time is not None:
                dur = max(float(clip.end_time) - float(clip.start_time), 0.0)
            seg_start = base + local
            seg_end = seg_start + dur if dur is not None else float(end_time)
            if dur is not None:
                local += dur
            if n == 1:
                label = f"Clip of the video between {seg_start:.1f} and {seg_end:.1f} seconds"
            else:
                label = (
                    f"Segment {idx + 1} of {n} (consecutive) — video between "
                    f"{seg_start:.1f} and {seg_end:.1f} seconds; together these segments "
                    f"cover the {float(start_time):.1f}-{float(end_time):.1f}s window"
                )

            user_message_contents.extend([
                {
                    "type": "text",
                    "text": label,
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
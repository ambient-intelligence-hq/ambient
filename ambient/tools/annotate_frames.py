import tenacity
import os
from typing import List, Dict
from ambient.config import settings, get_provider_quality_settings
from ambient.tools.video_backend import make_video_tools
from ambient.utils.s3 import get_s3_client
from ambient.llm import llm_call
import asyncio
from pydantic import BaseModel, Field

# from ambient.tools.video_description import get_video_description
from ambient.prompt import ANNOTATE_FRAMES_TOOL_PROMPT
from ambient.tools.citations import parse_frame_annotations
from ambient.config import model_modalities, get_model_modalities

s3_client = get_s3_client()
ENABLE_RETURN_CITATION_IMAGES = False


class AnnotateFramesTool(BaseModel):
    video_id: str = Field(description="The id of the video to search the clip from.")
    start_time: float = Field(description="The start time of the clip in seconds.")
    end_time: float = Field(
        description="The end time of the clip to search in seconds. The end time should be within 5 mins from the start_time."
    )
    annotation_prompt: str = Field(
        description="Detailed description of the annotation you want to add to the frame. The LLM tool would return the bounding box coordinates of the annotation"
    )


@tenacity.retry(
    stop=tenacity.stop_after_attempt(3),
    wait=tenacity.wait_exponential(multiplier=1, min=4, max=10),
    reraise=True,
)
async def annotate_frames(
    video_id: str,
    start_time: float,
    end_time: float,
    annotation_prompt: str = None,
    video_description: str = None,
) -> tuple[str, List[Dict]]:
    provider_quality_settings = get_provider_quality_settings(settings.llm_model)
    video_tools = make_video_tools(
        video_id, max_frame_dimention=provider_quality_settings.max_dimentions
    )
    user_message_contents = []

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

    # # The e2b backend already uploaded clips and set clip_url. Only the local
    # # backend returns bare local files that still need handling here.
    # for clip in clips:
    #     if clip.clip_url is None:
    #         if is_local_llm:
    #             # Local server can't reach an S3 presigned URL; leave clip_url unset
    #             # so construct_payload embeds the clip as a base64 data URL instead.
    #             clip.clip_url = None
    #         else:
    #             key = f"{clip.video_id}/clips/{clip.id}.mp4"
    #             s3_client.upload_file(clip.clip_file_path, key)
    #             clip.clip_url = s3_client.get_presigned_url(key, expires_in=7200)

    prompt = ANNOTATE_FRAMES_TOOL_PROMPT
    prompt = prompt.replace("{annotation}", annotation_prompt or "")

    if video_description:
        prompt = prompt + f"\n\n Highlevel overview of the Video:\n{video_description}"

    print(f"[annotate_frames] frames: {frames}")

    llm_response = await llm_call(
        prompt=prompt,
        query="Now annotate the frames with the given annotation.",
        model=settings.llm_model,
        base_url=settings.llm_base_url,
        api_key=settings.llm_api_key,
        video_frames=frames,
    )

    reasoning = (
        llm_response.get("raw", {})
        .get("choices", [{}])[0]
        .get("message", {})
        .get("reasoning")
    )
    if "choinces" not in llm_response:
        print(f"[search_clip] No choices in LLM Response: {llm_response}")
    response_text = llm_response["choices"][0]["message"]["content"]

    frame_annotations = parse_frame_annotations(response_text, start_time)


    # Draw the model's bounding boxes onto the real frames and attach a shareable
    # image URL to each annotated frame. The model emits [y_min, x_min, y_max,
    # x_max] normalized to 0-1000 (Gemini convention); the sandbox denormalizes to
    # the still's pixels, so the box is drawn on the actual video frame at its
    # global timestamp. Best-effort: a failed draw never sinks the tool result.
    if hasattr(video_tools, "annotate_frame"):
        for frame in frame_annotations:
            if not frame.get("bounding_box"):
                continue
            try:
                annotated = video_tools.annotate_frame(
                    timestamp=frame["global_timestamp"],
                    annotations=[frame],
                )
                if annotated and annotated.get("annotated_url"):
                    frame["annotated_url"] = annotated["annotated_url"]
            except Exception as e:  # noqa: BLE001 - annotation is non-essential
                print(
                    f"[grab_frames] annotate_frame failed at "
                    f"{frame.get('global_timestamp')}s: {e}"
                )

    # TODO: Build user message contents from frame annotations to return to the LLM
    # if ENABLE_RETURN_CITATION_IMAGES:
    #     user_message_contents = _build_user_message_contents_from_citations(
    #         video_tools, citations, clip_start, end_time
    #     )

    result = f"Agent Reasoning: {reasoning}\nFinal Response: {response_text}"

    agent_modalities = get_model_modalities(settings.agent_model)

    if agent_modalities and model_modalities.IMAGE in agent_modalities:
        for frame in frame_annotations:
            if not frame.get("annotated_url"):
                continue
            user_message_contents.append({"type": "image_url", "image_url": {"url": frame.get("annotated_url") }})
        # print(f"[search_clip] LLM Response: {llm_response}")
    return result, user_message_contents


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
        annotate_frames("Seattle_bad_driver_accident", query, start_time, end_time)
    )
    print(result)

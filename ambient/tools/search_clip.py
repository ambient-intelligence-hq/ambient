import tenacity
from typing import List, Dict
from ambient.config import settings, get_provider_quality_settings
from ambient.tools.video_backend import make_video_tools
from ambient.utils.s3 import get_s3_client
from ambient.llm import llm_call
from ambient.tools.citations import _extract_temporal_citations, _build_user_message_contents_from_citations
from pydantic import BaseModel, Field
# from ambient.tools.video_description import get_video_description
from ambient.prompt import SEARCH_CLIP_TOOL_PROMPT
from ambient.tools.citations import _replace_citations_with_global_video_timestamps

s3_client = get_s3_client()
ENABLE_RETURN_CITATION_IMAGES = False

class SearchClipTool(BaseModel):
    video_id: str = Field(description="The id of the video to search the clip from.")
    query: str = Field(description="The query to search the clip for. Provide a detailed descriptive query of what you are looking for in the video.")
    start_time: float = Field(description="The start time of the clip to search in seconds.")
    end_time: float = Field(description="The end time of the clip to search in seconds. The end time should be within 5 mins from the start_time.")

@tenacity.retry(
    stop=tenacity.stop_after_attempt(3),
    wait=tenacity.wait_exponential(multiplier=1, min=4, max=10),
    reraise=True,
)
async def search_clip(video_id: str, query: str, start_time: float, end_time: float, video_description: str = None) -> tuple[str, List[Dict]]:
    provider_quality_settings = get_provider_quality_settings(settings.llm_model)
    video_tools = make_video_tools(video_id, max_frame_dimention=provider_quality_settings.max_dimentions)
    user_message_contents = []
    global PROMPT

    try:
        # Returns one or more clips on a single continuous 0-based timeline plus the
        # window's global start (clip_start). With a size cap on a tiled video the
        # covering tiles come back as separate (already-small) clips.
        clips, clip_start = await video_tools.fetch_clips(start_time, end_time, fps=provider_quality_settings.fps,crf=provider_quality_settings.crf,max_size_mb=provider_quality_settings.max_size_mb)
    except ValueError as e:
        if "clip size is still too large" in str(e):
            return f"Error fetching clip: {e} , please try again with a smaller time window", []
        return f"Error fetching clip: {e} , please try again", []

    base_url = settings.llm_base_url or ""
    is_local_llm = "localhost" in base_url or "127.0.0.1" in base_url

    # The e2b backend already uploaded clips and set clip_url. Only the local
    # backend returns bare local files that still need handling here.
    for clip in clips:
        if clip.clip_url is None:
            if is_local_llm or settings.inline_clips:
                # Local server can't reach an S3 presigned URL (and inline_clips
                # forces this for an S3-less run); leave clip_url unset so
                # construct_payload embeds the clip as a base64 data URL instead.
                clip.clip_url = None
            else:
                key = f"{clip.video_id}/clips/{clip.id}.mp4"
                s3_client.upload_file(clip.clip_file_path, key)
                clip.clip_url = s3_client.get_presigned_url(key,expires_in=7200)

    PROMPT = SEARCH_CLIP_TOOL_PROMPT
    if video_description:
        PROMPT = PROMPT + f"\n\n Highlevel overview of the Video:\n{video_description}"

    print(f"[search_clip] clips: {clips}")

    llm_response = await llm_call(
        prompt=PROMPT,
        query=f"Query: {query}",
        model=settings.llm_model,
        base_url=settings.llm_base_url,
        api_key=settings.llm_api_key,
        video_clips=clips,
    )

    reasoning = llm_response.get('raw', {}).get('choices', [{}])[0].get('message', {}).get('reasoning')
    if "choinces" not in llm_response:
        print(f"[search_clip] No choices in LLM Response: {llm_response}")
    response_text = llm_response["choices"][0]["message"]["content"]
    
    citations = _extract_temporal_citations(response_text)
    # clip_start is the window's global start; the model's local mm:ss citations
    # map to absolute video time by adding it.
    if ENABLE_RETURN_CITATION_IMAGES:
        user_message_contents = _build_user_message_contents_from_citations(video_tools, citations, clip_start, end_time)

    result = f"Agent Reasoning: {reasoning}\nFinal Response: {response_text}"
    result = _replace_citations_with_global_video_timestamps(result, citations, clip_start)
    # print(f"[search_clip] LLM Response: {llm_response}")
    return result, user_message_contents

if __name__ == "__main__":
    import asyncio
    import dotenv
    dotenv.load_dotenv()
    start_time = 240
    end_time = 360
    query = "Find the moment when the police car arrived at the scene after the accident."
    result, user_message_contents = asyncio.run(search_clip("Seattle_bad_driver_accident", query, start_time, end_time))
    print(result)
  

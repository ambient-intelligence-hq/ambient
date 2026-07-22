import base64
from typing import List , Optional
import os
from ambient.config import settings
import aiohttp
import asyncio
from pydantic import BaseModel
import tenacity
from ambient import Clip, Frame
import logging

logger = logging.getLogger(__name__)

RETRYABLE_STATUS_CODES = {408, 409, 425, 429, 500, 502, 503, 504}


def is_retryable_exception(exc: BaseException) -> bool:
    if isinstance(exc, (aiohttp.ClientConnectionError, asyncio.TimeoutError)):
        return True
    if isinstance(exc, aiohttp.ClientResponseError):
        return exc.status in RETRYABLE_STATUS_CODES
    return False


def video_to_data_url(path: str, mime="video/mp4") -> str:
    with open(path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")
    return f"data:{mime};base64,{b64}"


def construct_payload(clips: Optional[List[Clip]] = None, frames: Optional[List[Frame]] = None):
    payload = []
    if clips is None and frames is None:
        raise ValueError("Either clips or frames must be provided")
    if clips is not None:
        for clip in clips:
            # check if local file or s3 url
            is_clip_url = clip.clip_url is not None
            is_clip_file_path = clip.clip_file_path is not None and os.path.exists(clip.clip_file_path)

            if is_clip_url:
                url = clip.clip_url
            elif is_clip_file_path:
                url = video_to_data_url(clip.clip_file_path)
            else:
                raise ValueError(f"Clip {clip.id} has no valid url or file path")


            timestamp = f"Timestamp: {clip.start_time} seconds to {clip.end_time} seconds"

            payload.extend(
                [
                    {
                        "type": "text",
                        "text": f"Clip ID: {clip.id}\n{timestamp}",
                    },
                    {
                        "type": "video_url",
                        "video_url": {"url": url},
                    },
                ]
            )

    if frames is not None:
        start_time = frames[0].timestamp
        end_time = frames[-1].timestamp
        payload.append(
                {
                    "type": "text",
                    "text": f"Frames from video between {start_time} seconds and {end_time} seconds",
                }
            )
        for idx,frame in enumerate(frames):
            # check if local file or s3 url
            is_frame_url = frame.frame_url is not None
            is_frame_file_path = frame.frame_file_path is not None and os.path.exists(frame.frame_file_path)

            if is_frame_url:
                url = frame.frame_url
            elif is_frame_file_path:
                url = video_to_data_url(frame.frame_file_path,"image/jpeg")
            else:
                raise ValueError(f"Frame {frame.id} has no valid url or file path")

            timestamp = f"Timestamp: {frame.timestamp} seconds"

            if idx%5 == 0 and idx+5 < len(frames):
                payload.append(
                    {
                        "type": "text",
                        "text": f"Frames between {frame.timestamp} seconds and {frames[idx+5].timestamp} seconds",
                    }
                )
            
            payload.append(
                {
                    "type": "image_url",
                    "image_url": {"url": url},
                }
            )

    return payload


def get_provider_params(model: str, base_url: str) -> dict:
    if "gemini" in model.lower() and "openrouter" in base_url.lower():
        return {
            "provider": {
                "only": ["google-ai-studio"],
            },
        }
    
    # if "qwen3.5-35b-a3b" in model.lower() and "openrouter" in base_url.lower():
    #     return {
    #         "provider": {
    #             "only": ["atlas-cloud/fp8"],
    #         },  
    #     }
    return {}


@tenacity.retry(
    stop=tenacity.stop_after_attempt(2),
    wait=tenacity.wait_exponential(multiplier=1, min=2, max=6),
    retry=tenacity.retry_if_exception(is_retryable_exception),
)
async def llm_call(
    prompt: str,
    query: str,
    model: str,
    base_url: str,
    api_key: str,
    video_clips: Optional[List[Clip]] = None,
    history: Optional[List[dict]] = None,
    timeout: int = 300,
    video_frames: Optional[List[Frame]] = None,
    reasoning_enabled: bool = True,
    max_tokens: Optional[int] = None,
) -> BaseModel:
    payload = construct_payload(video_clips,video_frames)
    
    provider_params = get_provider_params(model, base_url)
    # print(payload)
    logger.info(f"Making LLM call to model {model} with {len(payload)} items")
    async with aiohttp.ClientSession(
        timeout=aiohttp.ClientTimeout(total=timeout)
    ) as session:
        async with session.post(
            f"{base_url}/chat/completions",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": settings.llm_model,
                "messages": [
                    {"role": "system", "content": prompt},
                    *([{
                        "role": "user",
                        "content": f"History Context:\n\n{history}",
                    }] if history else []),
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": f"query: {query}"},
                            *payload,
                        ],
                    },
                ],
                "reasoning": {"enabled": reasoning_enabled},
                **({"max_tokens": max_tokens} if max_tokens else {}),
                **provider_params,
            },
        ) as response:
            if response.status != 200:
                body = await response.text()
                logger.error("LLM call failed: %s %s", response.status, body)
                raise aiohttp.ClientResponseError(
                    response.request_info,
                    response.history,
                    status=response.status,
                    message=body,
                    headers=response.headers,
                )
            try:
                response_json = await response.json()
                logger.info(f"LLM call successful for model {model}")
            except Exception as e:
                print(f"Error: {e} {await response.text()}")
                raise e
            return response_json
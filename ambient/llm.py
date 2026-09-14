import base64
import contextvars
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

# Per-call sink for LLM token usage. When a caller (e.g. the tool dispatcher) sets
# this to a list, every `llm_call` in that context appends a normalized usage
# record {model, usage} to it. `asyncio.to_thread` copies the context into the
# worker thread, so tool LLM calls running off-thread still report back into the
# same list object. None (the default) means "don't collect".
usage_sink: contextvars.ContextVar[Optional[list]] = contextvars.ContextVar(
    "llm_usage_sink", default=None
)


def normalize_usage(raw: Optional[dict]) -> dict:
    """Map an OpenAI/OpenRouter chat-completions `usage` block onto the ATIF-style
    token fields used across the session (input/output/cache/total).

    OpenRouter also reports the actual `cost` (USD) and cache details when the
    request opts in (`usage: {include: true}` / `stream_options.include_usage`);
    we carry `cost` through when present so callers can prefer it over estimation.
    """
    raw = raw or {}
    prompt_details = raw.get("prompt_tokens_details") or {}
    cache_read = int(
        prompt_details.get("cached_tokens")
        or raw.get("cache_read_input_tokens")
        or raw.get("prompt_cache_hit_tokens")
        or 0
    )
    cache_write = int(
        prompt_details.get("cache_write_tokens")
        or raw.get("cache_creation_input_tokens")
        or 0
    )
    input_tokens = int(raw.get("prompt_tokens") or raw.get("input_tokens") or 0)
    output_tokens = int(raw.get("completion_tokens") or raw.get("output_tokens") or 0)
    total = int(raw.get("total_tokens") or (input_tokens + output_tokens))
    norm = {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "cache_read_input_tokens": cache_read,
        "cache_creation_input_tokens": cache_write,
        "total_tokens": total,
    }
    # Provider-reported cost (OpenRouter). None when the provider didn't report it.
    if raw.get("cost") is not None:
        try:
            norm["cost"] = float(raw["cost"])
        except (TypeError, ValueError):
            pass
    return norm


def _record_usage(model: str, raw_usage: Optional[dict]) -> None:
    # Store the provider's raw usage block; the consumer (runner) normalizes it
    # once so cost/token extraction stays in one place.
    sink = usage_sink.get()
    if sink is None or not raw_usage:
        return
    sink.append({"model": model, "usage": raw_usage})

RETRYABLE_STATUS_CODES = {408, 409, 425, 429, 500, 502, 503, 504}


def is_retryable_exception(exc: BaseException) -> bool:
    if isinstance(exc, (aiohttp.ClientConnectionError, asyncio.TimeoutError)):
        print(f"LLM call failed, Retryable exception: {exc}")
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
    """OpenRouter provider-routing params for `model`, from the preferred-provider
    lookup (``artifacts/openrouter_providers.json`` via
    ``config.get_openrouter_route``).

    Pins the maintained upstream with ``provider.only`` so a model always lands on
    the same vetted provider — on OpenRouter the frame budget and image-block cap are
    a per-upstream lottery otherwise (same model id, 69->1083 frames across upstreams;
    see findings.md). Returns ``{}`` to auto-route when the model has no preferred
    provider, and for non-OpenRouter endpoints.
    """
    if "openrouter" not in (base_url or "").lower():
        return {}
    from ambient.config import get_openrouter_route

    provider = get_openrouter_route(model).get("provider")
    if provider:
        return {"provider": {"only": [provider]}}
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
    response_format: Optional[dict] = None,
) -> BaseModel:
    payload = construct_payload(video_clips,video_frames)

    provider_params = get_provider_params(model, base_url)
    # Native structured outputs: when a response_format (json_schema) is set, also
    # tell OpenRouter to only route to providers that actually honor it, so we
    # don't silently land on one that ignores the schema.
    if response_format:
        prov = dict(provider_params.get("provider") or {})
        prov["require_parameters"] = True
        provider_params = {**provider_params, "provider": prov}
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
                # Opt into OpenRouter usage accounting so tool LLM calls report
                # token counts + actual cost (picked up by the usage sink).
                "usage": {"include": True},
                **({"max_tokens": max_tokens} if max_tokens else {}),
                **({"response_format": response_format} if response_format else {}),
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
            _record_usage(settings.llm_model, response_json.get("usage"))
            return response_json
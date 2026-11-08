"""Streaming OpenAI chat-completions client.

Yields parsed chunk dicts. The HTTP route relays each as
`event: chat.completion.chunk`.
"""
from __future__ import annotations

import asyncio
import json
import logging
from typing import AsyncIterator, Optional

import aiohttp

from ambient.config import settings
from ambient.llm import get_provider_params
from ambient.media_plan import classify_service

# Hosted APIs that validate request bodies strictly (Fireworks rejects unknown
# fields per model), so the OpenRouter-only `usage`/`reasoning` fields are omitted.
_STRICT_SERVICES = {"fireworks", "together", "openai"}

logger = logging.getLogger(__name__)

# Transient upstream failures are retried, but only before the first chunk: once
# output has streamed to the client, a retry would duplicate it. One provider 500
# (seen on OpenRouter/Alibaba) used to end the whole agent run.
_RETRY_STATUS = frozenset({429, 500, 502, 503, 504, 529})
_RETRY_DELAYS = (2.0, 8.0)


class _RetryableStatus(Exception):
    pass


async def stream_chat_completion(
    *,
    model: str,
    messages: list[dict],
    tools: list[dict],
    max_tokens: int = 16000,
    reasoning_enabled: bool = True,
    base_url: Optional[str] = None,
    api_key: Optional[str] = None,
    timeout: int = 300,
    response_format: Optional[dict] = None,
    extra_body: Optional[dict] = None,
) -> AsyncIterator[dict]:
    endpoint = (base_url or settings.agent_base_url or settings.llm_base_url or "").rstrip("/")
    api_key = api_key or settings.agent_api_key or settings.llm_api_key
    if not endpoint:
        raise ValueError(
            f"No LLM base_url configured for model {model!r}: set AGENT_BASE_URL/LLM_BASE_URL "
            "in .env (loaded relative to the repo root), or set base_url on the agent record. "
            "Without it the endpoint resolves to a bare '/chat/completions', which is not a valid URL."
        )
    if not endpoint.endswith("/chat/completions"):
        endpoint = f"{endpoint}/chat/completions"

    body = {
        "model": model,
        "max_tokens": max_tokens,
        "messages": messages,
        "stream": True,
        # Ask the provider to include a final usage chunk in the stream so the
        # runner can attribute token/cost to the session. `stream_options` is the
        # OpenAI-native switch; `usage.include` is OpenRouter's (which also returns
        # the actual USD cost). Sending both maximizes coverage, except to strict
        # hosted APIs that reject unknown fields.
        "stream_options": {"include_usage": True},
    }
    service = classify_service(endpoint)
    strict = service in _STRICT_SERVICES
    if service == "openai":
        # OpenAI's newer models reject max_tokens on chat completions.
        body["max_completion_tokens"] = body.pop("max_tokens")
    if not strict:
        body["usage"] = {"include": True}
    # Only include tools/tool_choice when there actually are tools — some servers
    # (vLLM) reject an empty `tools` array. Fast mode passes tools=[].
    if tools:
        body["tools"] = tools
        body["tool_choice"] = "auto"
    if reasoning_enabled and not strict:
        body["reasoning"] = {"enabled": True}
    provider_params = get_provider_params(model, endpoint)
    if response_format:
        body["response_format"] = response_format
        # Only route to providers that honor structured outputs.
        prov = dict(provider_params.get("provider") or {})
        prov["require_parameters"] = True
        provider_params = {**provider_params, "provider": prov}
    body.update(provider_params)
    if extra_body:
        # Passthrough for provider-specific params (e.g. vLLM chat_template_kwargs
        # to toggle a thinking model). Callers own compatibility.
        body.update(extra_body)

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
        "Accept": "text/event-stream",
    }

    # A long agent turn (a big multimodal prompt + a multi-minute reasoning pass +
    # a long answer) can stream healthily for well over any fixed budget, so a
    # `total` timeout would abort a perfectly live request mid-stream and surface as
    # a spurious `llm_error`. Use an *idle* timeout instead: `sock_read` bounds the
    # gap between chunks (and prefill before the first byte) and resets on every
    # read, so we only give up when the provider actually goes silent. `total=None`
    # lifts the wall-clock cap; the run is already bounded by max_turns and the
    # session lease.
    client_timeout = aiohttp.ClientTimeout(
        total=None, sock_connect=30, sock_read=timeout,
    )
    logger.info(f"Making LLM call to Agent - {model}")
    for attempt in range(len(_RETRY_DELAYS) + 1):
        streamed = False
        try:
            async with aiohttp.ClientSession(timeout=client_timeout) as session:
                async with session.post(endpoint, headers=headers, json=body) as resp:
                    if resp.status != 200:
                        err = await resp.text()
                        msg = f"HTTP {resp.status} from {endpoint}: {err[:500]}"
                        if resp.status in _RETRY_STATUS:
                            raise _RetryableStatus(msg)
                        raise RuntimeError(msg)
                    async for raw in resp.content:
                        if not raw:
                            continue
                        line = raw.decode("utf-8", errors="replace").rstrip("\r\n")
                        if not line or line.startswith(":"):
                            continue
                        if not line.startswith("data:"):
                            continue
                        data = line[5:].strip()
                        if data == "[DONE]":
                            return
                        try:
                            chunk = json.loads(data)
                        except json.JSONDecodeError:
                            continue
                        # Some providers (vLLM/SGLang gateways, OpenRouter mid-stream) report
                        # a rejected request as HTTP 200 + an `error` chunk. Raise it like a
                        # non-200 so the run fails visibly instead of ending empty.
                        if isinstance(chunk, dict) and chunk.get("error") and not chunk.get("choices"):
                            raise RuntimeError(f"stream error from {endpoint}: {json.dumps(chunk['error'])[:500]}")
                        streamed = True
                        yield chunk
            return
        except (_RetryableStatus, aiohttp.ClientConnectionError, asyncio.TimeoutError) as exc:
            if streamed or attempt == len(_RETRY_DELAYS):
                if isinstance(exc, _RetryableStatus):
                    raise RuntimeError(str(exc)) from None
                raise
            delay = _RETRY_DELAYS[attempt]
            logger.warning("LLM call to %s failed before streaming (%s: %s); retry %d in %.0fs",
                           model, type(exc).__name__, str(exc)[:200], attempt + 1, delay)
            await asyncio.sleep(delay)


def assemble_assistant_message(
    chunks: list[dict],
) -> tuple[dict, Optional[str], list[dict], str, Optional[dict]]:
    """Fold streamed chunks into an Anthropic-style assistant message.

    Returns (message, finish_reason, tool_uses, reasoning, usage). `message`
    follows the shape `run_agent` already builds: role=assistant,
    content=[{type:text,...} | {type:tool_use,...}]. `tool_uses` is the list of
    tool_use blocks (already inside message.content too) for convenient dispatch
    by the runner. `reasoning` is the raw extended-thinking text (empty if none).
    `usage` is the raw provider usage block from the final chunk (or None).
    """
    text_buf: list[str] = []
    reasoning_buf: list[str] = []
    tool_calls: dict[int, dict] = {}
    finish_reason: Optional[str] = None
    usage: Optional[dict] = None

    for chunk in chunks:
        # The final stream chunk (stream_options.include_usage) carries usage and
        # usually has an empty choices list.
        if isinstance(chunk.get("usage"), dict):
            usage = chunk["usage"]
        for choice in chunk.get("choices") or []:
            delta = choice.get("delta") or {}
            if isinstance(delta.get("content"), str):
                text_buf.append(delta["content"])
            elif isinstance(delta.get("content"), list):
                for part in delta["content"]:
                    if isinstance(part, dict) and part.get("type") == "text":
                        text_buf.append(part.get("text", ""))
            if isinstance(delta.get("reasoning"), str):
                reasoning_buf.append(delta["reasoning"])
            for tc in delta.get("tool_calls") or []:
                idx = tc.get("index", 0)
                slot = tool_calls.setdefault(idx, {"id": None, "name": None, "arguments": ""})
                if tc.get("id"):
                    slot["id"] = tc["id"]
                fn = tc.get("function") or {}
                if fn.get("name"):
                    slot["name"] = fn["name"]
                if isinstance(fn.get("arguments"), str):
                    slot["arguments"] += fn["arguments"]
            if choice.get("finish_reason"):
                finish_reason = choice["finish_reason"]

    content: list[dict] = []
    text = "".join(text_buf).strip()
    if text:
        content.append({"type": "text", "text": text})
    reasoning = "".join(reasoning_buf).strip()
    if reasoning:
        # Persist reasoning as a text block prefixed for downstream inspection.
        content.append({"type": "text", "text": f"<think>\n{reasoning}\n</think>"})

    tool_uses: list[dict] = []
    for idx in sorted(tool_calls.keys()):
        slot = tool_calls[idx]
        if not slot["id"] or not slot["name"]:
            continue
        try:
            tool_input = json.loads(slot["arguments"] or "{}")
        except json.JSONDecodeError:
            tool_input = {"_raw_arguments": slot["arguments"]}
        block = {
            "type": "tool_use",
            "id": slot["id"],
            "name": slot["name"],
            "input": tool_input,
        }
        content.append(block)
        tool_uses.append(block)

    message = {"role": "assistant", "content": content}
    return message, finish_reason, tool_uses, reasoning, usage

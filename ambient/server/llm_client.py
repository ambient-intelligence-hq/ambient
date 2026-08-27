"""Streaming OpenAI chat-completions client.

Yields parsed chunk dicts. The HTTP route relays each as
`event: chat.completion.chunk`.
"""
from __future__ import annotations

import json
from typing import AsyncIterator, Optional

import aiohttp

from ambient.config import settings
from ambient.llm import get_provider_params


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
        # the actual USD cost). Sending both is harmless and maximizes coverage.
        "stream_options": {"include_usage": True},
        "usage": {"include": True},
    }
    # Only include tools/tool_choice when there actually are tools — some servers
    # (vLLM) reject an empty `tools` array. Fast mode passes tools=[].
    if tools:
        body["tools"] = tools
        body["tool_choice"] = "auto"
    if reasoning_enabled:
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

    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=timeout)) as session:
        async with session.post(endpoint, headers=headers, json=body) as resp:
            if resp.status != 200:
                err = await resp.text()
                raise RuntimeError(f"HTTP {resp.status} from {endpoint}: {err[:500]}")
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
                yield chunk


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

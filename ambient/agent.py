import asyncio
import json
import os
import sys
import urllib.error
import urllib.request
import uuid
from typing import Optional

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ambient.config import settings
from ambient.llm import get_provider_params , video_to_data_url
from ambient.prompt import SYSTEM_PROMPT
from ambient.tools import TOOL_REGISTRY, TOOLS
from ambient.tools import video_description as _video_description
import inspect
from functools import partial
import tenacity

"""
TODO:
X Important: Normalize the citation range to actual clip timestamps.
- Structured response validation and retry with forced structured response in the request.
X Better handle in memory video description reuse for tool calls - focus_clip and search_clip tools.
- Usage token and metrics of sessions.
- Fix the deviation in video description for different runs.
- Enable Citing and sharing few frames for verification (optionally) to agent llm
- Test need for search-retrieval tools
- Audio in the clip
- Utility to retrieve subtitle associated with clip - in search_clip focus_clip and search_video tools
- Parity with transcribe audio if it works with subtitles
"""

MAX_CLIPS = 5
MAX_FRAMES = 30
DEFAULT_TIMEOUT = 120
DEFAULT_MAX_TOKENS = 16000

VIDEO_DESCRIPTION = {}


# ---------------------------------------------------------------------------
# Tool dispatch
# ---------------------------------------------------------------------------

async def execute_tool(tool_name: str, tool_input: dict) -> dict:
    func = TOOL_REGISTRY.get(tool_name)
    if func is None:
        raise ValueError(f"Unknown tool: {tool_name}")
    if tool_name == "get_video_description":
        video_description, user_message_contents = await func(**tool_input)
        video_id = tool_input.get("video_id")
        if video_id and video_description:
            VIDEO_DESCRIPTION[video_id] = video_description
        return video_description, user_message_contents
    else:
        # get arguments of the function
        args = inspect.getfullargspec(func).args
        video_id = tool_input.get("video_id")
        if "video_description" in args and video_id and VIDEO_DESCRIPTION.get(video_id):
            tool_input["video_description"] = VIDEO_DESCRIPTION.get(video_id)
    return await func(**tool_input)


# ---------------------------------------------------------------------------
# Message shaping
# ---------------------------------------------------------------------------

def _retain_last_n_by_type(chat_messages: list[dict], content_type: str, n: int) -> list[dict]:
    """Walk messages newest-first and drop content blocks of `content_type` past the Nth."""
    new_messages: list[dict] = []
    seen = 0
    for message in chat_messages[::-1]:
        if message["role"] == "user" and isinstance(message["content"], list):
            kept = []
            for content in message["content"]:
                if content.get("type") == content_type:
                    seen += 1
                    if seen >= n:
                        continue
                kept.append(content)
            message["content"] = kept
            if kept:
                new_messages.append(message)
        else:
            new_messages.append(message)
    return new_messages[::-1]


def _normalize_messages_for_chat(raw_messages: list[dict]) -> list[dict]:
    """Convert Anthropic-style assistant/tool_result blocks into OpenAI chat-completions shape."""
    chat_messages: list[dict] = []
    for message in raw_messages:
        role = message.get("role")
        content = message.get("content")

        # Anthropic-style assistant tool_use blocks -> OpenAI tool_calls
        if role == "assistant" and isinstance(content, list):
            text_parts: list[str] = []
            tool_calls: list[dict] = []
            for block in content:
                if not isinstance(block, dict):
                    continue
                block_type = block.get("type")
                if block_type == "text":
                    text_parts.append(block.get("text", "") or block.get("content", ""))
                elif block_type == "tool_use":
                    tool_calls.append({
                        "id": block.get("id"),
                        "type": "function",
                        "function": {
                            "name": block.get("name"),
                            "arguments": json.dumps(block.get("input", {})),
                        },
                    })

            assistant_message: dict = {
                "role": "assistant",
                "content": "\n".join(p for p in text_parts if p).strip(),
            }
            if tool_calls:
                assistant_message["tool_calls"] = tool_calls
            chat_messages.append(assistant_message)
            continue

        if role == "user" and isinstance(content, list):
            # Anthropic-style tool_result user blocks -> OpenAI tool messages
            tool_results = [
                block for block in content
                if isinstance(block, dict) and block.get("type") == "tool_result"
            ]
            if tool_results and len(tool_results) == len(content):
                for tool_result in tool_results:
                    tool_content = tool_result.get("content", "")
                    if isinstance(tool_content, (dict, list)):
                        tool_content = json.dumps(tool_content)
                    chat_messages.append({
                        "role": "tool",
                        "tool_call_id": tool_result.get("tool_use_id"),
                        "content": str(tool_content),
                    })
                continue

            # Plain text blocks -> single text content
            only_text_blocks = all(
                isinstance(block, dict) and block.get("type") == "text"
                for block in content
            )
            if only_text_blocks:
                merged_text = "\n".join(
                    (block.get("text", "") or block.get("content", ""))
                    for block in content
                ).strip()
                chat_messages.append({"role": "user", "content": merged_text})
                continue

        chat_messages.append(message)

    return chat_messages


def _normalize_response_payload(payload: dict) -> dict:
    """Normalize an OpenAI chat-completions response into the anthropic-like shape used by run_agent."""
    choices = payload.get("choices")
    if not (choices and isinstance(choices, list)):
        return payload

    message = (choices[0] or {}).get("message", {}) or {}
    content_blocks: list[dict] = []

    message_content = message.get("content")
    if isinstance(message_content, str) and message_content:
        content_blocks.append({
            "type": "text",
            "text": message_content,
            "content": message_content,
        })
    elif isinstance(message_content, list):
        for item in message_content:
            if (
                isinstance(item, dict)
                and item.get("type") == "text"
                and item.get("text")
            ):
                content_blocks.append({
                    "type": "text",
                    "text": item["text"],
                    "content": item["text"],
                })

    for tool_call in message.get("tool_calls", []) or []:
        function = tool_call.get("function", {}) or {}
        raw_arguments = function.get("arguments", "{}")
        if isinstance(raw_arguments, str):
            try:
                tool_input = json.loads(raw_arguments)
            except json.JSONDecodeError:
                tool_input = {"_raw_arguments": raw_arguments}
        else:
            tool_input = raw_arguments or {}

        content_blocks.append({
            "type": "tool_use",
            "id": tool_call.get("id"),
            "name": function.get("name"),
            "input": tool_input,
            "content": tool_input,
        })

    stop_reason = "tool_use" if message.get("tool_calls") else "end_turn"
    return {"content": content_blocks, "stop_reason": stop_reason, "raw": payload}


# ---------------------------------------------------------------------------
# HTTP
# ---------------------------------------------------------------------------

async def _send_request(
    url: str,
    api_key: str,
    model: str,
    messages: list[dict],
    use_effort: bool = False,
    timeout: int = DEFAULT_TIMEOUT,
    max_tokens: int = DEFAULT_MAX_TOKENS,
) -> dict:
    endpoint = url.rstrip("/")
    provider_params = get_provider_params(model, endpoint)
    if not endpoint.endswith("/chat/completions"):
        endpoint = f"{endpoint}/chat/completions"

    chat_messages = _normalize_messages_for_chat(messages)
    chat_messages = _retain_last_n_by_type(chat_messages, "video_url", MAX_CLIPS)
    chat_messages = _retain_last_n_by_type(chat_messages, "image_url", MAX_FRAMES)

    body = {
        "model": model,
        "max_tokens": max_tokens,
        "reasoning": {"enabled": True},
        "tools": TOOLS,
        "tool_choice": "auto",
        "messages": chat_messages,
    }
    if use_effort:
        body["reasoning"]["effort"] = "medium"
    body.update(provider_params)

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }
    req = urllib.request.Request(endpoint, data=json.dumps(body).encode(), headers=headers)

    @tenacity.retry(
        stop=tenacity.stop_after_attempt(3),
        wait=tenacity.wait_exponential(multiplier=1, min=4, max=10),
        reraise=True,
    )
    def _do_request() -> dict:
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                raw_body = resp.read().decode("utf-8")
                if resp.status != 200:
                    raise RuntimeError(f"HTTP {resp.status}: {raw_body}")
                try:
                    payload = json.loads(raw_body)
                except json.JSONDecodeError as e:
                    raise RuntimeError(
                        f"Invalid JSON response from {endpoint}: {raw_body[:500]}"
                    ) from e
                return _normalize_response_payload(payload)
        except urllib.error.HTTPError as e:
            error_body = e.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"HTTP {e.code}: {error_body}") from e
        except urllib.error.URLError as e:
            raise RuntimeError(f"Request failed: {e.reason}") from e

    return await asyncio.to_thread(_do_request)


# ---------------------------------------------------------------------------
# Agent loop
# ---------------------------------------------------------------------------

def _trajectory_filename(video_id: str, question: str) -> str:
    questions_gist = "_".join(question.split()[:5]).lower()
    return f"trajectory_{video_id}_{questions_gist}_{uuid.uuid4().hex[:8]}.json"


def _extract_reasoning(resp: dict) -> Optional[str]:
    try:
        return resp.get("raw", {}).get("choices", [{}])[0].get("message", {}).get("reasoning")
    except (IndexError, AttributeError):
        return None


async def _run_tool_calls(content: list[dict]) -> list[tuple]:
    """Execute every tool_use block in `content` concurrently. Returns [(tool_use_id, result), ...]."""
    tasks = {
        block["id"]: asyncio.create_task(execute_tool(block["name"], block["input"]))
        for block in content
        if block["type"] == "tool_use"
    }
    if not tasks:
        return []
    results = await asyncio.gather(*tasks.values())
    return list(zip(tasks.keys(), results))


async def run_agent(
    video_id: str,
    question: str,
    max_turns: int = 5,
    model: str = settings.agent_model,
    subtitle_path: Optional[str] = None,
    output_structure: Optional[dict] = None,
) -> list[dict]:
    if subtitle_path is not None:
        print(f"Adding Subtitle path: {subtitle_path}")
        with open(subtitle_path, "r") as f:
            _video_description._TRANSCRIPT = f.read()

    messages: list[dict] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": f"Video id: {video_id}, question to answer: {question}, Your final answer should strictly follow the schema: {output_structure.model_json_schema()}"}
            ],
        },
    ]

    trajectory_dir = os.path.join(os.path.dirname(__file__), "trajectories")
    os.makedirs(trajectory_dir, exist_ok=True)
    trajectory_file = os.path.join(trajectory_dir, _trajectory_filename(video_id, question))
    trajectory: dict = {
        "model": model,
        "question": question,
        "video_id": video_id,
        "turns": [],
    }

    print("--------------------------------")
    print(f"Model: {model}")
    print(f"video_id: {video_id}")
    print(f"question: {question}")
    print(f"trajectory_file: {trajectory_file}")
    print("--------------------------------")

    try:
        for turn in range(1, max_turns + 1):
            print(f"Turn {turn}")
            turn_trajectory: dict = {"turn": turn, "messages": list(messages)}

            resp = await _send_request(
                settings.llm_base_url, settings.llm_api_key, model, messages
            )
            if "content" not in resp:
                print(f"No content in response: {resp}")
                continue
            content = resp["content"]
            stop_reason = resp["stop_reason"]
            reasoning = _extract_reasoning(resp)
            turn_trajectory["reasoning"] = reasoning

            print(" ---------------- Assistant Response ----------------")
            print(f"Reasoning: {reasoning}")
            print(f"    turn {turn} ({stop_reason})")
            for block in content:
                turn_trajectory[block["type"]] = block["content"]
                if block["type"] == "tool_use":
                    print(f"    Tool Use: {block['name']} - {block['id']}")
                print(f"    {block['type']}:: {block['content']}")
                print("    --------------------------------")

            if stop_reason == "end_turn":
                trajectory["turns"].append(turn_trajectory)
                break

            if stop_reason == "tool_use":
                if reasoning:
                    content.append({
                        "type": "text",
                        "text": f"<think>\n{reasoning}\n</think>",
                    })
                messages.append({"role": "assistant", "content": content})

                print(" ---------------- Tool Results ----------------")
                tool_results_trajectory = []
                tool_results = await _run_tool_calls(content)
                print(f"Tool Results: {tool_results}")
                for tool_call_id, (analysis, user_message_contents) in tool_results:
                    tool_result_block = {
                        "role": "user",
                        "content": [{
                            "type": "tool_result",
                            "tool_use_id": tool_call_id,
                            "content": analysis,
                        }],
                    }
                    messages.append(tool_result_block)
                    tool_results_trajectory.append(tool_result_block)
                    if user_message_contents:
                        messages.append({"role": "user", "content": user_message_contents})

                    print(f"    Tool Result: {analysis}")
                    print(f"    Tool Call ID: {tool_call_id}")
                    print(f"    Number of User Message Contents: {len(user_message_contents)}")
                    print("    --------------------------------")
                turn_trajectory["tool_results"] = tool_results_trajectory

            trajectory["turns"].append(turn_trajectory)
    finally:
        with open(trajectory_file, "w") as f:
            json.dump(trajectory, f, indent=4, default=str)
        print(f"Trajectory saved to {trajectory_file}")

    return messages


if __name__ == "__main__":
    import dotenv

    from pydantic import BaseModel, Field

    class response_structure(BaseModel):
        time_taken_police_car: float = Field(description="The time taken for the police car to arrive at the scene after the accident in seconds.")
        time_taken_tow_vehicle: float = Field(description="The time taken for the tow vehicle to arrive at the scene after the accident in seconds.")
        description: str = Field(description="The description of the event.")
        citations: list[dict] = Field(description="Clip timstamp of the event (with start and end time)")

    output_structure = response_structure

    dotenv.load_dotenv()
    question = "What is the time taken for the police car to arrive at the scene after the accident? and when did the tow vechile arrive after the accident?"
    asyncio.run(
        run_agent(
            video_id="Seattle_bad_driver_accident",
            question=question,
            max_turns=20,
            # subtitle_path="",
            output_structure=output_structure,
        )
    )

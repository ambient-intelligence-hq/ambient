"""Ambient quickstart — drive the video agent with the official Anthropic SDK.

Ambient speaks the Anthropic **Managed Agents** protocol, so the stock SDK
(`client.beta.*`) talks to it directly — no custom client. The whole flow is:

    upload a video  ->  create a session over it  ->  ask  ->  stream the run

Run it:

    # 1. start the infra + server (see the repo README)
    docker compose up -d postgres redis
    uv run python -m ambient.server            # http://127.0.0.1:8080

    # 2. ask a question about a local clip (agent mode = tool loop, the default)
    uv run python examples/quickstart.py --video path/to/clip.mp4 "What happens in this video? give a quick summary"

    # ...fast mode: one dense-frame vision pass, no tools — much faster, good for
    #    whole-video summaries/Q&A that don't need frame-accurate grounding:
    uv run python examples/quickstart.py --mode fast --video path/to/clip.mp4 "Summarize this video."

    # ...or reuse an already-uploaded video by id (skip the upload):
    uv run python examples/quickstart.py --video-id <video_id> "Summarize it."

Config via env (sensible local defaults): AMBIENT_BASE_URL, AMBIENT_API_KEY.
"""
from __future__ import annotations

import argparse
import os
import re
import sys
import time

from anthropic import Anthropic

BASE_URL = os.environ.get("AMBIENT_BASE_URL", "http://127.0.0.1:8080")
API_KEY = os.environ.get("AMBIENT_API_KEY", "dev-token")

# The server auto-seeds these agents at startup, so you never need agents.create /
# environments.create — just reference them by id:
#   "ambient_v1"   = the agentic tool loop (search/focus clips, grab frames, ...)
#   "ambient_fast" = a single dense-frame vision pass, no tools
AGENT_BY_MODE = {"agent": "ambient_v1", "fast": "ambient_fast"}
ENV_ID = "default"


def main() -> None:
    ap = argparse.ArgumentParser(description="Ask the Ambient video agent a question.")
    ap.add_argument("question", help="the question to ask about the video")
    ap.add_argument(
        "--mode", choices=("agent", "fast"), default="agent",
        help="agent = tool loop (default); fast = one dense-frame vision pass, no tools",
    )
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--video", help="path to a local video file to upload")
    src.add_argument("--video-id", help="reuse an already-uploaded video by id")
    args = ap.parse_args()

    agent_id = AGENT_BY_MODE[args.mode]

    client = Anthropic(base_url=BASE_URL, api_key=API_KEY)

    # 1. Resolve the video id — upload a local file, or reuse an existing one.
    #    Uploading kicks off background description ingestion server-side; the
    #    session below never blocks on it.
    if args.video_id:
        video_id = args.video_id
        print(f"using existing video: {video_id}")
    else:
        print(f"uploading {args.video} ...")
        with open(args.video, "rb") as fh:
            video = client.beta.files.upload(file=fh)
        video_id = video.id
        print(f"uploaded -> {video_id}")

    # 2. Create a session over the video. The video lives in metadata.video_id;
    #    agent + environment are required by the SDK (use the seeded defaults).
    session = client.beta.sessions.create(
        agent=agent_id,
        environment_id=ENV_ID,
        metadata={"video_id": video_id},
        title=f"quickstart ({args.mode})",
    )
    print(f"session: {session.id} (mode={args.mode}, agent={agent_id})")

    # 3. Wait (bounded) for the per-session sandbox to boot -> status "idle".
    for _ in range(90):
        s = client.beta.sessions.retrieve(session.id)
        if s.status in ("idle", "running"):
            break
        if s.status == "terminated":
            sys.exit("session terminated before it was ready")
        time.sleep(1)
    print(f"ready: {s.status}\n")

    # 4. Send the question, then stream the run. The agent runs its tool loop
    #    server-side; we just render the events as they arrive.
    answer = ask(client, session.id, args.question)
    print("\n=== ANSWER ===\n" + answer)

    # Follow-ups reuse the same session.id — the conversation and everything the
    # agent already gathered (clips, frames) are preserved server-side.
    #   ask(client, session.id, "Now give me timestamps for each key moment.")


def ask(client: Anthropic, session_id: str, question: str) -> str:
    """Send one user message and stream the resulting run to stdout."""
    client.beta.sessions.events.send(
        session_id, events=[{"type": "user.message", "content": question}]
    )
    answer = ""
    turn = 0
    with client.beta.sessions.events.stream(session_id) as stream:
        for ev in stream:
            if ev.type == "agent.thinking":
                # The route attaches the model's full reasoning text as `content`
                # (an extra field on the SDK's progress-only thinking event).
                turn += 1
                reasoning = _text(ev)
                print(f"\n── turn {turn} ──")
                if reasoning:
                    print(_block("thinking", reasoning))
            elif ev.type == "agent.tool_use":
                print(f"  → {ev.name}({_compact(getattr(ev, 'input', {}))})")
            elif ev.type == "agent.tool_result":
                flag = " [error]" if getattr(ev, "is_error", False) else ""
                print(_block(f"result{flag}", _text(ev), limit=400))
            elif ev.type == "span.model_request_end":
                u = ev.model_usage
                print(f"    · tokens in={u.input_tokens} out={u.output_tokens} "
                      f"(cache_read={u.cache_read_input_tokens})")
            elif ev.type == "agent.message":
                answer = _text(ev)
            elif ev.type == "session.status_idle":
                print(f"\n  done: {ev.stop_reason}")
            elif ev.type == "session.error":
                print(f"  error: {ev.error}")
    return answer


def _text(ev) -> str:
    """Concatenate the text of an event's content blocks (thinking/result/message)."""
    return "".join(getattr(b, "text", "") for b in (getattr(ev, "content", None) or [])).strip()


def _compact(obj) -> str:
    """One-line, trimmed repr of a tool-call input dict."""
    parts = [f"{k}={v!r}" for k, v in (obj or {}).items() if k != "video_id"]
    return ", ".join(parts)


def _block(label: str, text: str, limit: int = 2000, width: int = 100) -> str:
    """Indent a possibly-multi-line block under a label, wrapped and length-capped."""
    import textwrap

    text = re.sub(r"\s+", " ", text).strip()
    if len(text) > limit:
        text = text[:limit].rstrip() + " …"
    wrapped = textwrap.fill(
        text, width=width, initial_indent="    ", subsequent_indent="    "
    )
    return f"  {label}:\n{wrapped}" if wrapped else f"  {label}:"


if __name__ == "__main__":
    main()

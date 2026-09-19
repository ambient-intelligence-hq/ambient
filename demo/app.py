"""Gradio demo for the Ambient video-agent service.

Drives the managed-agents HTTP API through the Anthropic SDK, exactly like
notebooks/test_sdk.ipynb, but wrapped in a UI that showcases the full loop:

    upload a video  ->  start a session (boots the sandbox)
                    ->  ask a question  ->  watch agent + tool events stream
                    ->  ask follow-ups on the same session

Tool calls and tool results render as collapsible cards in the chat; the final
answer renders as a normal assistant message.

Run locally (server must be up — `docker compose up -d` or `python -m ambient.server`):
    uv run python demo/app.py
Then open http://127.0.0.1:7860.

For Hugging Face Spaces, copy the contents of this `demo/` folder to the Space
root and set the secrets AMBIENT_BASE_URL / AMBIENT_API_KEY (see README.md).
"""
from __future__ import annotations

import io
import json
import mimetypes
import os
import time
from typing import Any
from urllib.parse import parse_qs, urlparse
from dotenv import load_dotenv
import gradio as gr
import httpx
from anthropic import Anthropic

load_dotenv()

DEFAULT_BASE_URL = os.getenv("AMBIENT_BASE_URL", "http://127.0.0.1:8080")
DEFAULT_API_KEY = os.getenv("AMBIENT_API_KEY", "dev-token")
DEFAULT_MODEL = os.getenv("AGENT_MODEL", "openai/gpt-5.6-sol")
DEFAULT_SYSTEM = "Analyze videos and answer questions about them with citations."

SANDBOX_BOOT_TIMEOUT_S = 90
# A YouTube source is downloaded + remuxed inside the ingest sandbox before the
# session can become ready, so it needs a far more generous readiness window than
# a direct upload (the file is already in R2 for those).
YOUTUBE_BOOT_TIMEOUT_S = 600


def _client(base_url: str, api_key: str) -> Anthropic:
    return Anthropic(base_url=(base_url or DEFAULT_BASE_URL).strip(),
                     api_key=(api_key or DEFAULT_API_KEY).strip())


def _youtube_video_id(url: str) -> str | None:
    """Extract the 11-char video id from common YouTube URL shapes."""
    parsed = urlparse((url or "").strip())
    host = (parsed.hostname or "").lower().removeprefix("www.").removeprefix("m.")
    if host == "youtu.be":
        vid = parsed.path.lstrip("/").split("/")[0]
        return vid or None
    if host in ("youtube.com", "youtube-nocookie.com"):
        if parsed.path == "/watch":
            vals = parse_qs(parsed.query).get("v")
            return vals[0] if vals else None
        for prefix in ("/embed/", "/shorts/", "/v/"):
            if parsed.path.startswith(prefix):
                return parsed.path[len(prefix):].split("/")[0] or None
    return None


def _youtube_preview(url: str) -> Any:
    """Render an embedded YouTube player for a pasted URL (empty when not one)."""
    vid = _youtube_video_id(url)
    if not vid:
        return gr.update(value="", visible=False)
    iframe = (
        '<div style="position:relative;padding-bottom:56.25%;height:0;overflow:hidden;border-radius:8px">'
        f'<iframe src="https://www.youtube.com/embed/{vid}" '
        'style="position:absolute;top:0;left:0;width:100%;height:100%;border:0" '
        'allow="accelerometer;autoplay;clipboard-write;encrypted-media;gyroscope;picture-in-picture" '
        "allowfullscreen></iframe></div>"
    )
    return gr.update(value=iframe, visible=True)


def _import_youtube(base_url: str, api_key: str, url: str) -> str:
    """Submit a YouTube URL to the server's import endpoint; return the video id.

    The Anthropic SDK has no method for this custom route, so we POST directly to
    `/v1/files/import` with the same `x-api-key` auth the SDK uses.
    """
    endpoint = (base_url or DEFAULT_BASE_URL).strip().rstrip("/") + "/v1/files/import"
    resp = httpx.post(
        endpoint,
        headers={"x-api-key": (api_key or DEFAULT_API_KEY).strip()},
        json={"source_type": "youtube", "url": url.strip()},
        timeout=30,
    )
    if resp.status_code >= 400:
        detail = resp.text
        try:
            detail = resp.json().get("error", {}).get("message") or detail
        except Exception:
            pass
        raise gr.Error(f"YouTube import failed ({resp.status_code}): {detail}")
    return resp.json()["id"]


def _event_text(content: Any) -> str:
    """Join the text blocks of a streamed event's content."""
    if isinstance(content, str):
        return content
    return "".join(getattr(b, "text", "") or "" for b in (content or []))


def start_session(video_path, youtube_url, base_url, api_key, model, system,
                  progress=gr.Progress()):
    """Mount a video (upload or YouTube URL), create the session, wait for idle."""
    youtube_url = (youtube_url or "").strip()
    if not youtube_url and not video_path:
        raise gr.Error("Upload a video or paste a YouTube URL first.")
    client = _client(base_url, api_key)

    # A YouTube URL takes precedence: the server downloads it inside the sandbox.
    if youtube_url:
        progress(0.1, desc="Importing YouTube video…")
        video_id = _import_youtube(base_url, api_key, youtube_url)
        boot_timeout = YOUTUBE_BOOT_TIMEOUT_S
    else:
        progress(0.1, desc="Uploading video…")
        filename = os.path.basename(video_path)
        mime = mimetypes.guess_type(filename)[0] or "video/mp4"
        with open(video_path, "rb") as fh:
            data = fh.read()
        meta = client.beta.files.upload(file=(filename, io.BytesIO(data), mime))
        video_id = meta.id
        boot_timeout = SANDBOX_BOOT_TIMEOUT_S

    progress(0.4, desc="Creating agent + session…")
    agent = client.beta.agents.create(
        name="Video Analyst",
        model=(model or DEFAULT_MODEL).strip(),
        system=(system or DEFAULT_SYSTEM).strip(),
        tools=[{"type": "agent_toolset_20260401"}],
    )
    env = client.beta.environments.create(name="ambient-demo")
    session = client.beta.sessions.create(
        agent=agent.id,
        environment_id=env.id,
        metadata={"video_id": video_id},
        title="Ambient demo",
    )

    boot_desc = (
        "Downloading & analyzing YouTube video (can take a few minutes)…"
        if youtube_url else "Booting sandbox…"
    )
    progress(0.6, desc=boot_desc)
    deadline = time.time() + boot_timeout
    status = "pending"
    while time.time() < deadline:
        status = client.beta.sessions.retrieve(session.id).status
        if status == "idle":
            break
        if status == "terminated":
            raise gr.Error("Session terminated before it became ready.")
        time.sleep(1)
    if status != "idle":
        raise gr.Error("Sandbox did not become ready in time — try again.")

    state = {
        "base_url": base_url,
        "api_key": api_key,
        "session_id": session.id,
        "video_id": video_id,
    }
    banner = (
        f"✅ **Session ready** — `{session.id}`\n\n"
        f"Video `{video_id}` mounted. Ask a question below; follow-ups reuse this session."
    )
    return (
        state,
        banner,
        [],  # reset chat
        gr.update(interactive=True, placeholder="Ask about the video…"),
        gr.update(interactive=True),
    )


def respond(message, history, state):
    """Send a user message and stream the agent + tool events into the chat."""
    if not state or not state.get("session_id"):
        raise gr.Error("Start a session first (upload a video, then 'Start session').")
    message = (message or "").strip()
    if not message:
        return

    client = _client(state["base_url"], state["api_key"])
    sid = state["session_id"]

    history = list(history) + [{"role": "user", "content": message}]
    yield history, gr.update(value="", interactive=False)

    # A "thinking" placeholder while the first run event arrives.
    pending = {"role": "assistant", "content": "_working…_",
               "metadata": {"title": "⏳ Running"}}
    history.append(pending)
    yield history, gr.update()

    client.beta.sessions.events.send(
        sid, events=[{"type": "user.message", "content": message}]
    )

    answered = False
    try:
        with client.beta.sessions.events.stream(sid) as stream:
            for ev in stream:
                if ev.type == "agent.tool_use":
                    tool_input = getattr(ev, "input", None) or {}
                    history.append({
                        "role": "assistant",
                        "content": "```json\n" + json.dumps(tool_input, indent=2) + "\n```",
                        "metadata": {"title": f"🔧 Tool call · {getattr(ev, 'name', 'tool')}"},
                    })
                    yield history, gr.update()
                elif ev.type == "agent.tool_result":
                    text = _event_text(getattr(ev, "content", None))
                    is_err = getattr(ev, "is_error", False)
                    title = "⚠️ Tool error" if is_err else "📄 Tool result"
                    history.append({
                        "role": "assistant",
                        "content": text or "_(no output)_",
                        "metadata": {"title": title},
                    })
                    yield history, gr.update()
                elif ev.type == "agent.message":
                    answer = _event_text(getattr(ev, "content", None))
                    if answer:
                        history.append({"role": "assistant", "content": answer})
                        answered = True
                        yield history, gr.update()
                elif ev.type == "session.status_idle":
                    break
    except Exception as exc:  # surface stream/transport errors in the chat
        history.append({"role": "assistant",
                        "content": f"❌ {type(exc).__name__}: {exc}",
                        "metadata": {"title": "Error"}})

    # Drop the "Running" placeholder.
    if pending in history:
        history.remove(pending)
    if not answered and (not history or history[-1]["role"] != "assistant"):
        history.append({"role": "assistant",
                        "content": "_(the run produced no final answer)_"})
    yield history, gr.update(interactive=True)


with gr.Blocks(title="Ambient Video Agent") as demo:
    gr.Markdown(
        "# 🎬 Ambient Video Agent\n"
        "Upload a video, start a session, and chat with the agent. Tool calls and "
        "results stream in as collapsible cards."
    )
    state = gr.State(None)

    with gr.Row():
        with gr.Column(scale=2):
            video = gr.Video(label="Video", sources=["upload"], height=300)
            youtube_url = gr.Textbox(
                label="…or paste a YouTube URL",
                placeholder="https://www.youtube.com/watch?v=…",
            )
            youtube_preview = gr.HTML(visible=False)
            with gr.Accordion("Connection & agent settings", open=False):
                base_url = gr.Textbox(label="Server base URL", value=DEFAULT_BASE_URL)
                api_key = gr.Textbox(label="API key", value=DEFAULT_API_KEY, type="password")
                model = gr.Textbox(label="Agent model", value=DEFAULT_MODEL)
                system = gr.Textbox(label="System prompt", value=DEFAULT_SYSTEM, lines=2)
            start_btn = gr.Button("Start session", variant="primary")
            session_status = gr.Markdown("_No session yet — upload a video and start one._")

        with gr.Column(scale=3):
            # Gradio 6 standardized on the messages format; assistant messages
            # carrying `metadata={"title": ...}` render as collapsible cards,
            # which is how tool calls/results are shown.
            chatbot = gr.Chatbot(label="Agent session", height=520)
            with gr.Row():
                msg = gr.Textbox(show_label=False, scale=8, interactive=False,
                                 placeholder="Start a session first…")
                send_btn = gr.Button("Send", scale=1, variant="primary", interactive=False)

    youtube_url.change(
        _youtube_preview, inputs=[youtube_url], outputs=[youtube_preview]
    )
    start_btn.click(
        start_session,
        inputs=[video, youtube_url, base_url, api_key, model, system],
        outputs=[state, session_status, chatbot, msg, send_btn],
    )
    for trigger in (msg.submit, send_btn.click):
        trigger(respond, inputs=[msg, chatbot, state], outputs=[chatbot, msg])


if __name__ == "__main__":
    demo.queue().launch(theme=gr.themes.Soft())

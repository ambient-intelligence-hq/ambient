"""MCP server that exposes the Ambient video agent as a Claude Code tool.

Claude Code (or any MCP client) connects to this server over stdio and calls
`analyze_video(...)`. The tool drives a managed-agents session against the
Ambient HTTP API and streams the agent's progress — each tool call and tool
result — back as MCP progress/log notifications while the run is in flight, then
returns the final answer plus a transcript of what the agent did.

Why a tool (not a Claude Code "subagent"): a subagent is itself a Claude
instance and can't call an external HTTP service directly. An MCP tool is the
mechanism that lets Claude Code invoke your API. (You can still wrap this tool in
a subagent for ergonomics — see .claude/agents/video-analyst.md.)

Config (env):
    AMBIENT_BASE_URL   default http://127.0.0.1:8080
    AMBIENT_API_KEY    default dev-token
    AMBIENT_AGENT_MODEL default z-ai/glm-5.2

Run standalone (stdio):
    uv run python integrations/mcp_video_agent.py

Register with Claude Code (project-scoped via .mcp.json is already set up; or):
    claude mcp add ambient-video -- uv run python integrations/mcp_video_agent.py
"""
from __future__ import annotations

import asyncio
import json
import mimetypes
import os
from typing import Optional

import httpx
from mcp.server.fastmcp import Context, FastMCP

BASE_URL = os.getenv("AMBIENT_BASE_URL", "http://127.0.0.1:8080").rstrip("/")
API_KEY = os.getenv("AMBIENT_API_KEY", "dev-token")
MODEL = os.getenv("AMBIENT_AGENT_MODEL", "z-ai/glm-5.2")
SANDBOX_BOOT_TIMEOUT_S = 120

mcp = FastMCP("ambient-video-agent")


def _headers() -> dict[str, str]:
    return {"x-api-key": API_KEY}


async def _ensure_session(
    client: httpx.AsyncClient,
    ctx: Context,
    video_id: Optional[str],
    video_path: Optional[str],
    session_id: Optional[str],
) -> tuple[str, Optional[str]]:
    """Return (session_id, video_id), creating the session if needed."""
    if session_id:
        return session_id, video_id

    if video_path:
        if not os.path.isfile(video_path):
            raise ValueError(f"video_path does not exist: {video_path}")
        await ctx.info(f"Uploading {os.path.basename(video_path)}…")
        mime = mimetypes.guess_type(video_path)[0] or "video/mp4"
        with open(video_path, "rb") as fh:
            r = await client.post(
                f"{BASE_URL}/v1/files", headers=_headers(),
                files={"file": (os.path.basename(video_path), fh, mime)},
            )
        r.raise_for_status()
        video_id = r.json()["id"]

    if not video_id:
        raise ValueError("Provide one of: session_id, video_id, or video_path.")

    await ctx.info("Creating agent + session…")
    agent = (await client.post(
        f"{BASE_URL}/v1/agents", headers=_headers(),
        json={"model": MODEL, "name": "Video Analyst",
              "tools": [{"type": "agent_toolset_20260401"}]},
    )).json()
    env = (await client.post(
        f"{BASE_URL}/v1/environments", headers=_headers(),
        json={"name": "claude-code"},
    )).json()
    session = (await client.post(
        f"{BASE_URL}/v1/sessions", headers=_headers(),
        json={"agent": agent["id"], "environment_id": env["id"],
              "metadata": {"video_id": video_id}},
    )).json()
    sid = session["id"]

    await ctx.info("Booting sandbox…")
    for _ in range(SANDBOX_BOOT_TIMEOUT_S):
        status = (await client.get(
            f"{BASE_URL}/v1/sessions/{sid}", headers=_headers())).json()["status"]
        if status == "idle":
            break
        if status == "terminated":
            raise RuntimeError("session terminated before it became ready")
        await asyncio.sleep(1)
    else:
        raise RuntimeError("sandbox did not become ready in time")
    return sid, video_id


async def _run_and_stream(
    client: httpx.AsyncClient, ctx: Context, sid: str, question: str
) -> tuple[str, list[str]]:
    """Send the question, stream events, emit progress, collect answer + steps."""
    await client.post(
        f"{BASE_URL}/v1/sessions/{sid}/events", headers=_headers(),
        json={"events": [{"type": "user.message", "content": question}]},
    )

    answer = ""
    transcript: list[str] = []
    step = 0
    event: Optional[str] = None

    async with client.stream(
        "GET", f"{BASE_URL}/v1/sessions/{sid}/events/stream", headers=_headers()
    ) as resp:
        resp.raise_for_status()
        async for line in resp.aiter_lines():
            if line.startswith("event:"):
                event = line[6:].strip()
            elif line.startswith("data:"):
                data = json.loads(line[5:].strip())
                if event == "agent.tool_use":
                    step += 1
                    name = data.get("name", "tool")
                    args = json.dumps(data.get("input") or {})
                    await ctx.report_progress(step, None, f"🔧 {name}")
                    await ctx.info(f"tool call · {name}({args})")
                    transcript.append(f"🔧 {name}({args})")
                elif event == "agent.tool_result":
                    text = "".join(b.get("text", "") for b in data.get("content") or [])
                    flag = "⚠️ error" if data.get("is_error") else "📄 result"
                    await ctx.info(f"{flag}: {text[:300]}")
                    transcript.append(f"{flag}: {text}")
                elif event == "agent.message":
                    answer = "".join(b.get("text", "") for b in data.get("content") or [])
                elif event == "session.status_idle":
                    break
    return answer, transcript


@mcp.tool()
async def analyze_video(
    question: str,
    video_id: Optional[str] = None,
    video_path: Optional[str] = None,
    session_id: Optional[str] = None,
    ctx: Context = None,
) -> str:
    """Ask the Ambient video agent a question about a video.

    Provide exactly one video source on the first call:
      - video_path: a local video file to upload, OR
      - video_id:   an id already known to the server.
    For follow-up questions about the same video, pass back the `session_id`
    returned by a previous call (it preserves the conversation + analysis).

    Returns the agent's answer, the session_id (for follow-ups), and a transcript
    of the tool calls the agent made. Progress streams live via MCP notifications.
    """
    async with httpx.AsyncClient(timeout=httpx.Timeout(600.0)) as client:
        sid, vid = await _ensure_session(client, ctx, video_id, video_path, session_id)
        answer, transcript = await _run_and_stream(client, ctx, sid, question)

    steps = "\n".join(transcript) if transcript else "(no tool calls)"
    return (
        f"session_id: {sid}   (pass this as session_id for follow-up questions)\n"
        f"video_id: {vid}\n\n"
        f"=== agent steps ===\n{steps}\n\n"
        f"=== answer ===\n{answer or '(no final answer produced)'}"
    )


if __name__ == "__main__":
    mcp.run()  # stdio transport — what Claude Code launches

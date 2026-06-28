<h1 align="center"><img src="demo/web/logo.jpg" alt="" width="28" style="vertical-align:-6px; border-radius:6px; margin-right:6px;" /> Ambient</h1>


Ambient is a video understanding and research agent that can reason over long-form videos, interpret complex visual events, and return structured responses for advanced questions, analysis, and insights.

---

## Quick start

### 1. Prerequisites
- [uv](https://docs.astral.sh/uv/) and Python 3.13
- Docker (for the bundled Postgres + Redis, or the full stack)
- An OpenAI-compatible LLM endpoint (e.g. OpenRouter) for the agent model
- Optional: S3/R2 bucket (required for the `e2b` backend) and an e2b account

### 2. Configure
Create a `.env` in the repo root:

```dotenv
# --- LLM (OpenAI-compatible chat-completions gateway) ---
LLM_BASE_URL=https://openrouter.ai/api/v1
LLM_API_KEY=sk-...
AGENT_MODEL=z-ai/glm-5.2          # any model your gateway serves

# --- server ---
API_KEY=dev-token                    # clients send this as x-api-key
SANDBOX_BACKEND=inprocess            # "inprocess" (local ffmpeg) or "e2b"

# --- persistence / coordination (match docker-compose) ---
DATABASE_URL=postgresql://ambient:ambient@localhost:5432/ambient
REDIS_URL=redis://localhost:6379/0

# --- S3 / R2 (required for SANDBOX_BACKEND=e2b) ---
S3_ENDPOINT=https://<account>.r2.cloudflarestorage.com
S3_BUCKET=video-analysis-tests
S3_VIDEO_BASE_KEY=videos
AWS_ACCESS_KEY_ID=...
AWS_SECRET_ACCESS_KEY=...

# --- e2b (only for SANDBOX_BACKEND=e2b) ---
E2B_API_KEY=e2b_...
E2B_TEMPLATE=video-analysis-v1-dev
```

### 3. Run

**Full stack (recommended) — API + Postgres + Redis:**
```bash
docker compose up -d --build          # server on http://127.0.0.1:8080
```

**Or run the server from source against the bundled infra:**
```bash
docker compose up -d postgres redis
uv run python -m ambient.server       # http://127.0.0.1:8080
```

Health check: `curl http://127.0.0.1:8080/v1/healthz` → `{"status":"ok"}`.

> For local dev without e2b/S3, keep `SANDBOX_BACKEND=inprocess` (needs `ffmpeg`
> on PATH). The `inprocess` tools read videos from `VIDEO_FOLDER`.

---

## Usage with the Anthropic SDK

The server implements the Managed Agents surface, so the official SDK drives it:

```python
from anthropic import Anthropic

DEFAULT_AGENT_ID = "ambient_v1"
DEFAULT_ENV_ID = "default"

client = Anthropic(base_url="http://127.0.0.1:8080", api_key="dev-token")

# 1. Upload a video (also pushed to S3 so the e2b sandbox can fetch it).
video = client.beta.files.upload(file=open("clip.mp4", "rb"))

# 2. Create an agent + environment, then a session over the video.
agent = client.beta.agents.create(
    name="Video Analyst",
    model="openai/gpt-5.5",
    system="Analyze videos and answer with citations.",
    tools=[{"type": "agent_toolset_20260401"}],
)

session = client.beta.sessions.create(
    agent=DEFAULT_AGENT_ID
    environment_id=DEFAULT_ENV_ID,
    metadata={"video_id": video.id},   # where the video lives
)

# 3. Wait for the sandbox to boot, then ask a question.
import time
while client.beta.sessions.retrieve(session.id).status != "idle":
    time.sleep(1)

client.beta.sessions.events.send(session.id, events=[{
    "type": "user.message",
    "content": "How long after the accident did the police car arrive?",
}])

# 4. Stream the run: tool calls, tool results, and the final answer.
answer = ""
with client.beta.sessions.events.stream(session.id) as stream:
    for ev in stream:
        if ev.type == "agent.tool_use":
            print("tool:", ev.name)
        elif ev.type == "agent.tool_result":
            print("result:", "".join(b.text for b in ev.content)[:200])
        elif ev.type == "agent.message":
            answer = "".join(b.text for b in ev.content)
        elif ev.type == "session.status_idle":
            break
print(answer)
```

Follow-up questions reuse the same `session.id` (the conversation + analysis are
preserved). A runnable version is in [`notebooks/test_sdk.ipynb`](notebooks/test_sdk.ipynb).

---

## Demo UI

A single-file web UI ([`demo/web/index.html`](demo/web/index.html)) — upload a video or paste a YouTube URL, start a session, and chat with live streaming tool calls and agent responses.

**1. Start the agent server** (from repo root):
```bash
docker compose up -d            # API + Postgres + Redis on :8080
```

**2. Open the UI:**
```bash
open demo/web/index.html        # or serve it with any static file server
```

Configure the server URL and API key via the **Settings** panel in the UI.
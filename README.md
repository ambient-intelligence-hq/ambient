<h1 align="center"><img src="demo/web/logo.jpg" alt="" width="28" style="vertical-align:-6px; border-radius:6px; margin-right:6px;" /> Ambient</h1>

<p align="center">
  <img src="docs/assets/ambient-prompts.gif" alt="Ambient: turn video into … — a cycle of five example outcomes" width="800" />
</p>


<p align="center"><b>One video. Any outcome.</b></p>

Ambient is a video agent that reasons over your footage, short clips or hours long, and turns it into what you need: a structured reports, datasets,  processed clips with bounding boxannotations , GIFs and more.

<div align="center">

## 🏆 Ambient wins at ECCV 2026

Ambient won **2 tracks** and placed **runner-up in a third** at the<br>
**Meta Wearable AI Challenge, ECCV 2026**.

🥇 **EgoProactive — Large Model** &nbsp;·&nbsp;
🥇 **EgoLongQA — Small Model** &nbsp;·&nbsp;
🥈 **EgoProactive — Small Model**

[Leaderboard](https://huggingface.co/spaces/facebook/wearable-ai-leaderboard) ·
[EgoProactive Report](https://arxiv.org/abs/2609.07099) ·
[EgoLongQA Report](https://arxiv.org/abs/2609.07154) ·
[Models & Datasets](https://huggingface.co/collections/ambient-intelligence-labs/wearables-ai-workshop-eccv-2026)

</div>

---

## Quick start (local)

The fastest way to try Ambient on your own machine: one command installs the
engine and the Studio (web UI) with Postgres and Redis, all in Docker. No S3, no
cloud sandbox — videos are processed locally with ffmpeg inside the engine
container.

**You need:** Docker ([Docker Desktop](https://www.docker.com/products/docker-desktop/)
on macOS/Windows; Docker Engine + the compose plugin on Linux) and an API key for
an OpenAI-compatible LLM endpoint ([OpenRouter](https://openrouter.ai/keys) by
default).

```bash
curl -fsSL https://raw.githubusercontent.com/ambient-intelligence-hq/ambient/main/install/install.sh | bash
```

The installer:

1. checks Docker is installed and running;
2. asks for the LLM endpoint, the **agent model** (plans and answers), the
   **video model** (watches clips; must accept video input) and your API key;
3. writes `~/.ambient/` — `docker-compose.yml`, a private `.env` with generated
   secrets, and `bin/ambientctl` (also linked into `~/.local/bin`);
4. pulls the images, starts everything, waits until it's healthy and opens the
   Studio.

| | URL |
|---|---|
| Studio (web UI) | http://localhost:3000 |
| API (Anthropic SDK `base_url`) | http://localhost:8080 — key: `ambientctl config get ENGINE_API_KEY` |

If a port is taken, the next free one is used and printed at the end.

**First run:** open the Studio, click **Add video** (or drop a file on the stage,
or paste a YouTube URL), then ask a question. **Agent** mode works step by step
with tools; **Fast** mode answers in one quick pass.

**Day to day:**

```bash
ambientctl status                 # containers + URLs
ambientctl logs [engine|studio]   # follow logs
ambientctl stop | start | restart
ambientctl update                 # pull newer images and restart
ambientctl config show            # settings (secrets redacted)
ambientctl config set AGENT_MODEL google/gemini-3.1-pro-preview   # applies immediately
ambientctl config set LLM_API_KEY sk-or-...
ambientctl uninstall              # remove containers, keep data (--purge deletes it)
```

Re-running the install command upgrades in place and keeps your settings and
data. Installer options (`--dir`, `--version`, `--no-start`, …) and environment
knobs are listed in `bash install/install.sh --help`.

> Working from a clone? `bash install/install.sh` does the same thing.

---

## Self-hosting

Running Ambient for a team or on a server? See the
**[self-hosting guide](docs/self-hosting.md)**. It covers:

- the layers (Studio, engine, Postgres, Redis, media sandbox, object storage,
  LLM endpoint) and what each needs;
- three ways to deploy: the install script, Docker Compose, or external managed
  Postgres / Redis / S3;
- running media in an **e2b sandbox** (recommended for shared installs: it
  isolates the agent's shell and ffmpeg work from your server);
- HTTPS, access control, upgrades, backups and a configuration reference.

---

## Development: run from source

For working on Ambient itself. The installer above runs the published images;
this runs the engine from your checkout.

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
AGENT_MODEL=deepseek/deepseek-v4.1-flash          # any model your gateway serves

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
    agent=DEFAULT_AGENT_ID,
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
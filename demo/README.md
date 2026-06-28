---
title: Ambient Video Agent
emoji: 🎬
colorFrom: indigo
colorTo: purple
sdk: gradio
sdk_version: 6.19.0
app_file: app.py
pinned: false
---

# Ambient Video Agent — demo UI

A Gradio front-end for the Ambient managed-agents video service. It drives the same
HTTP API as `notebooks/test_sdk.ipynb` (via the Anthropic SDK) and showcases the
full loop:

1. **Upload a video** — pushed to the server's Files API (and on to S3, so the
   e2b sandbox can fetch it).
2. **Start a session** — creates the agent + environment + session and waits for
   the sandbox to boot.
3. **Chat** — ask a question; the user message, each **tool call** and **tool
   result** (collapsible cards), and the final **agent answer** stream in live.
4. **Follow up** — further questions reuse the same session (and its context).

## Run locally

The agent server must be reachable. Bring it up from the repo root:

```bash
docker compose up -d            # api + postgres + redis  (server on :8080)
# or, against your own infra:   python -m ambient.server
```

Then launch the UI:

```bash
uv run python demo/app.py       # http://127.0.0.1:7860
```

Connection defaults (`http://127.0.0.1:8080`, key `dev-token`) are editable in
the **Connection & agent settings** accordion, or via env vars
`AMBIENT_BASE_URL` / `AMBIENT_API_KEY` / `AMBIENT_AGENT_MODEL`.

## Deploy to Hugging Face Spaces

Copy the contents of this `demo/` folder to the Space root (so `app.py` and
`requirements.txt` are at the top level — this README's frontmatter selects the
Gradio SDK). Point the Space at a publicly reachable server by setting Space
**secrets**:

- `AMBIENT_BASE_URL` — your deployed agent server URL
- `AMBIENT_API_KEY` — its API key

> The server itself is not part of the Space — host it separately (it needs
> Postgres + Redis, and e2b/LLM credentials).

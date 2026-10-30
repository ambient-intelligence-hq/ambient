# Examples

Ambient speaks the Anthropic **Managed Agents** protocol, so the official
`anthropic` SDK drives it directly — point the client at your server and use
`client.beta.*`. No custom client, no bespoke HTTP.

## [`quickstart.py`](quickstart.py)

End-to-end: upload a video → open a session over it → ask a question → stream the
run (tool calls, tool results, usage, and the final answer).

```bash
# 1. Infra + server (see the repo README for details)
docker compose up -d postgres redis
uv run python -m ambient.server            # http://127.0.0.1:8080

# 2. Ask about a local clip
uv run python examples/quickstart.py --video path/to/clip.mp4 "What happens in this video?"

# ...fast mode: one dense-frame vision pass, no tools (much faster; good for
# whole-video summaries/Q&A that don't need frame-accurate grounding)
uv run python examples/quickstart.py --mode fast --video path/to/clip.mp4 "Summarize this video."

# ...or reuse an already-uploaded video by id
uv run python examples/quickstart.py --video-id <video_id> "Summarize it."
```

`--mode` picks the seeded agent: `agent` (default) → `ambient_v1`, the tool loop;
`fast` → `ambient_fast`, a single dense-frame vision pass. Same SDK flow either
way — only the agent id changes.

Config via env (local defaults shown): `AMBIENT_BASE_URL=http://127.0.0.1:8080`,
`AMBIENT_API_KEY=dev-token`.

## [`car_damage_report.ipynb`](car_damage_report.ipynb)

A worked use case: a **vehicle damage inspector**. A custom agent (task-specific
`system` prompt) inspects a walk-around video and returns a damage report with
timestamps and annotated evidence frames (an uploaded HTML report when the server
has S3 configured), then a follow-up turns the same findings into JSON validated
against a Pydantic schema via `output_structure`. Shows creating and reusing an
agent, sessions and follow-ups, streaming tool calls, and session cost.

```bash
uv run --with jupyterlab jupyter lab examples/car_damage_report.ipynb   # set VIDEO_PATH in the setup cell
```

### The shape of it

```python
from anthropic import Anthropic

client = Anthropic(base_url="http://127.0.0.1:8080", api_key="dev-token")

video   = client.beta.files.upload(file=open("clip.mp4", "rb"))   # -> video.id
session = client.beta.sessions.create(
    agent="ambient_v1",            # seeded default: the agentic tool loop
    environment_id="default",      # seeded default
    metadata={"video_id": video.id},
)
client.beta.sessions.events.send(session.id, events=[
    {"type": "user.message", "content": "What happens in this video?"},
])
with client.beta.sessions.events.stream(session.id) as stream:
    for ev in stream:
        ...   # agent.tool_use / agent.tool_result / agent.message / session.status_idle
```

### Notes

- **Agents are pre-seeded** — `ambient_v1` (agentic tool loop) and `ambient_fast`
  (single dense-frame vision pass) exist on startup, so you can skip
  `agents.create` / `environments.create` entirely. Both `agent` and
  `environment_id` are still required arguments of `sessions.create`.
- **The video lives in `metadata.video_id`.** Uploading also starts background
  description ingestion; session readiness never blocks on it.
- **Follow-ups reuse `session.id`** — the conversation and everything the agent
  already gathered (clips, frames) are preserved server-side.
- **Structured output:** pass a JSON schema via
  `events.send(..., extra_body={"output_structure": MyModel.model_json_schema()})`
  to force the final answer into that shape.
- A notebook version with an offline SDK-compatibility check is in
  [`notebooks/test_sdk_agents.ipynb`](../notebooks/test_sdk_agents.ipynb).

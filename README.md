<h1 align="center">Ambient: Video Engine for AI Agents - Video In, Action out </h1>

<p align="center">
  <img src="docs/assets/ambient-prompts.gif" alt="Ambient turning a video into five example outcomes" width="800" />
</p>

Give Ambient a video, from a short clip to hours of footage, and ask for what you need: answers with timestamps, reports, datasets, annotated clips, GIFs and more.

🏆 Won 2 tracks and placed second in a third at the [Meta Wearable AI Challenge, ECCV 2026](https://huggingface.co/spaces/facebook/wearable-ai-leaderboard).

## Install

**You need two things:**

1. **Docker, installed and running.** Get [Docker Desktop](https://www.docker.com/products/docker-desktop/) on Mac or Windows (on Windows, run the commands below in WSL2). On Linux, install Docker Engine with the Compose plugin.
2. **An API key** from a model provider. See [Pick a model](#pick-a-model).

Then choose one option. The first install downloads about 1.5 GB and takes a few minutes.

### Option A: let your coding agent do it (easiest)

Paste this into Claude Code, Codex, Cursor, or any agent that can run commands:

```text
Set up Ambient, an open-source video engine for AI agents, on this machine.
Read the setup guide and follow it:
curl -fsSL https://goambient.ai/agents/setup.md
Show me your plan before installing anything
```

The agent checks Docker, asks for your model and API key, shows you its plan, and waits for your OK before installing.

### Option B: run one command

```bash
curl -fsSL https://goambient.ai/install.sh | bash
```

It asks three questions. Use the answers from [Pick a model](#pick-a-model), or press Enter to accept the default:

| Question | Default |
|---|---|
| LLM endpoint | `https://openrouter.ai/api/v1` |
| Model | `qwen/qwen3.8-27b` |
| API key | Empty. You can add it later. |

When it finishes, the Studio opens in your browser.

## Use it

1. Open the Studio at **http://localhost:3000**. If that port was taken, the installer prints the one it used.
2. Click **Add video**, drop in a file, or paste a YouTube link.
3. Type a question, like *"List the key moments with timestamps."*

Choose **Agent** mode for careful, step-by-step answers, or **Fast** mode for one quick pass.

## Pick a model

Ambient is recommended to use the following models and providers for best performance. Even though, you can use any model and provider you want.

| Model | Use it when | Best provider | Also good |
|---|---|---|---|
| **DeepSeek V4.1 Flash** | You want low cost. Good place to start. | [Modal](https://modal.com) | [Together](https://www.together.ai) |
| **Kimi K3** | You don't mind about 10× DeepSeek's input cost for an answer in fewer steps. | [Modal](https://modal.com/docs/guide/shared-endpoints) Shared endpoint | [Together](https://www.together.ai) |
| **GPT-6.1 Sol** | You want OpenAI's higher tier. | [OpenAI](https://platform.openai.com) | [OpenRouter](https://openrouter.ai) |
| **GPT-6 Luna** | You want OpenAI at the lowest cost. | [OpenAI](https://platform.openai.com) | [OpenRouter](https://openrouter.ai) |

Each provider has its own endpoint and its own name for the model. Copy both from this table:

| Provider | LLM endpoint | Model names |
|---|---|---|
| Modal | Your own endpoint URL from Modal, ending in `/v1` | `deepseek-ai/DeepSeek-V4.1-Flash-1017`, `moonshotai/Kimi-K3` |
| Together | `https://api.together.ai/v1` | `deepseek-ai/DeepSeek-V4.1-Flash-1017`, `moonshotai/Kimi-K3` |
| OpenAI | `https://api.openai.com/v1` | `gpt-6.1-sol`, `gpt-6-luna` |
| OpenRouter | `https://openrouter.ai/api/v1` | `openai/gpt-6.1-sol`, `openai/gpt-6-luna` |

Your provider can limit how many images and how much video fit in one request. Ambient handles this for you. To compare providers, see [vlm.fyi](https://vlm.fyi).

**To change model later,** run these and replace the values with your own. Each command applies right away:

```bash
ambientctl config set LLM_BASE_URL https://api.together.ai/v1
ambientctl config set AGENT_MODEL deepseek-ai/DeepSeek-V4.1-Flash-1017
ambientctl config set LLM_MODEL deepseek-ai/DeepSeek-V4.1-Flash-1017
ambientctl config set LLM_API_KEY your-key-here
```

To remove Ambient but keep your data, run `ambientctl uninstall`.

## Use it from code

Ambient is built for scalable workflows. It works programmatically with the official Anthropic Python SDK. Run `pip install anthropic`, then get your key with `ambientctl config get ENGINE_API_KEY`.

```python
import time
from anthropic import Anthropic

client = Anthropic(base_url="http://localhost:8080", api_key="your-engine-api-key")

# 1. Upload a video and open a session on it
video = client.beta.files.upload(file=open("clip.mp4", "rb"))
session = client.beta.sessions.create(
    agent="ambient_v1",            # or "ambient_fast" for Fast mode
    environment_id="default",
    metadata={"video_id": video.id},
)
while client.beta.sessions.retrieve(session.id).status != "idle":
    time.sleep(1)

# 2. Ask a question
client.beta.sessions.events.send(session.id, events=[
    {"type": "user.message", "content": "List the key moments with timestamps."},
])

# 3. Read the answer
with client.beta.sessions.events.stream(session.id) as stream:
    for event in stream:
        if event.type == "agent.message":
            print("".join(block.text for block in event.content))
        elif event.type == "session.status_idle":
            break
```

To ask a follow-up, send another message on the same `session.id`. More examples are in [`examples/`](examples/).

## Learn more

- [Documentation](https://goambient.ai/docs/)
- [Self-hosting for a team or server](docs/self-hosting.md)
- [Run from source](https://goambient.ai/docs/quickstart/)
- ECCV 2026 reports: [EgoProactive](https://arxiv.org/abs/2609.07099), [EgoLongQA](https://arxiv.org/abs/2609.07154), [models and datasets](https://huggingface.co/collections/ambient-intelligence-labs/wearables-ai-workshop-eccv-2026)

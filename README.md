<p align="center">
  <a href="https://huggingface.co/spaces/facebook/wearable-ai-leaderboard"><img src="https://img.shields.io/badge/ECCV%202026-2%20wins%20%2B%201%20runner--up-f5b301?logo=huggingface&logoColor=white" alt="ECCV 2026: 2 wins and 1 runner-up at the Meta Wearable AI Challenge" /></a>
</p>

<h1 align="center">Ambient: Open-source video engine for AI agents</h1>

<p align="center">
  <strong>Video in, any outcome out.</strong><br/>
  A long video rarely fits in one model request. Ambient agents work through the footage step by step, so you can ask for what you need from a short clip or hours of video.
</p>

<p align="center">
  <a href="#quick-start">Quick Start</a> •
  <a href="https://goambient.ai/docs/">Docs</a> •
  <a href="https://goambient.ai/docs/sdk/quickstart/">SDK</a> •
  <a href="examples/">Examples</a>
</p>

<p align="center">
  <img src="docs/assets/ambient-prompts.gif" alt="Ambient turning a video into five example outcomes" width="800" />
</p>

🏆 Won 2 tracks and placed second in a third at the [Meta Wearable AI Challenge, ECCV 2026](https://huggingface.co/spaces/facebook/wearable-ai-leaderboard).

## Why Ambient

| Capability                | What it changes                                                                                               |
| ------------------------- | ------------------------------------------------------------------------------------------------------------- |
| **Any outcome**           | Ask for answers with timestamps, reports, datasets, annotated clips, GIFs and more.                           |
| **Answers you can check** | Citations point to the moments in the video, so you can see the evidence yourself.                            |
| **Agent or Fast mode**    | Agent mode searches and zooms in step by step for careful answers. Fast mode takes one quick pass.            |
| **Your choice of model**  | Use any OpenAI-compatible endpoint. Ambient sizes each request to fit your provider's image and video limits. |
| **Runs on your machine**  | The engine, Studio and database run in Docker on your computer. Frames go to the model provider you choose.   |

## Quick Start

You need [Docker](https://www.docker.com/products/docker-desktop/), installed and running, and an API key from a model provider. On Windows, run this in WSL2. On Linux, install Docker Engine with the Compose plugin.

```bash
curl -fsSL https://goambient.ai/install.sh | bash
```

The first install takes a few minutes. It asks three questions. To start with DeepSeek V4.1 Flash on [Together](https://www.together.ai), enter:

| Question     | Answer                                   |
| ------------ | ---------------------------------------- |
| LLM endpoint | `https://api.together.ai/v1`             |
| Model        | `deepseek-ai/DeepSeek-V4.1-Flash-1017`   |
| API key      | Your Together key. You can add it later. |

When it finishes, the Studio opens in your browser.

**Or let your coding agent do it.** Paste this into Claude Code, Codex, Cursor, or any agent that can run commands. It checks Docker, asks for your model and API key, and waits for your OK before installing:

```text
Set up Ambient, an open-source video engine for AI agents, on this machine.
Read the setup guide and follow it:
curl -fsSL https://goambient.ai/agents/setup.md
Show me your plan before installing anything
```

## How to use

1. Open the Studio at **[http://localhost:3000](http://localhost:3000)**. If that port was taken, the installer prints the one it used.
2. Click **Add video**, drop in a file, or paste a YouTube link.
3. Type a question, like *"List the key moments with timestamps."*

## Choose a model

DeepSeek V4.1 Flash is a good place to start. You can use any model and provider you want.

<details>
<summary><strong>Recommended models and providers</strong></summary>

| Model                   | Use it when                                                                  | Best provider                                                          | Also good                           |
| ----------------------- | ---------------------------------------------------------------------------- | ---------------------------------------------------------------------- | ----------------------------------- |
| **DeepSeek V4.1 Flash** | You want low cost. Good place to start.                                      | [Modal](https://modal.com)                                             | [Together](https://www.together.ai) |
| **Kimi K3**             | You don't mind about 10× DeepSeek's input cost for an answer in fewer steps. | [Modal](https://modal.com/docs/guide/shared-endpoints) Shared endpoint | [Together](https://www.together.ai) |
| **GPT-6.1 Sol**         | You want OpenAI's higher tier.                                               | [OpenAI](https://platform.openai.com)                                  | [OpenRouter](https://openrouter.ai) |
| **GPT-6 Luna**          | You want OpenAI at the lowest cost.                                          | [OpenAI](https://platform.openai.com)                                  | [OpenRouter](https://openrouter.ai) |

Each provider has its own endpoint and its own name for the model. Copy both from this table:

| Provider   | LLM endpoint                                      | Model names                                                  |
| ---------- | ------------------------------------------------- | ------------------------------------------------------------ |
| Modal      | Your own endpoint URL from Modal, ending in `/v1` | `deepseek-ai/DeepSeek-V4.1-Flash-1017`, `moonshotai/Kimi-K3` |
| Together   | `https://api.together.ai/v1`                      | `deepseek-ai/DeepSeek-V4.1-Flash-1017`, `moonshotai/Kimi-K3` |
| OpenAI     | `https://api.openai.com/v1`                       | `gpt-6.1-sol`, `gpt-6-luna`                                  |
| OpenRouter | `https://openrouter.ai/api/v1`                    | `openai/gpt-6.1-sol`, `openai/gpt-6-luna`                    |

We made extensive comparisons of provider and model combinations to come up with our recommendations, see [vlm.fyi](https://vlm.fyi).

</details>

## Use it from code

Ambient is built for scale and async workflows. It works with the official Anthropic Python SDK. Point the client at your engine and use `client.beta.*`:

```python
from anthropic import Anthropic

client = Anthropic(base_url="http://localhost:8080", api_key="your-engine-api-key")
```

<details>
<summary><strong>Full example: upload a video and ask a question</strong></summary>

Run `pip install anthropic`, then get your key with `ambientctl config get ENGINE_API_KEY`.

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

To ask a follow-up, send another message on the same `session.id`.

</details>

## Resources

| Need                           | Link                                                                                                                                                                                                                 |
| ------------------------------ | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Read the full documentation    | [Docs](https://goambient.ai/docs/)                                                                                                                                                                                   |
| Build on Ambient               | [SDK quickstart](https://goambient.ai/docs/sdk/quickstart/) and `[examples/](examples/)`                                                                                                                             |
| Self-host for a team or server | [Self-hosting guide](docs/self-hosting.md)                                                                                                                                                                           |
| Run from source                | [Quickstart](https://goambient.ai/docs/quickstart/)                                                                                                                                                                  |
| Compare model providers        | [vlm.fyi](https://vlm.fyi)                                                                                                                                                                                           |
| Read the ECCV 2026 work        | [EgoProactive](https://arxiv.org/abs/2609.07099), [EgoLongQA](https://arxiv.org/abs/2609.07154), [models and datasets](https://huggingface.co/collections/ambient-intelligence-labs/wearables-ai-workshop-eccv-2026) |

## FAQ

<details>
<summary><strong>How can I self-host Ambient?</strong></summary>

Ambient is built for scale. However, We are still working on the right recommendations for self-hosting. See the current[self-hosting guide](docs/self-hosting.md). If you face issues, [open an issue](https://github.com/ambient-intelligence-hq/ambient/issues/).

</details>

<details>
<summary><strong>Which commands do I use day to day?</strong></summary>

```bash
ambientctl status      # is it running? shows the URLs
ambientctl open        # open the Studio
ambientctl stop        # stop it
ambientctl start       # start it again
ambientctl update      # get the latest version
ambientctl logs engine # watch the logs (Ctrl+C to exit)
```

</details>

<details>
<summary><strong>How do I change the model later?</strong></summary>

Run these and replace the values with your own. Each command applies right away:

```bash
ambientctl config set LLM_BASE_URL https://api.together.ai/v1
ambientctl config set AGENT_MODEL deepseek-ai/DeepSeek-V4.1-Flash-1017
ambientctl config set LLM_MODEL deepseek-ai/DeepSeek-V4.1-Flash-1017
ambientctl config set LLM_API_KEY your-key-here
```

</details>

<details>
<summary><strong>Does my model need to accept video?</strong></summary>

No. If a model or provider can't take video, Ambient sends frames instead.

</details>

<details>
<summary><strong>Something went wrong. What do I do?</strong></summary>

| What you see                                                       | What to do                                                                                               |
| ------------------------------------------------------------------ | -------------------------------------------------------------------------------------------------------- |
| `Docker is not installed` or `Cannot connect to the Docker daemon` | Install Docker, or start Docker Desktop, then run the install again.                                     |
| `ambientctl: command not found`                                    | Use the full path: `~/.ambient/bin/ambientctl status`                                                    |
| The Studio doesn't open                                            | Run `ambientctl status` and use the URL it prints.                                                       |
| Questions fail with a 401 or 403 error                             | Your API key is wrong or out of credit. Set a new one: `ambientctl config set LLM_API_KEY your-key-here` |
| A service is unhealthy or keeps restarting                         | Run `ambientctl logs engine`. It's often a wrong model name or endpoint.                                 |

Running the install command again is always safe. It keeps your settings and videos.

</details>

<details>
<summary><strong>How do I remove Ambient?</strong></summary>

To remove Ambient but keep your data, run `ambientctl uninstall`.

</details>

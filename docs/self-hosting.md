# Self-hosting Ambient

This guide covers running Ambient on your own infrastructure: what the pieces
are, three ways to deploy them, how to use external Postgres, Redis and S3, and
why we recommend the **e2b sandbox** for any shared or long-lived install.

For a quick local try-out, see [Quick start](../README.md#quick-start-local)
in the main README.

- [Architecture: the layers](#architecture-the-layers)
- [Choose a deployment](#choose-a-deployment)
- [Recommended: run media in an e2b sandbox](#recommended-run-media-in-an-e2b-sandbox)
- [Option A — the install script](#option-a--the-install-script)
- [Option B — Docker Compose](#option-b--docker-compose)
- [Option C — external Postgres, Redis and S3](#option-c--external-postgres-redis-and-s3)
- [Reverse proxy and access control](#reverse-proxy-and-access-control)
- [Operations](#operations)
- [Configuration reference](#configuration-reference)

---

## Architecture: the layers

```
 Browser ──► Studio (Next.js, :3000) ──► Engine (FastAPI, :8080) ──► LLM endpoint
                  │                          │  │  │                 (OpenAI-compatible)
                  ▼                          │  │  └──► Media backend
             Postgres (studio db)            │  │        inprocess: ffmpeg in the engine
                                             │  │        e2b:       per-session sandbox ──► S3
                                             │  └──► Redis   (leases, SSE fan-out, ingest queue)
                                             └─────► Postgres (engine db) + video storage
 Anthropic SDK ────────────────────────────► Engine (Managed Agents API)
```

| Layer | What it does | Needs |
|---|---|---|
| **Studio** | Web UI: chats, video library, streaming answers. Talks only to the engine; the engine API key never reaches the browser. | Its own Postgres database, `AUTH_SECRET`, `ENGINE_URL`, `ENGINE_API_KEY` |
| **Engine** | The agent: sessions, tool loop, video ingest (descriptions, YouTube imports). Serves the [Managed Agents API](../README.md#usage-with-the-anthropic-sdk). | Postgres, Redis, an LLM endpoint + key |
| **Postgres** | Engine: sessions, event logs, agents, files. Studio: users, chats, messages. | Postgres 14+ (16 is what we ship) |
| **Redis** | Per-session run lease and SSE fan-out across workers, plus the ingest job stream. | Redis 6.2+ with Streams and Pub/Sub; persistence (AOF) recommended |
| **Media backend** | Cuts clips, samples frames, runs the agent's shell tool. `inprocess` = ffmpeg inside the engine container. `e2b` = an isolated cloud sandbox per session. | `inprocess`: nothing extra. `e2b`: an e2b account + S3 |
| **Object storage** | Durable copy of each video, pre-cut clip tiles, and files the agent exports (`upload_artifact`). | S3-compatible bucket. Optional for `inprocess`, **required** for `e2b` |
| **LLM endpoint** | The agent model (plans, answers) and the video model (watches clips). | Any OpenAI-compatible chat-completions endpoint: OpenRouter, a self-hosted vLLM, … |

Every piece of state lives in Postgres, Redis and storage, never in the engine
process — so engine workers and replicas are interchangeable.

---

## Choose a deployment

| | A. Install script | B. Docker Compose | C. External services |
|---|---|---|---|
| Best for | One VM, fastest setup | Your own compose file / CI-managed stack | Production, multiple replicas |
| Postgres + Redis | Bundled containers | Bundled or yours | Managed (RDS, Cloud SQL, Upstash, …) |
| Media backend | `inprocess` (add e2b with an override) | Your choice | `e2b` recommended |
| Updates | `ambientctl update` | `docker compose pull && up -d` | Your deploy pipeline |

All three run the same published images:
`ghcr.io/ambient-intelligence-hq/ambient-engine` and
`ghcr.io/ambient-intelligence-hq/ambient-studio` (linux/amd64 + arm64, tagged
`latest`, `main`, `sha-<commit>` and release versions).

---

## Recommended: run media in an e2b sandbox

For anything beyond a personal install, set `SANDBOX_BACKEND=e2b`.

- **Isolation.** The agent has a shell tool (`ENABLE_BASH_TOOL`, on by default)
  and runs ffmpeg on untrusted video files. With `inprocess`, both run **inside
  the engine container**, with the engine's privileges and access to its
  environment: your LLM key, database URL and S3 credentials. With `e2b`, they
  run in a separate, disposable microVM per session, which only receives S3
  credentials and never an LLM key.
- **Capacity.** Clip cutting, tiling and YouTube downloads move off your server.
  The engine stays a light API process, so a small VM serves many sessions.
- **Speed on long videos.** Ingest pre-cuts each video into clip tiles in S3,
  and the sandbox streams sources with range reads instead of downloading them.
- **Horizontal scale.** Videos live in S3 rather than on one host's disk, so any
  engine replica can serve any session.

What it takes:

1. **An e2b account** and API key: [e2b.dev](https://e2b.dev).
2. **An S3-compatible bucket** the sandbox can reach (see
   [Object storage](#object-storage-s3)).
3. **The media sandbox template, built in your e2b team** (one-off, and again
   after upgrading Ambient if the template changed):
   ```bash
   git clone https://github.com/ambient-intelligence-hq/ambient && cd ambient
   uv sync --extra e2b
   export E2B_API_KEY=e2b_...
   cd ambient/sandboxes/e2b/templates/video-analysis-v1
   uv run python build_prod.py        # builds "video-analysis-v1" (8 vCPU, 8 GB)
   ```
4. **An engine image with the e2b SDK.** It's included from the images built
   after this guide was added. Older images fail on `SANDBOX_BACKEND=e2b`: pull
   a newer tag, or build from source.
5. **Engine settings:**
   ```dotenv
   SANDBOX_BACKEND=e2b
   E2B_API_KEY=e2b_...
   E2B_TEMPLATE=video-analysis-v1
   # plus the S3_* / AWS_* settings below
   ```

Each Agent-mode session gets one sandbox, booted when the session starts
(Fast mode doesn't need one). A sandbox shuts
itself down after `SANDBOX_WALL_SECONDS` (default 1800) without activity, so
idle sessions stop costing money. Ingest (descriptions, YouTube imports) uses
short-lived sandboxes of its own. Set a spend limit in the e2b dashboard.

> **Caution:** if you keep `inprocess` on a shared install, turn the shell tool
> off: `ENABLE_BASH_TOOL=false`.

---

## Option A — the install script

One VM, everything in Docker, managed with `ambientctl`.

### 1. Prepare the server

- Linux (Ubuntu 22.04+ / Debian 12+). With `inprocess`, start at 2 vCPU / 4 GB
  RAM and scale CPU with usage (ffmpeg). With `e2b`, 1 vCPU / 2 GB is plenty.
  Leave disk space for videos unless they go to S3.
- Docker Engine with the compose plugin:
  ```bash
  curl -fsSL https://get.docker.com | sh
  sudo usermod -aG docker $USER    # then log out and back in
  sudo systemctl enable docker     # start on boot
  ```
- A DNS name (e.g. `ambient.example.com`) pointing at the server, ports 80/443
  open.

### 2. Install

Non-interactive, ports bound to localhost so only the reverse proxy can reach
them:

```bash
curl -fsSL https://raw.githubusercontent.com/ambient-intelligence-hq/ambient/main/install/install.sh \
  | AMBIENT_LLM_KEY=sk-or-... \
    AMBIENT_MODEL=google/gemini-3.1-pro-preview \
    AMBIENT_VIDEO_MODEL=google/gemini-3.8-flash \
    bash -s -- --yes --no-open --version <tag>
```

`--version` pins the image tag, so `ambientctl update` only picks up fixes for
that tag. Use `AMBIENT_LLM_BASE_URL` for a non-OpenRouter endpoint. Everything
lands in `~/.ambient/` (`docker-compose.yml`, `.env`, `bin/ambientctl`).

### 3. Add e2b + S3 (recommended)

The installer's `docker-compose.yml` is regenerated on every re-run, so put your
changes in `~/.ambient/docker-compose.override.yml`. Docker Compose (and
`ambientctl`) merges it automatically:

```yaml
# ~/.ambient/docker-compose.override.yml
services:
  engine:
    environment:
      SANDBOX_BACKEND: e2b
      E2B_API_KEY: ${E2B_API_KEY}
      E2B_TEMPLATE: video-analysis-v1
      S3_ENDPOINT: ${S3_ENDPOINT}
      S3_BUCKET: ${S3_BUCKET}
      AWS_ACCESS_KEY_ID: ${AWS_ACCESS_KEY_ID}
      AWS_SECRET_ACCESS_KEY: ${AWS_SECRET_ACCESS_KEY}
```

Put the values in `~/.ambient/.env`; `ambientctl config set` keeps the file
private and applies the change:

```bash
ambientctl config set E2B_API_KEY e2b_...
ambientctl config set S3_ENDPOINT https://<account>.r2.cloudflarestorage.com
ambientctl config set S3_BUCKET ambient-videos
ambientctl config set AWS_ACCESS_KEY_ID ...
ambientctl config set AWS_SECRET_ACCESS_KEY ...
```

Check the engine picked it up: `ambientctl logs engine` should show sessions
booting e2b sandboxes when you start a chat.

### 4. Reverse proxy

See [Reverse proxy and access control](#reverse-proxy-and-access-control).

---

## Option B — Docker Compose

Use this when you manage the stack yourself (your own compose file, GitOps, a
PaaS that takes compose). A production-shaped starting point, with e2b + S3 and
the bundled Postgres/Redis:

```yaml
# docker-compose.yml
services:
  postgres:
    image: postgres:16-alpine
    restart: unless-stopped
    environment:
      POSTGRES_USER: ambient
      POSTGRES_PASSWORD: ${POSTGRES_PASSWORD}
      POSTGRES_DB: ambient
    volumes: [postgres:/var/lib/postgresql/data]
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U ambient -d ambient"]
      interval: 5s
      retries: 30

  redis:
    image: redis:7-alpine
    restart: unless-stopped
    command: ["redis-server", "--appendonly", "yes"]
    volumes: [redis:/data]
    healthcheck:
      test: ["CMD", "redis-cli", "ping"]
      interval: 5s
      retries: 30

  engine:
    image: ghcr.io/ambient-intelligence-hq/ambient-engine:${AMBIENT_VERSION:-latest}
    restart: unless-stopped
    depends_on:
      postgres: { condition: service_healthy }
      redis: { condition: service_healthy }
    env_file: .env          # LLM_*, AGENT_MODEL, LLM_MODEL, E2B_*, S3_*, AWS_*
    environment:
      DATABASE_URL: postgresql://ambient:${POSTGRES_PASSWORD}@postgres:5432/ambient
      REDIS_URL: redis://redis:6379/0
      API_KEY: ${ENGINE_API_KEY}
      SANDBOX_BACKEND: e2b
      SERVER_WORKERS: "2"
    volumes: [videos:/data]
    ports: ["127.0.0.1:8080:8080"]
    healthcheck:
      test: ["CMD", "python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8080/v1/healthz', timeout=3)"]
      interval: 10s
      start_period: 60s
      retries: 12

  studio:
    image: ghcr.io/ambient-intelligence-hq/ambient-studio:${AMBIENT_VERSION:-latest}
    restart: unless-stopped
    depends_on:
      engine: { condition: service_healthy }
    environment:
      POSTGRES_URL: postgresql://ambient:${POSTGRES_PASSWORD}@postgres:5432/ambient_studio
      AUTH_SECRET: ${AUTH_SECRET}
      ENGINE_URL: http://engine:8080
      ENGINE_API_KEY: ${ENGINE_API_KEY}
    ports: ["127.0.0.1:3000:3000"]

volumes:
  postgres:
  redis:
  videos:
```

```dotenv
# .env  (chmod 600)
AMBIENT_VERSION=<tag>
POSTGRES_PASSWORD=<openssl rand -hex 32>
ENGINE_API_KEY=<openssl rand -hex 32>
AUTH_SECRET=<openssl rand -hex 32>

LLM_BASE_URL=https://openrouter.ai/api/v1
LLM_API_KEY=sk-or-...
AGENT_MODEL=google/gemini-3.1-pro-preview   # plans + answers
LLM_MODEL=google/gemini-3.8-flash           # watches clips (must accept video)

E2B_API_KEY=e2b_...
E2B_TEMPLATE=video-analysis-v1
S3_ENDPOINT=https://<account>.r2.cloudflarestorage.com
S3_BUCKET=ambient-videos
AWS_ACCESS_KEY_ID=...
AWS_SECRET_ACCESS_KEY=...
```

```bash
docker compose up -d --wait
```

The Studio creates its database (`ambient_studio`) and runs its migrations on
every start; the engine creates its tables on start.

**Building from source** instead of pulling images: replace `image:` with
`build: .` for the engine and `build: ambient/studio/console` for the Studio.

---

## Option C — external Postgres, Redis and S3

Point the engine and Studio at managed services and drop the bundled
containers. The engine and Studio are then stateless; run as many replicas as
you need behind a load balancer.

### Postgres

```dotenv
# engine
DATABASE_URL=postgresql://ambient:<password>@db.example.com:5432/ambient?sslmode=require
# studio
POSTGRES_URL=postgresql://ambient:<password>@db.example.com:5432/ambient_studio?sslmode=require
```

- Use two databases, `ambient` (engine) and `ambient_studio` (Studio), on one
  server or two.
- The engine creates its tables and indexes on startup (`CREATE … IF NOT
  EXISTS`); give its user DDL rights on its database.
- The Studio creates its database if it's missing, then migrates. If your user
  can't `CREATE DATABASE`, create `ambient_studio` yourself first.
- Size connections for `replicas × SERVER_WORKERS × 10` on the engine side (each
  worker holds a pool of up to 10).

### Redis

```dotenv
REDIS_URL=rediss://default:<password>@redis.example.com:6379/0   # rediss:// = TLS
```

- Needs **Streams** (`XADD`, `XREADGROUP`) and **Pub/Sub**. Most managed Redis
  offerings qualify; cluster mode isn't required. Use a single primary endpoint.
- Turn on persistence (AOF) so queued video-ingest jobs survive a restart.
  Leases and pub/sub are transient.
- All engine replicas must share the same Redis — it's what lets any replica
  stream any session.

### Object storage (S3)

```dotenv
S3_ENDPOINT=https://<account>.r2.cloudflarestorage.com   # R2, MinIO, or AWS
S3_BUCKET=ambient-videos
S3_VIDEO_BASE_KEY=videos
AWS_ACCESS_KEY_ID=...
AWS_SECRET_ACCESS_KEY=...
```

- Any S3-compatible store with SigV4. Cloudflare R2 is the tested path (no
  egress fees, which matters for video). For AWS S3, use the regional endpoint
  (`https://s3.<region>.amazonaws.com`) and set `AWS_DEFAULT_REGION`.
- Layout: `videos/<video_id>.mp4` (sources), `<video_id>/tiles/…` (pre-cut
  clips), `artifacts/<video_id>/…` (files the agent exports, shared as presigned
  links valid for `ARTIFACT_URL_TTL_SECONDS`, default 7 days).
- The key needs `s3:GetObject`, `s3:PutObject` and `s3:ListBucket` on the
  bucket. Keep the bucket private — everything is served through the engine or
  presigned URLs.
- With `inprocess`, S3 is optional: videos are always kept in the engine's
  `/data` volume and copied to S3 when it's configured. Without S3, keep a
  single engine host so every worker sees the same volume.

### Engine replicas

- `SERVER_WORKERS` runs several processes in one container; replicas run
  several containers. Both coordinate through Postgres + Redis.
- With `inprocess`, replicas need S3 (or a shared `/data` volume) so each can
  read every video. With `e2b` they're fully interchangeable.
- The load balancer needs no sticky sessions, but must allow long-lived
  streaming responses (see the nginx settings below).

---

## Reverse proxy and access control

> **Put access control in front of the Studio.** Sign-in is currently off:
> every visitor gets a guest session, and every question spends your LLM
> credits. Don't expose it to the internet without proxy auth, SSO, a VPN or a
> private network such as Tailscale.

[Caddy](https://caddyserver.com/docs/install) handles TLS automatically.
`/etc/caddy/Caddyfile`:

```caddyfile
ambient.example.com {
    # Generate the hash with: caddy hash-password
    basic_auth {
        admin $2a$14$REPLACE_WITH_HASH
    }
    reverse_proxy 127.0.0.1:3000
}

# Optional: the API for the Anthropic SDK (protected by ENGINE_API_KEY).
api.ambient.example.com {
    reverse_proxy 127.0.0.1:8080
}
```

Then `sudo systemctl reload caddy`.

With **nginx**, answers stream over long-lived connections (agent runs can take
many minutes) and uploads are large:

```nginx
proxy_buffering off;
proxy_read_timeout 3600s;
client_max_body_size 4g;
```

For SSO, put an identity-aware proxy in front instead of basic auth
(oauth2-proxy, Cloudflare Access, Tailscale Serve, …).

To serve the Studio under a sub-path (e.g. `/studio`), build it with
`NEXT_PUBLIC_BASE_PATH=/studio`.

---

## Operations

- **Health:** `GET /v1/healthz` on the engine, `GET /ping` on the Studio.
- **Logs:** `ambientctl logs engine` (Option A) or `docker compose logs -f engine`.
- **Changing models or keys:** update the settings and restart the engine. On
  start, the engine re-syncs its built-in agents from `AGENT_MODEL` / `LLM_MODEL`,
  so new chats use the new models. Existing chats keep the model they started
  with.
- **Upgrades:** pull the new tag and restart (`ambientctl update`, or re-run
  the installer with a new `--version`). The Studio migrates its database and
  the engine creates any new tables on start. If you use e2b, rebuild the
  template when `ambient/sandboxes/e2b/templates/` changed in the release.
- **Backups:** Postgres holds everything except videos:
  ```bash
  docker compose exec -T postgres pg_dumpall -U ambient > ambient-$(date +%F).sql
  ```
  Videos live in the `videos` volume and, when configured, in S3. With the
  installer, back up `~/.ambient/.env` too: the restored database needs its
  `POSTGRES_PASSWORD`.
- **Costs:** LLM usage dominates. Every session reports token usage and cost
  (the Studio's "Usage & cost" line). With e2b, sandboxes stop after
  `SANDBOX_WALL_SECONDS` idle.

---

## Configuration reference

Engine settings are read from the environment (or a `.env` file), case-insensitive.

| Setting | Default | Purpose |
|---|---|---|
| `LLM_BASE_URL`, `LLM_API_KEY` | — | OpenAI-compatible endpoint + key |
| `AGENT_MODEL` | `google/gemini-3.1-pro-preview` | Model that plans and answers (Agent mode) |
| `LLM_MODEL` | `qwen/qwen3.5-27b` | Video model: watches clips; Fast mode; descriptions |
| `AGENT_BASE_URL`, `AGENT_API_KEY` | = `LLM_*` | Separate endpoint for the agent model |
| `API_KEY` | `dev-token` | Engine API key (the installer calls it `ENGINE_API_KEY`). **Change it.** |
| `DATABASE_URL` | local Postgres | Engine database |
| `REDIS_URL` | local Redis | Leases, streaming fan-out, ingest queue |
| `SERVER_WORKERS` | `1` | Worker processes per container |
| `SANDBOX_BACKEND` | `inprocess` | `inprocess` (ffmpeg in the engine) or `e2b` |
| `E2B_API_KEY`, `E2B_TEMPLATE` | —, `video-analysis-v1` | e2b backend |
| `SANDBOX_WALL_SECONDS` | `1800` | Idle timeout for a session's e2b sandbox |
| `ENABLE_BASH_TOOL` | `true` | Agent shell tool. Turn off on shared `inprocess` installs |
| `S3_ENDPOINT`, `S3_BUCKET`, `S3_VIDEO_BASE_KEY` | —, —, `videos` | Object storage |
| `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY` | — | Object storage credentials |
| `VIDEO_FOLDER` | `/data/videos` (image) | Local video storage |
| `YOUTUBE_IMPORT_ENABLED` | `true` | YouTube URL imports |
| `YOUTUBE_MAX_DURATION_SECONDS` | `10800` | Longest YouTube video accepted (3 h) |

Studio settings:

| Setting | Purpose |
|---|---|
| `POSTGRES_URL` | Studio database (created on first start) |
| `AUTH_SECRET` | Session encryption (`openssl rand -base64 32`) |
| `ENGINE_URL`, `ENGINE_API_KEY` | How the Studio reaches the engine (server-side only) |
| `STUDIO_MAX_MESSAGES_PER_HOUR` | Optional per-user message quota |
| `NEXT_PUBLIC_BASE_PATH` | Serve under a sub-path (build-time) |

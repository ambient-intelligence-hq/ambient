# syntax=docker/dockerfile:1.9

############################  builder  ############################
FROM python:3.13-slim-bookworm AS builder

# grab the uv binary from uv's official image (instead of pip-installing it)
COPY --from=ghcr.io/astral-sh/uv:0.5 /uv /bin/uv

# use the system python; don't let uv download its own
ENV UV_PYTHON_DOWNLOADS=0

WORKDIR /app

# Dependencies first, in their own layer, so a source change doesn't reinstall
# every package. The `youtube` extra adds yt-dlp (+ its YouTube challenge
# solvers), which the host backend (SANDBOX_BACKEND != e2b) shells out to for
# YouTube imports — no e2b sandbox or S3 needed. /app/.venv/bin is on PATH below.
# The `e2b` extra is the SDK for SANDBOX_BACKEND=e2b (media in an E2B sandbox).
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --locked --no-dev --extra youtube --extra e2b --no-install-project

# Then the project itself.
COPY ambient ./ambient
RUN uv sync --locked --no-dev --extra youtube --extra e2b

############################  runtime  ############################
FROM python:3.13-slim-bookworm AS runtime

# Links the published image (ghcr.io/ambient-intelligence-hq/ambient-engine)
# to its repository on GitHub.
LABEL org.opencontainers.image.source="https://github.com/ambient-intelligence-hq/ambient" \
      org.opencontainers.image.description="Ambient engine — video-understanding agent (Managed Agents API)"

# ffmpeg/ffprobe are needed only by the in-process media backend
# (SANDBOX_BACKEND=inprocess). Drop this layer entirely if you run e2b.
RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Deno: the JavaScript runtime yt-dlp uses to solve YouTube's challenges (with the
# yt-dlp-ejs scripts from the `youtube` extra). Without it yt-dlp warns that
# YouTube extraction is deprecated and formats may be missing. Single static
# binary from the official image; pinned for reproducible builds.
COPY --from=denoland/deno:bin-2.9.7 /deno /usr/local/bin/deno

RUN useradd --create-home --uid 10001 appuser \
    && mkdir -p /data && chown appuser:appuser /data

WORKDIR /app
COPY --from=builder --chown=appuser:appuser /app /app

ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    SERVER_HOST=0.0.0.0 \
    SERVER_PORT=8080 \
    VIDEO_FOLDER=/data/videos

USER appuser
EXPOSE 8080
# Uploaded videos land here (the sqlite store is gone; sessions/events live in
# Postgres). For the inprocess backend the tools read videos back from this dir.
VOLUME ["/data"]

# Durable state (Postgres) and cross-worker coordination (Redis) live outside the
# process, so the server is multi-worker safe. Set SERVER_WORKERS>1 (and point
# DATABASE_URL / REDIS_URL at shared instances) to run multiple workers.
CMD ["python", "-m", "ambient.server"]
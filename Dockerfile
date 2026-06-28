# syntax=docker/dockerfile:1.9

############################  builder  ############################
FROM python:3.13-slim-bookworm AS builder

# grab the uv binary from uv's official image (instead of pip-installing it)
COPY --from=ghcr.io/astral-sh/uv:0.5 /uv /bin/uv

# use the system python; don't let uv download its own
ENV UV_PYTHON_DOWNLOADS=0

WORKDIR /app

# everything uv needs to build the venv
COPY pyproject.toml uv.lock README.md ./
COPY ambient ./ambient

# create /app/.venv with the locked deps + the project
RUN uv sync --frozen --no-dev

############################  runtime  ############################
FROM python:3.13-slim-bookworm AS runtime

# ffmpeg/ffprobe are needed only by the in-process media backend
# (SANDBOX_BACKEND=inprocess). Drop this layer entirely if you run e2b.
RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg \
    && rm -rf /var/lib/apt/lists/*

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
# Self-contained image for the Ambient video-agent MCP server.
# Claude Code launches it over stdio with `docker run -i --rm …`, so it works
# from any project/folder without this repo's uv environment.
#
# Build (from repo root):
#   docker build -f integrations/Dockerfile -t ambient-video-mcp .
#
# It reaches the agent API over the network — default host.docker.internal:8080
# (the host's published port; works on Docker Desktop / macOS + Windows). On
# Linux, add `--add-host=host.docker.internal:host-gateway` to the run args, or
# point AMBIENT_BASE_URL at the API's reachable address.
FROM python:3.13-slim

RUN pip install --no-cache-dir "mcp>=1.2" "httpx>=0.28"

WORKDIR /app
COPY integrations/mcp_video_agent.py /app/mcp_video_agent.py

ENV AMBIENT_BASE_URL=http://host.docker.internal:8080 \
    AMBIENT_API_KEY=dev-token \
    AMBIENT_AGENT_MODEL=z-ai/glm-5.2 \
    PYTHONUNBUFFERED=1

# stdio MCP transport — no extra stdout, JSON-RPC only.
ENTRYPOINT ["python", "mcp_video_agent.py"]

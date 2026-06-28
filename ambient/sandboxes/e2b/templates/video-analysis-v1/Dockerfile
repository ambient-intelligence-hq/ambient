FROM e2bdev/base:latest

# System dependencies: ffmpeg (includes ffprobe) for video processing.
# unzip/curl/ca-certificates are needed to fetch the deno runtime below.
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    curl \
    unzip \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# JS runtime for yt-dlp. Without one, recent yt-dlp can only use the deprecated
# non-JS YouTube clients (e.g. android_vr), whose AV1/SABR formats return HTTP 403
# mid-download. deno lets yt-dlp drive the default web clients reliably.
# See https://github.com/yt-dlp/yt-dlp/wiki/EJS
RUN curl -fsSL https://github.com/denoland/deno/releases/latest/download/deno-x86_64-unknown-linux-gnu.zip -o /tmp/deno.zip \
    && unzip /tmp/deno.zip -d /usr/local/bin \
    && rm /tmp/deno.zip \
    && chmod +x /usr/local/bin/deno

# Python dependencies required by main.py (media-only; no LLM path):
#   - Pillow: image I/O when using the decord backend
#   - decord: fast video reader (preferred backend; falls back to ffmpeg if unavailable)
#   - pydantic: result-envelope models
#   - boto3: S3 download / upload
#   - yt-dlp: YouTube URL materialization inside the sandbox
RUN pip install --no-cache-dir \
    Pillow \
    decord \
    pydantic \
    boto3 \
    yt-dlp

COPY . /app

WORKDIR /app

# Default: print usage. Override with the desired sub-command when running:
#   docker run <image> python main.py extract-frames --video-id x --upload-s3 ...
#   docker run <image> python main.py fetch-clip     --video-id x --start 10 --end 40 --upload-s3
#   docker run <image> python main.py prepare-youtube --video-id x --url https://youtu.be/... --upload-s3
CMD ["python", "main.py", "--help"]

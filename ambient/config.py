from functools import lru_cache
from typing import Optional
from pydantic_settings import BaseSettings, SettingsConfigDict
from dataclasses import dataclass

class Settings(BaseSettings):
    """
    Settings for the application. Environment variables are loaded from the .env file and overrides the default values.
    Environment variables are case-insensitive. example S3_ENDPOINT matches with s3_endpoint.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )
    s3_endpoint: str | None = None
    s3_bucket: str | None = None
    s3_video_base_key: str = "videos"
    aws_access_key_id: str | None = None
    aws_secret_access_key: str | None = None

    llm_model: str = "qwen/qwen3.5-27b"
    llm_base_url: str | None = None
    llm_api_key: str | None = None

    agent_model: str = "google/gemini-3.1-pro-preview"

    # Default agent seeded at startup so SDK callers can create sessions without
    # first calling agents.create / environments.create.
    default_agent_name: str = "Video Analyst"
    default_agent_system: str = "Analyze videos and answer with citations."

    redis_url: str = "redis://localhost:6379/0"
    database_url: str = "postgresql://ambient:ambient@localhost:5432/ambient"
    sqlite_db_path: str = "ambient_db.sqlite"
    sqlite_collection_name: str = "video_clips"
    
    # vllm embedding client
    # embedding_model: str = "Qwen/Qwen3-VL-embedding-2B"
    # embedding_service_base_url: str = "https://infinitylogesh--embedding-service-serve.modal.run"
    # embedding_service_api_key: str = ""
    # embedding_dimension: int = 512
    # embedding_client: str = "vllm"

    # gemini embedding client
    embedding_service_api_key: str = ""
    embedding_service_base_url: str = "https://generativelanguage.googleapis.com/v1beta"
    embedding_model: str = "models/gemini-embedding-2-preview"
    embedding_dimension: int = 1536
    embedding_client: str = "gemini"

    video_folder: str = "/Users/logesh/self/video-llm-tests/videos"
    video_clip_duration: int = 60 # secs
    video_clip_fps: int = 5
    video_clip_max_dimentions: Optional[int] = 768

    # Server / agent API
    api_key: str = "dev-token"
    server_host: str = "127.0.0.1"
    server_port: int = 8080
    server_db_path: str = "ambient_server.sqlite"
    session_idle_ttl_seconds: int = 30 * 60
    session_hard_ttl_seconds: int = 24 * 60 * 60
    # Selects where VideoFrameTools' media ops run:
    #   "inprocess" -> ffmpeg/decord run locally in ambient/tools/video_tools.py
    #   "e2b"       -> media ops run in an E2B sandbox (LLM calls always stay on the host)
    sandbox_backend: str = "inprocess"
    sandbox_cpu_seconds: int = 600
    sandbox_memory_mb: int = 4096
    sandbox_wall_seconds: int = 1800
    e2b_template: str = "video-analysis-v1"
    max_turns_per_run: int = 5

    # Background video-description ingestion. On upload we enqueue the video id
    # on a Redis stream; a per-worker consumer boots an ephemeral E2B sandbox,
    # runs get_video_description, and persists the result on the files row.
    ingest_stream: str = "ingest:video"
    ingest_group: str = "ingest-workers"
    ingest_concurrency: int = 2          # max concurrent sandbox boots per worker
    ingest_block_ms: int = 5000          # XREADGROUP block timeout
    ingest_max_attempts: int = 3         # give up (status=failed) after N tries
    ingest_stale_seconds: int = 300      # a 'processing' row older than this is reclaimable
    ingest_sweep_seconds: int = 300      # re-enqueue stuck pending/processing rows this often
    ingest_wait_timeout_seconds: int = 60   # how long a tool call waits for an in-flight job
    ingest_wait_poll_seconds: float = 1.5   # poll cadence while waiting

    # Source-video streaming (video I/O optimization): the sandbox reads the
    # source via a presigned URL with HTTP range reads instead of downloading
    # the whole file. See docs/video-io-optimizations-spec.md.
    stream_source_video: bool = True
    source_url_ttl: int = 7200                  # presigned source-URL lifetime (s)
    stream_min_bytes: int = 100 * 1024 * 1024   # below this, download instead
    overview_seek_concurrency: int = 16         # parallel seeks for overview sampling

    # Pre-transcoded clip tiles (removes fetch_clip latency). At ingestion the
    # whole video is transcoded once into fixed, GOP-aligned tiles at the target
    # provider quality and uploaded to S3; fetch_clip then assembles the covering
    # tiles instead of transcoding on demand. Tiles must match the provider's
    # fps/dimension or fetch_clip falls back to on-demand transcode.
    tiling_enabled: bool = True
    tile_seconds: int = 45                       # tile length (also the GOP/keyframe interval)
    tile_fps: int = 2                            # must match provider_quality_settings.fps
    tile_max_dim: int = 768                      # must match provider_quality_settings.max_dimentions
    tile_workers: int = 8                        # parallel decode ranges; set to the box vCPU count

    # YouTube URL imports. The API records the URL and the ingest worker asks an
    # ephemeral E2B sandbox to download/remux/upload it before description.
    youtube_import_enabled: bool = True
    youtube_max_duration_seconds: int = 3 * 60 * 60
    youtube_max_size_bytes: int = 5 * 1024 * 1024 * 1024
    youtube_download_timeout_seconds: int = 900
    # Multi-process coordination. Each worker process gets a unique id at import
    # time; the Redis ownership lease is keyed by session and stamped with it.
    worker_id: str = ""
    server_workers: int = 1

@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()

@dataclass
class provider_quality_settings:
    crf: Optional[int] = None
    fps: Optional[int] = 2
    max_dimentions: Optional[int] = 768
    max_size_mb: Optional[int] = None


def get_provider_quality_settings(model: str) -> provider_quality_settings:
    if "gemini" in model:
        return provider_quality_settings(max_size_mb=14)
    else:
        return provider_quality_settings()


settings = get_settings()

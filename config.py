from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


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
    aws_access_key_id: str | None = None
    aws_secret_access_key: str | None = None

    redis_url: str = "redis://localhost:6379/0"
    stream_job_stream: str = "stream.jobs"
    stream_job_group: str = "stream.processors"
    stream_job_maxlen: int = 50000
    llm_job_stream: str = "llm.jobs"
    llm_job_group: str = "llm.processors"
    stream_segmenter_max_retries: int = 3
    chunk_output_pattern: str = "chunk_%Y%m%d_%H%M%S.mp4"
    # 1080p : 1920x1080, 720p : 1280x720, 480p : 640x480 , 360p : 640x360
    # video chunks are resized to this dimension if not provided in the request
    stream_chunk_target_width: int = 640
    stream_chunk_target_height: int = 360

    stream_tmp_dir: str = "/tmp/streams"
    num_active_streams_per_worker: int = 2
    ingestion_timeout: int = 300
    chunk_presigned_url_ttl: int = 600

    screener_timeout: int = 30
    screener_base_url: str | None = None
    screener_api_key: str | None = None
    screener_model: str = "Qwen/Qwen3-VL-Reranker-2B"

    ambient_api_root_path: str = ""
    llm_api_key: str | None = None
    llm_base_url: str = "https://openrouter.ai/api"
    llm_model: str = "google/gemini-3-flash-preview"
    llm_max_in_flight: int = 10
    llm_timeout: int = 300
    llm_context_chunk_lookback: int = 0
    # number of concurrent llm analysis tasks per worker
    num_active_llm_streams_per_worker: int = 10
    webhook_timeout: int = 300
    webhook_max_retries: int = 3


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()


settings = get_settings()

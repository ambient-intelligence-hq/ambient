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
    agent_base_url: str | None = None
    agent_api_key: str | None = None


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
    # When True, clip tools skip the S3 upload and leave clip_url unset so the LLM
    # payload embeds the local clip as a base64 data URL instead. Lets the core
    # agent run with no S3/R2 bucket (see notebooks/test_sdk.ipynb).
    inline_clips: bool = False
    sandbox_cpu_seconds: int = 600
    sandbox_memory_mb: int = 4096
    sandbox_wall_seconds: int = 1800
    e2b_template: str = "video-analysis-v1"
    max_turns_per_run: int = 5
    # Analysis-clip quality for the local (non-gemini) vision endpoint. Lower
    # fps/dimension and a size cap make each focus_clip/search_clip analysis call
    # dramatically faster on the vLLM endpoint (60s@2fps/768px≈18s vs
    # 30s@1fps/512px≈4.5s) at some loss of temporal/spatial detail. Tunable via
    # ANALYSIS_* env vars so the benchmark can sweep speed/accuracy.
    analysis_fps: int = 2
    analysis_max_dim: int = 768
    analysis_max_size_mb: int = 8
    

    # Video-description generation (the high-level overview computed at ingestion).
    # Exposed as knobs; an optional faster `description_model` is the main lever
    # for trimming the description LLM call once the source download is off the
    # path. Frame count is kept at 50 (do not reduce) for description quality.
    description_max_frames: int = 50           # overview frames sent to the LLM
    description_max_dim: int = 768             # frame longest-edge for the description
    description_model: str | None = None       # None -> settings.llm_model

    # If True, the first agent run waits (bounded) for the video description to
    # land before its first LLM call, so the first answer is grounded in the
    # description. Session readiness is unaffected (still ~instant); only the first
    # run blocks. On timeout it falls back to the metadata seed + later injection.
    first_turn_wait_for_description: bool = True
    first_turn_description_timeout_seconds: int = 300

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
    # Cap the YouTube download height. Downstream consumers never use more than
    # 768px (tiles, description frames, provider quality caps), so 720p loses
    # nothing while downloading 3-10x less. 0 = uncapped (old behavior).
    youtube_max_height: int = 720
    youtube_max_duration_seconds: int = 3 * 60 * 60
    youtube_max_size_bytes: int = 5 * 1024 * 1024 * 1024
    youtube_download_timeout_seconds: int = 900
    # Multi-process coordination. Each worker process gets a unique id at import
    # time; the Redis ownership lease is keyed by session and stamped with it.
    worker_id: str = ""
    server_workers: int = 1
    model_definitions_file: str = os.path.join(os.path.dirname(__file__),"artifacts","models.json")

@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()

@dataclass
class provider_quality_settings:
    crf: Optional[int] = None
    fps: Optional[int] = 2
    max_dimentions: Optional[int] = 768
    max_size_mb: Optional[int] = None

class model_modalities(Enum):
    TEXT = "text"
    IMAGE = "image"
    VIDEO = "video"
    AUDIO = "audio"


def get_provider_quality_settings(model: str) -> provider_quality_settings:
    if "gemini" in model:
        return provider_quality_settings(max_size_mb=14)
    else:
        return provider_quality_settings(
            fps=settings.analysis_fps,
            max_dimentions=settings.analysis_max_dim,
            max_size_mb=settings.analysis_max_size_mb,
        )

# TODO: if no match with existing models file, fetch from https://openrouter.ai/api/v1/models
@lru_cache(maxsize=1)
def get_model_modalities(model: str) -> list[str]:
    """ 
    Get the input modalities for a given model.
    """

    def match_model(model:str, model_id:str) -> bool:
        if model == model_id:
            return True
        # check if model_id ends with model
        if model_id.endswith(model):
            return True
        return False

    model_definitions = {}
    try:
        with open(settings.model_definitions_file, "r") as f:
            model_definitions = json.load(f)
    except Exception as e:
        raise Exception(f"Error loading model definitions file: {e}")
    
    modalities = [model_json.get("architecture",{}).get("input_modalities", []) for model_json in model_definitions.get("data", []) if match_model(model, model_json.get("id"))]
    modalities = modalities[0] if modalities and isinstance(modalities[0], list) else modalities
    return [model_modalities(modality) for modality in modalities if modality in model_modalities]


settings = get_settings()

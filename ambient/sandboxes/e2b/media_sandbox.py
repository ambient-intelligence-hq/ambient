from ambient.server.sandbox.interface import MediaSandbox
from e2b import Sandbox
from ambient.config import settings
import re
import shlex
import json
import threading
import logging

logger = logging.getLogger(__name__)

_BOX_VIDEO_FOLDER = "/tmp/videos"
RESULT_SENTINEL = "===AMBIENT_RESULT==="
_SENTINEL_RE = re.compile(re.escape(RESULT_SENTINEL) + r"(.*?)" + re.escape(RESULT_SENTINEL), re.S)

# Media work (download + ffmpeg/decord + upload) can take minutes; the SDK
# default of 60s is far too short. Whole-video `transcode-tiles` at ingestion is
# the longest single command (~source_duration/6 on a 4-vCPU box), so the ceiling
# is set high enough to cover tiling of a max-duration (~3h) source.
_CMD_TIMEOUT_SECS = max(1800, settings.youtube_download_timeout_seconds)

# Cap how many ffmpeg/media commands run concurrently on a single box. Tool calls
# in a turn are dispatched on separate threads, so without a bound they'd all
# launch ffmpeg at once and thrash the box's CPU/memory. 3 lets a handful overlap
# while keeping the box from falling over.
_MAX_CONCURRENT_CMDS = 3

def _parse_envelope(stdout: str, stderr: str, exit_code: int) -> dict:
    """Extract and validate the fenced result envelope from main.py's stdout."""
    m = _SENTINEL_RE.search(stdout or "")
    if not m:
        tail = (stderr or "")[-2000:]
        raise RuntimeError(f"sandbox produced no result envelope (exit={exit_code}). stderr tail:\n{tail}")
    env = json.loads(m.group(1))
    if not env.get("ok"):
        err = env.get("error") or {}
        raise RuntimeError(f"sandbox tool failed: {err.get('code', 'error')}: {err.get('message', '')}")
    return env.get("data") or {}

def _media_env() -> dict[str, str]:
    """Env passed to the box: S3 only — never an LLM key."""
    env: dict[str, str] = {
        "S3_VIDEO_BASE_KEY": settings.s3_video_base_key,
        "VIDEO_FOLDER": _BOX_VIDEO_FOLDER,
        # Source-video streaming (video I/O optimization).
        "STREAM_SOURCE_VIDEO": "true" if settings.stream_source_video else "false",
        "SOURCE_URL_TTL": str(settings.source_url_ttl),
        "STREAM_MIN_BYTES": str(settings.stream_min_bytes),
        "OVERVIEW_SEEK_CONCURRENCY": str(settings.overview_seek_concurrency),
        "YOUTUBE_MAX_HEIGHT": str(settings.youtube_max_height),
        "YOUTUBE_MAX_DURATION_SECONDS": str(settings.youtube_max_duration_seconds),
        "YOUTUBE_MAX_SIZE_BYTES": str(settings.youtube_max_size_bytes),
        "YOUTUBE_DOWNLOAD_TIMEOUT_SECONDS": str(settings.youtube_download_timeout_seconds),
        # Pre-transcoded clip tiles (see transcode-tiles / concat-tiles).
        "TILE_SECONDS": str(settings.tile_seconds),
        "TILE_FPS": str(settings.tile_fps),
        "TILE_MAX_DIM": str(settings.tile_max_dim),
        "TILE_WORKERS": str(settings.tile_workers),
    }
    if settings.s3_endpoint:
        env["S3_ENDPOINT"] = settings.s3_endpoint
    if settings.s3_bucket:
        env["S3_BUCKET"] = settings.s3_bucket
    if settings.aws_access_key_id:
        env["AWS_ACCESS_KEY_ID"] = settings.aws_access_key_id
    if settings.aws_secret_access_key:
        env["AWS_SECRET_ACCESS_KEY"] = settings.aws_secret_access_key
    return env


class E2BMediaSandbox(MediaSandbox):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Bound concurrent media commands rather than serializing them outright,
        # so several tool calls in a turn can run ffmpeg in parallel (up to N).
        self._sem = threading.Semaphore(_MAX_CONCURRENT_CMDS)
        self.sandbox = None
        self.envs = _media_env()

    def create(self, *args, **kwargs):
        template = kwargs.get("template")
        limits = kwargs.get("limits")
        keepalive = (limits.wall_seconds if limits and limits.wall_seconds else 1800)
        envs = kwargs.get("envs") or  self.envs
        self._keepalive_secs = keepalive
        self.sandbox = Sandbox.create(template, timeout=keepalive,envs=envs)

    def connect(self, sandbox_id: str) -> None:
        """Reattach to an already-running sandbox by id.

        Used when a worker rehydrates a session it didn't create; the e2b sandbox
        lives in the e2b cloud, so any worker can reconnect to it by id.
        """
        self._keepalive_secs = self._keepalive_secs if getattr(self, "_keepalive_secs", None) else 1800
        self.sandbox = Sandbox.connect(sandbox_id)

    @property
    def id(self) -> str:
        if self.sandbox is None:
            raise RuntimeError("Sandbox not created")
        return self.sandbox.sandbox_id

    def run(self, argv: list[str]) -> dict:
        if self.sandbox is None:
            raise RuntimeError("Sandbox not created or killed")
        from e2b.sandbox.commands.command_handle import CommandExitException

        cmd = "python /app/main.py " + " ".join(shlex.quote(a) for a in argv)
        with self._sem:
            # Refresh the idle timeout so an active session isn't reaped mid-turn.
            self.sandbox.set_timeout(self._keepalive_secs)
            try:
                res = self.sandbox.commands.run(
                    cmd,
                    envs=self.envs,
                    cwd="/app",
                    timeout=_CMD_TIMEOUT_SECS,
                    on_stderr=lambda line: logger.info("[box %s] %s", self.sandbox.sandbox_id[:8], line.rstrip()),
                )
            except CommandExitException as exc:
                # main.py exits 1 on failure but still writes the error envelope
                # to stdout; parse it for a clean message instead of the raw
                # CommandExitException.
                return _parse_envelope(exc.stdout, exc.stderr, exc.exit_code)
        return _parse_envelope(res.stdout, res.stderr, res.exit_code)

    def run_shell(self, command: str, timeout: int | None = None) -> dict:
        """Run a raw shell command inside the sandbox (bash tool, e2b backend).

        Unlike `run`, which invokes the media CLI (`python /app/main.py …`), this
        passes `command` straight to the box's shell. Returns
        {stdout, stderr, exit_code}; a non-zero exit still returns cleanly rather
        than raising."""
        if self.sandbox is None:
            raise RuntimeError("Sandbox not created or killed")
        from e2b.sandbox.commands.command_handle import CommandExitException

        with self._sem:
            self.sandbox.set_timeout(self._keepalive_secs)
            try:
                res = self.sandbox.commands.run(
                    command,
                    envs=self.envs,
                    cwd="/app",
                    timeout=int(timeout or _CMD_TIMEOUT_SECS),
                )
            except CommandExitException as exc:
                return {"stdout": exc.stdout, "stderr": exc.stderr, "exit_code": exc.exit_code}
        return {"stdout": res.stdout, "stderr": res.stderr, "exit_code": res.exit_code}

    def kill(self) -> None:
        try:
            self.sandbox.kill()
        except Exception:
            pass

    def get_status(self) -> dict:
        return self.sandbox.status

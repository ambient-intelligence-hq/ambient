"""Background video-description ingestion.

On upload the file lands in the `files` table with `description_status="pending"`
and its `video_id` is pushed onto a Redis stream. A per-worker consumer
(`IngestWorker`, started in the app lifespan) boots an *ephemeral* E2B media
sandbox, runs `get_video_description` against it (the sandbox pulls the video
from S3/R2 itself), and persists the text back onto the file row. This is the
notebook flow in `notebooks/test_tools.ipynb`, made durable and automatic.

Durability:
  * The stream + consumer group gives at-least-once delivery; a worker that dies
    mid-job leaves the message in the PEL, reclaimed by `XAUTOCLAIM`.
  * Normal failures self-re-enqueue with an attempt cap (`ingest_max_attempts`),
    so the PEL only ever handles crash-before-ack.
  * A leader-locked sweeper re-enqueues rows stuck in pending/processing past
    `ingest_stale_seconds` — covering the dual-write gap (row written but the
    XADD never landed) and orphaned `processing` rows.

`ensure_description()` is the read side used at session-readiness: it returns the
persisted description, waiting for an in-flight job, and computing live as a
fallback. It does not require the stream — a direct video id with no row just
computes live.
"""
from __future__ import annotations

import asyncio
import logging
import uuid
from datetime import datetime, timedelta, timezone

import redis.asyncio as aioredis
from redis.exceptions import (
    ConnectionError as RedisConnectionError,
    ResponseError,
    TimeoutError as RedisTimeoutError,
)

from ambient.config import settings
from ambient.server.store import Store

log = logging.getLogger(__name__)

_SWEEPER_LOCK = "ingest:sweeper:lock"


class SourcePreparationError(RuntimeError):
    """Raised when a file's remote source cannot be materialized."""


def _now_iso() -> str:
    return (
        datetime.now(timezone.utc)
        .isoformat(timespec="milliseconds")
        .replace("+00:00", "Z")
    )


def _is_stale(updated_at: str | None, max_age_seconds: int) -> bool:
    """True if `updated_at` (ISO-8601 'Z') is older than max_age_seconds."""
    if not updated_at:
        return True
    try:
        ts = datetime.fromisoformat(updated_at.replace("Z", "+00:00"))
    except ValueError:
        return True
    return datetime.now(timezone.utc) - ts > timedelta(seconds=max_age_seconds)


async def _generate_description(video_id: str, box: object | None) -> str:
    """Run get_video_description with `box` as the active media sandbox.

    Mirrors `ToolDispatcher.call_tool`: the box is published on the
    `current_media_box` ContextVar, then the (async) tool is driven to
    completion in a worker thread so its blocking sandbox/LLM work never stalls
    the event loop. `box=None` selects the inprocess (local ffmpeg) backend.
    """
    from ambient.tools.video_description import get_video_description
    from ambient.tools.video_backend import current_media_box

    token = current_media_box.set(box)
    try:
        log.info(f"Generating description for {video_id}")
        result = await asyncio.to_thread(
            lambda: asyncio.run(get_video_description(video_id))
        )
        log.info(f"Description generated for {video_id}")
    finally:
        current_media_box.reset(token)
    description = result[0] if isinstance(result, tuple) else result
    if not description:
        raise RuntimeError(f"empty description produced for {video_id!r}")
    return description


def _set_ready(description: str):
    def mut(rec: dict) -> dict:
        rec["description"] = description
        rec["description_status"] = "ready"
        rec["description_error"] = None
        return rec

    return mut


def _set_tiles_status(status: str, error: str | None = None):
    """Mutator recording tile-precompute status (observability only).

    The S3 manifest is the source of truth for fetch_clip's fast path; this just
    surfaces ready/failed on the file row.
    """
    def mut(rec: dict) -> dict:
        rec["tiles_status"] = status
        rec["tiles_error"] = error
        return rec

    return mut


async def ensure_description(
    store: Store,
    video_id: str,
    *,
    box: object | None = None,
    wait_timeout: float | None = None,
    poll: float | None = None,
) -> str:
    """Resolve the description for `video_id`, waiting for the background job.

    Read side for session-readiness. Returns the persisted text if ready; if a
    job is in flight (pending/processing) waits up to `wait_timeout`; otherwise
    (timeout, failed, or no row) computes live using `box` and writes through.
    """
    wait_timeout = (
        settings.ingest_wait_timeout_seconds if wait_timeout is None else wait_timeout
    )
    poll = settings.ingest_wait_poll_seconds if poll is None else poll

    while True:
        rec = await store.get_file(video_id)
        if rec is None:
            break  # no row (direct video id) -> compute live
        status = rec.get("description_status")
        if status == "ready" and rec.get("description"):
            return rec["description"]
        if status in (None, "failed"):
            break  # nothing in flight -> compute live
        if status == "processing" and _is_stale(rec.get("updated_at"), settings.ingest_stale_seconds):
            break  # stale processing row -> compute live
        await asyncio.sleep(poll)

    description = await _generate_description(video_id, box)
    try:
        await store.update_file(video_id, _set_ready(description))
    except KeyError:
        pass  # direct video id with no row to write through
    return description


class IngestWorker:
    """Per-worker Redis-stream consumer + sweeper for video-description jobs."""

    def __init__(self, store: Store, redis_url: str | None = None) -> None:
        self.store = store
        self.redis_url = redis_url or settings.redis_url
        self.stream = settings.ingest_stream
        self.group = settings.ingest_group
        self.max_attempts = settings.ingest_max_attempts
        self.stale_seconds = settings.ingest_stale_seconds
        self.consumer = settings.worker_id or uuid.uuid4().hex[:12]
        self.redis: aioredis.Redis | None = None
        self._sem = asyncio.Semaphore(settings.ingest_concurrency)
        self._stop = asyncio.Event()
        self._tasks: list[asyncio.Task] = []

    # --- lifecycle --------------------------------------------------------

    async def start(self) -> None:
        # socket_timeout must exceed the XREADGROUP BLOCK window: the asyncio
        # client (unlike sync redis-py) does not auto-extend the read timeout for
        # blocking commands, so a shorter socket_timeout would trip and raise
        # before Redis returns. health_check_interval keeps idle connections
        # (managed/TLS Redis) alive between reads.
        self.redis = aioredis.from_url(
            self.redis_url,
            decode_responses=True,
            socket_timeout=settings.ingest_block_ms / 1000 + 5,
            socket_keepalive=True,
            health_check_interval=30,
        )
        try:
            await self.redis.xgroup_create(self.stream, self.group, id="$", mkstream=True)
        except ResponseError as exc:
            if "BUSYGROUP" not in str(exc):
                raise
        self._tasks = [
            asyncio.create_task(self._consume_loop(), name="ingest-consume"),
            # disable sweeper for debugging purposes.
            # asyncio.create_task(self._sweep_loop(), name="ingest-sweep"),
        ]
        log.info("ingest worker %s started (group=%s)", self.consumer, self.group)

    async def stop(self) -> None:
        self._stop.set()
        for t in self._tasks:
            t.cancel()
        for t in self._tasks:
            try:
                await t
            except (asyncio.CancelledError, Exception):  # noqa: BLE001
                pass
        if self.redis is not None:
            await self.redis.aclose()
            self.redis = None

    # --- enqueue ----------------------------------------------------------

    async def enqueue(self, video_id: str) -> None:
        """Push a video id onto the ingest stream (the row is the source of truth)."""
        await self.redis.xadd(self.stream, {"video_id": video_id})

    # --- consumer ---------------------------------------------------------

    async def _consume_loop(self) -> None:
        while not self._stop.is_set():
            try:
                await self._reclaim_stale()
                resp = await self.redis.xreadgroup(
                    self.group,
                    self.consumer,
                    {self.stream: ">"},
                    count=settings.ingest_concurrency,
                    block=settings.ingest_block_ms,
                )
                if not resp:
                    continue
                for _stream, entries in resp:
                    for msg_id, fields in entries:
                        await self._sem.acquire()
                        log.info(f"Dispatching message {msg_id} for video {fields.get('video_id')}")
                        asyncio.create_task(self._dispatch(msg_id, fields))
            except asyncio.CancelledError:
                raise
            except (RedisConnectionError, RedisTimeoutError) as exc:
                # Transient: a blocking-read window elapsed or a connection blip.
                # The next iteration reconnects; don't spam tracebacks.
                log.debug("ingest consume transient: %s", exc)
                await asyncio.sleep(0.5)
            except Exception:  # noqa: BLE001 - keep the loop alive
                log.exception("ingest consume loop error")
                await asyncio.sleep(1)

    async def _reclaim_stale(self) -> None:
        """Reclaim PEL entries from dead workers (crash-before-ack)."""
        try:
            _next, claimed, _deleted = await self.redis.xautoclaim(
                self.stream,
                self.group,
                self.consumer,
                min_idle_time=self.stale_seconds * 1000,
                start_id="0-0",
                count=settings.ingest_concurrency,
            )
        except ResponseError:
            return
        for msg_id, fields in claimed:
            await self._sem.acquire()
            asyncio.create_task(self._dispatch(msg_id, fields))

    async def _dispatch(self, msg_id: str, fields: dict) -> None:
        video_id = (fields or {}).get("video_id")
        try:
            if video_id:
                await self._run_job(video_id)
        except Exception:  # noqa: BLE001 - failure already recorded on the row
            log.exception("ingest job failed for %s", video_id)
        finally:
            # Ack the delivery either way: retries are driven by row state +
            # re-enqueue, so the PEL only holds crash-before-ack.
            try:
                await self.redis.xack(self.stream, self.group, msg_id)
            except Exception:  # noqa: BLE001
                log.exception("xack failed for %s", msg_id)
            self._sem.release()

    async def _run_job(self, video_id: str) -> None:
        if not await self._claim(video_id):
            return  # ready, actively processing elsewhere, or over attempts
        box = None
        try:
            from ambient.sandboxes.e2b.media_sandbox import E2BMediaSandbox

            box = E2BMediaSandbox()
            await asyncio.to_thread(box.create, template=settings.e2b_template)
            rec = await self.store.get_file(video_id)
            if (
                rec
                and rec.get("source_type") == "youtube"
                and rec.get("source_status") != "ready"
            ):
                await self._prepare_youtube_source(video_id, rec, box)

            # Tiling (CPU-bound, ~source_duration/6 wall-time) overlaps
            # description generation (dominated by LLM wait); both share the box.
            # Tiling is best-effort and idempotent — a missing manifest just makes
            # fetch_clip fall back to on-demand transcode, so its failure never
            # blocks description readiness. Run concurrently and collect both.
            jobs = [asyncio.create_task(_generate_description(video_id, box))]
            if settings.tiling_enabled:
                jobs.append(asyncio.create_task(asyncio.to_thread(
                    box.run,
                    ["transcode-tiles", "--video-id", video_id, "--upload-s3",
                     "--workers", str(settings.tile_workers)],
                )))
            results = await asyncio.gather(*jobs, return_exceptions=True)

            description = results[0]
            if isinstance(description, Exception):
                raise description
            await self.store.update_file(video_id, _set_ready(description))
            log.info("description ready for %s", video_id)

            if settings.tiling_enabled:
                tiling_res = results[1]
                if isinstance(tiling_res, Exception):
                    log.warning("tiling failed for %s; fetch_clip will fall back: %s", video_id, tiling_res)
                    await self.store.update_file(video_id, _set_tiles_status("failed", str(tiling_res)))
                else:
                    n_tiles = len((tiling_res or {}).get("tiles", []))
                    await self.store.update_file(video_id, _set_tiles_status("ready"))
                    log.info("tiles ready for %s (%d tiles)", video_id, n_tiles)
        except Exception as exc:  # noqa: BLE001
            await self._mark_failure(video_id, exc)
            raise
        finally:
            if box is not None:
                try:
                    await asyncio.to_thread(box.kill)
                except Exception:  # noqa: BLE001
                    log.warning("failed to kill ingest sandbox for %s", video_id)

    async def _prepare_youtube_source(
        self, video_id: str, rec: dict, box: object
    ) -> None:
        source_url = rec.get("source_url")
        if not source_url:
            raise SourcePreparationError("YouTube source row is missing source_url")

        def mark_processing(row: dict) -> dict:
            if row.get("source_status") != "ready":
                row["source_status"] = "processing"
                row["source_error"] = None
                row["source_updated_at"] = _now_iso()
            return row

        await self.store.update_file(video_id, mark_processing)
        try:
            data = await asyncio.to_thread(
                box.run,
                [
                    "prepare-youtube",
                    "--video-id",
                    video_id,
                    "--url",
                    source_url,
                    "--upload-s3",
                ],
            )
        except Exception as exc:  # noqa: BLE001
            raise SourcePreparationError(str(exc)) from exc

        def mark_ready(row: dict) -> dict:
            row["source_status"] = "ready"
            row["source_error"] = None
            row["source_updated_at"] = _now_iso()
            row["r2_key"] = data.get("r2_key") or row.get("r2_key")
            row["size_bytes"] = data.get("size_bytes") or row.get("size_bytes")
            row["mime_type"] = "video/mp4"
            row["filename"] = row.get("filename") or f"{video_id}.mp4"
            if data.get("source_file_path"):
                row["sandbox_source_path"] = data["source_file_path"]
            if data.get("s3_uri"):
                row["s3_uri"] = data["s3_uri"]
            info = data.get("download_info") or {}
            row["youtube"] = {
                "webpage_url": info.get("webpage_url"),
                "title": info.get("title"),
                "extractor": info.get("extractor"),
                "duration": data.get("duration"),
                "width": data.get("width"),
                "height": data.get("height"),
                "format_id": info.get("format_id"),
            }
            return row

        await self.store.update_file(video_id, mark_ready)
        log.info("source ready for %s", video_id)

    async def _claim(self, video_id: str) -> bool:
        """Atomically take a claimable row to `processing`. True if we own it."""
        claimed = {"ok": False}

        def mut(rec: dict) -> dict:
            status = rec.get("description_status")
            attempts = rec.get("description_attempts", 0)
            if status == "ready":
                return rec
            if attempts >= self.max_attempts:
                rec["description_status"] = "failed"
                return rec
            # A fresh pending row, or a stale 'processing' from a dead worker.
            if status == "processing" and not _is_stale(
                rec.get("updated_at"), self.stale_seconds
            ):
                return rec  # someone is actively working it
            rec["description_status"] = "processing"
            claimed["ok"] = True
            return rec

        try:
            log.info(f"Updating file {video_id} with status")
            await self.store.update_file(video_id, mut)
        except KeyError:
            log.error(f"KeyError updating file {video_id} with status")
            return False  # file deleted before we got to it
        return claimed["ok"]

    async def _mark_failure(self, video_id: str, exc: Exception) -> None:
        if isinstance(exc, SourcePreparationError):
            await self._mark_source_failure(video_id, exc)
            return

        def mut(rec: dict) -> dict:
            attempts = rec.get("description_attempts", 0) + 1
            rec["description_attempts"] = attempts
            rec["description_error"] = str(exc)
            rec["description_status"] = (
                "pending" if attempts < self.max_attempts else "failed"
            )
            return rec

        try:
            rec = await self.store.update_file(video_id, mut)
            log.error(f"Marking failure for {video_id} with status {rec.get('description_status')}")
        except KeyError:
            return
        if rec.get("description_status") == "pending":
            await self.enqueue(video_id)  # immediate retry

    async def _mark_source_failure(
        self, video_id: str, exc: SourcePreparationError
    ) -> None:
        def mut(rec: dict) -> dict:
            attempts = rec.get("source_attempts", 0) + 1
            terminal = attempts >= self.max_attempts
            rec["source_attempts"] = attempts
            rec["source_error"] = str(exc)
            rec["source_status"] = "failed" if terminal else "pending"
            rec["source_updated_at"] = _now_iso()
            rec["description_attempts"] = max(
                rec.get("description_attempts", 0), attempts
            )
            rec["description_error"] = str(exc)
            rec["description_status"] = "failed" if terminal else "pending"
            return rec

        try:
            rec = await self.store.update_file(video_id, mut)
            log.error(
                "Marking source failure for %s with status %s",
                video_id,
                rec.get("source_status"),
            )
        except KeyError:
            return
        if rec.get("description_status") == "pending":
            await self.enqueue(video_id)

    # --- sweeper ----------------------------------------------------------

    async def _sweep_loop(self) -> None:
        while not self._stop.is_set():
            try:
                await asyncio.wait_for(
                    self._stop.wait(), timeout=settings.ingest_sweep_seconds
                )
                return  # stop signalled
            except asyncio.TimeoutError:
                pass
            try:
                await self._sweep_once()
            except Exception:  # noqa: BLE001
                log.exception("ingest sweep error")

    async def _sweep_once(self) -> None:
        # One worker sweeps per interval (best-effort leader lock).
        got = await self.redis.set(
            _SWEEPER_LOCK,
            self.consumer,
            nx=True,
            px=int(settings.ingest_sweep_seconds * 1000 * 0.9),
        )
        if not got:
            return
        cutoff = (
            (datetime.now(timezone.utc) - timedelta(seconds=self.stale_seconds))
            .isoformat(timespec="milliseconds")
            .replace("+00:00", "Z")
        )
        stale = await self.store.list_stale_ingest_files(cutoff)
        for rec in stale:
            vid = rec.get("video_id") or rec.get("id")
            if vid and rec.get("description_attempts", 0) < self.max_attempts:
                log.info("sweeper re-enqueueing stuck video %s", vid)
                await self.enqueue(vid)

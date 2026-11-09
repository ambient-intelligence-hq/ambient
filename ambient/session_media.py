"""Keep a session's inline media out of `sessions.messages`, without losing any.

The host-local media backend returns clips and frames as base64 data URLs, and the
runner rewrites the whole conversation to Postgres after every tool turn. Stored
inline, one session writes the same tens of MB again and again.

Instead, each data URL is stored once as a file (S3 when configured, otherwise on the
engine's volume), and the saved conversation holds a short reference to it. Loading
restores the original data URLs, so a rehydrated runner sees exactly what it had.

Fields starting with "_" are engine bookkeeping and are never sent to the provider
(see `_normalize_for_openai`). ARCHIVE_KEY lists references to media a message no
longer shows, such as originals the media budget reduced. They are not loaded back.
"""
from __future__ import annotations

import asyncio
import base64
import binascii
import hashlib
import logging
import os
import re
import shutil
import tempfile
from pathlib import Path
from typing import Any, Iterator, Optional

from ambient.config import settings

log = logging.getLogger(__name__)

REF_PREFIX = "ambient-media:"
ARCHIVE_KEY = "_media_archive"
MISSING_NOTE = "[A saved video clip or frame from this point in the conversation is no longer available.]"

_SAFE_SESSION_ID = re.compile(r"[A-Za-z0-9_-][A-Za-z0-9_.-]*")
# Conversations hold user text, so only this exact shape counts as a reference. The
# digest becomes a filename, and anything looser would let a message read other files.
_REF = re.compile(rf"{REF_PREFIX}([0-9a-f]{{64}}):(data:[^,]*;base64)")
_CONCURRENCY = 16


def _split_data_url(url: str) -> Optional[tuple[str, bytes]]:
    """(header, decoded bytes) if re-encoding the bytes restores `url` exactly."""
    if not url.startswith("data:"):
        return None
    header, comma, payload = url.partition(",")
    if not comma or not header.endswith(";base64"):
        return None
    try:
        raw = base64.b64decode(payload, validate=True)
    except (binascii.Error, ValueError):
        return None
    if base64.b64encode(raw).decode() != payload:
        return None
    return header, raw


def _ref(digest: str, header: str) -> str:
    return f"{REF_PREFIX}{digest}:{header}"


def reference(url: str) -> Optional[str]:
    """The reference a stored data URL gets, or None if it isn't one we can store."""
    parts = _split_data_url(url)
    if parts is None:
        return None
    header, raw = parts
    return _ref(hashlib.sha256(raw).hexdigest(), header)


def _strings(node: Any) -> Iterator[str]:
    if isinstance(node, str):
        yield node
    elif isinstance(node, list):
        for item in node:
            yield from _strings(item)
    elif isinstance(node, dict):
        for key, value in node.items():
            if not str(key).startswith("_"):
                yield from _strings(value)


def _replace(node: Any, mapping: dict[str, str]) -> Any:
    """A copy of `node` with strings swapped. The caller's messages stay untouched."""
    if isinstance(node, str):
        return mapping.get(node, node)
    if isinstance(node, list):
        return [_replace(item, mapping) for item in node]
    if isinstance(node, dict):
        return {k: v if str(k).startswith("_") else _replace(v, mapping) for k, v in node.items()}
    return node


class _LocalBlobs:
    """One file per item under <root>/<session_id>/<sha256>."""

    def __init__(self, root: Path) -> None:
        self.root = root

    def put(self, session_id: str, digest: str, raw: bytes) -> None:
        directory = self.root / session_id
        path = directory / digest
        if path.exists():
            return
        directory.mkdir(parents=True, exist_ok=True)
        # On disk before the row that references it is written, so a crash can't
        # leave a reference with nothing behind it.
        fd, tmp = tempfile.mkstemp(dir=directory, prefix=".tmp-")
        try:
            with os.fdopen(fd, "wb") as fh:
                fh.write(raw)
                fh.flush()
                os.fsync(fh.fileno())
            os.replace(tmp, path)
        except BaseException:
            Path(tmp).unlink(missing_ok=True)
            raise
        dir_fd = os.open(directory, os.O_RDONLY)
        try:
            os.fsync(dir_fd)
        finally:
            os.close(dir_fd)

    def get(self, session_id: str, digest: str) -> Optional[bytes]:
        try:
            return (self.root / session_id / digest).read_bytes()
        except FileNotFoundError:
            return None

    def delete_session(self, session_id: str) -> None:
        shutil.rmtree(self.root / session_id, ignore_errors=True)


class _S3Blobs:
    """One object per item under session-media/<session_id>/<sha256>."""

    def __init__(self, client: Any, bucket: str) -> None:
        self.client = client
        self.bucket = bucket
        self._stored: set[str] = set()   # saves the upload on every later turn

    @staticmethod
    def _key(session_id: str, digest: str) -> str:
        return f"session-media/{session_id}/{digest}"

    def put(self, session_id: str, digest: str, raw: bytes) -> None:
        key = self._key(session_id, digest)
        if key not in self._stored:
            self.client.put_object(Bucket=self.bucket, Key=key, Body=raw)
            self._stored.add(key)

    def get(self, session_id: str, digest: str) -> Optional[bytes]:
        key = self._key(session_id, digest)
        try:
            body = self.client.get_object(Bucket=self.bucket, Key=key)["Body"].read()
        except self.client.exceptions.NoSuchKey:
            return None
        self._stored.add(key)
        return body

    def delete_session(self, session_id: str) -> None:
        prefix = f"session-media/{session_id}/"
        pages = self.client.get_paginator("list_objects_v2").paginate(Bucket=self.bucket, Prefix=prefix)
        for page in pages:
            objects = [{"Key": o["Key"]} for o in page.get("Contents", [])]
            if objects:
                self.client.delete_objects(Bucket=self.bucket, Delete={"Objects": objects})
        self._stored = {k for k in self._stored if not k.startswith(prefix)}


def _default_blobs() -> _LocalBlobs | _S3Blobs:
    if settings.s3_bucket:
        from ambient.utils.s3 import get_s3_client
        return _S3Blobs(get_s3_client().s3, settings.s3_bucket)
    root = settings.session_media_folder or Path(settings.video_folder).parent / "session-media"
    return _LocalBlobs(Path(root))


def _check_session_id(session_id: str) -> None:
    # The id becomes a path, and deleting a session removes that whole directory.
    if not _SAFE_SESSION_ID.fullmatch(session_id):
        raise ValueError(f"unsafe session id: {session_id!r}")


class SessionMedia:
    def __init__(self, blobs: Any = None) -> None:
        self._blobs = blobs

    @property
    def blobs(self) -> Any:
        if self._blobs is None:
            self._blobs = _default_blobs()
        return self._blobs

    async def _gather(self, fn: Any, items: list[str], session_id: str) -> list[Any]:
        semaphore = asyncio.Semaphore(_CONCURRENCY)

        async def run(item: str) -> Any:
            async with semaphore:
                return await asyncio.to_thread(fn, session_id, item)

        return await asyncio.gather(*(run(item) for item in items))

    def _store(self, session_id: str, url: str) -> str:
        """The reference for `url`. Keeps `url` inline if it can't be stored, because
        a larger row is better than a run that fails or media that's lost."""
        parts = _split_data_url(url)
        if parts is None:
            return url
        header, raw = parts
        digest = hashlib.sha256(raw).hexdigest()
        try:
            self.blobs.put(session_id, digest, raw)
        except Exception:
            log.exception("[session_media] session %s: could not store %d bytes, keeping them "
                          "in the database", session_id, len(raw))
            return url
        return _ref(digest, header)

    async def dehydrate(self, session_id: str, messages: list[dict]) -> list[dict]:
        """A copy of `messages` with each data URL stored and replaced by a reference."""
        _check_session_id(session_id)
        urls = list({s for s in _strings(messages) if s.startswith("data:")})
        refs = await self._gather(self._store, urls, session_id)
        return _replace(messages, dict(zip(urls, refs)))

    def _load(self, session_id: str, ref: str) -> Optional[str]:
        digest, header = _REF.fullmatch(ref).groups()
        raw = self.blobs.get(session_id, digest)
        return None if raw is None else f"{header},{base64.b64encode(raw).decode()}"

    async def hydrate(self, session_id: str, messages: list[dict]) -> list[dict]:
        """Restore every reference to its data URL. Read errors raise, because saving
        after a partial load would turn a passing failure into permanent loss. Only
        media that is truly gone becomes a note, and its reference is kept."""
        _check_session_id(session_id)
        refs = list({s for s in _strings(messages) if _REF.fullmatch(s)})
        if not refs:
            return messages
        urls = dict(zip(refs, await self._gather(self._load, refs, session_id)))
        missing = {ref for ref, url in urls.items() if url is None}
        if missing:
            log.error("[session_media] session %s: %d stored media item(s) are missing",
                      session_id, len(missing))
        found = {ref: url for ref, url in urls.items() if url is not None}
        return [self._restore(message, found, missing) for message in messages]

    @staticmethod
    def _restore(message: dict, found: dict[str, str], missing: set[str]) -> dict:
        restored = _replace(message, found)
        gone = missing.intersection(_strings(message))
        content = restored.get("content")
        if not gone or not isinstance(content, list):
            return restored
        restored["content"] = [
            {"type": "text", "text": MISSING_NOTE} if gone.intersection(_strings(block)) else block
            for block in content
        ]
        restored[ARCHIVE_KEY] = [*restored.get(ARCHIVE_KEY, []), *sorted(gone)]
        return restored

    async def delete_session(self, session_id: str) -> None:
        _check_session_id(session_id)
        await asyncio.to_thread(self.blobs.delete_session, session_id)

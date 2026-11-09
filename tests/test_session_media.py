"""Session media is stored outside `sessions.messages` and restored exactly on load."""
from __future__ import annotations

import asyncio
import base64
import json
import os
import tempfile
import unittest
from pathlib import Path

from ambient.media_budget import MediaBudget, fit_media_budget
from ambient.server.runner import _normalize_for_openai
from ambient.server.store import Store
from ambient.session_media import (
    ARCHIVE_KEY,
    MISSING_NOTE,
    REF_PREFIX,
    SessionMedia,
    _LocalBlobs,
    _S3Blobs,
    reference,
)

SESSION = "ses_test"


def data_url(raw: bytes, mime: str = "image/jpeg") -> str:
    return f"data:{mime};base64,{base64.b64encode(raw).decode()}"


def image(url: str) -> dict:
    return {"type": "image_url", "image_url": {"url": url}}


def video(url: str) -> dict:
    return {"type": "video_url", "video_url": {"url": url}}


def conversation() -> list[dict]:
    return [
        {"role": "system", "content": "You analyze video."},
        {"role": "user", "content": [
            {"type": "text", "text": "Frames between 0 and 10 seconds"},
            image(data_url(b"frame-one")),
            image(data_url(b"frame-two")),
            image("https://bucket.example/hosted.jpg"),
        ]},
        {"role": "assistant", "content": [
            {"type": "tool_use", "id": "t1", "name": "focus_clip", "input": {"query": "data: not media"}},
        ]},
        {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "t1", "content": "seen"}]},
        {"role": "user", "content": [video(data_url(b"clip-bytes", "video/mp4"))]},
    ]


class Base(unittest.TestCase):
    def setUp(self) -> None:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        self.media = SessionMedia(_LocalBlobs(self.root))

    def files(self) -> list[Path]:
        return [p for p in self.root.rglob("*") if p.is_file()]

    def dehydrate(self, messages: list[dict]) -> list[dict]:
        return asyncio.run(self.media.dehydrate(SESSION, messages))

    def hydrate(self, messages: list[dict]) -> list[dict]:
        return asyncio.run(self.media.hydrate(SESSION, messages))


class RoundTripTests(Base):
    def test_a_saved_conversation_loads_back_byte_for_byte(self):
        original = conversation()
        self.assertEqual(self.hydrate(self.dehydrate(original)), original)

    def test_no_base64_is_left_in_what_gets_saved(self):
        saved = json.dumps(self.dehydrate(conversation()))
        self.assertNotIn(";base64,", saved)
        self.assertIn(REF_PREFIX, saved)
        self.assertIn("https://bucket.example/hosted.jpg", saved)

    def test_files_hold_the_decoded_bytes(self):
        self.dehydrate(conversation())
        self.assertEqual({p.read_bytes() for p in self.files()}, {b"frame-one", b"frame-two", b"clip-bytes"})

    def test_saving_leaves_the_runners_messages_alone(self):
        original = conversation()
        self.dehydrate(original)
        self.assertEqual(original, conversation())

    def test_saving_again_keeps_one_file_per_item(self):
        self.dehydrate(conversation())
        before = {p: p.stat().st_mtime_ns for p in self.files()}
        self.dehydrate(conversation())
        self.assertEqual({p: p.stat().st_mtime_ns for p in self.files()}, before)

    def test_rows_saved_before_this_change_still_load(self):
        old_row = conversation()
        self.assertEqual(self.hydrate(old_row), conversation())

    def test_loading_an_old_row_then_saving_moves_its_media_out(self):
        saved = json.dumps(self.dehydrate(self.hydrate(conversation())))
        self.assertNotIn(";base64,", saved)

    def test_base64_that_would_not_reencode_exactly_stays_inline(self):
        for url in ("data:image/jpeg;base64,not*base64", "data:image/jpeg;base64,QQ", "data:text/plain,hello"):
            messages = [{"role": "user", "content": [image(url)]}]
            self.assertEqual(self.dehydrate(messages), messages)
        self.assertEqual(self.files(), [])


class SafetyTests(Base):
    def test_user_text_cannot_read_files_through_a_fake_reference(self):
        secret = self.root / "secret.txt"
        secret.write_text("private")
        for text in (
            f"{REF_PREFIX}../secret.txt:data:text/plain;base64",
            f"{REF_PREFIX}{'a' * 63}/:data:text/plain;base64",
            f"please look at {REF_PREFIX}{'a' * 64}:data:image/jpeg;base64",
        ):
            messages = [{"role": "user", "content": [{"type": "text", "text": text}]}]
            self.assertEqual(self.hydrate(messages), messages)

    def test_unsafe_session_ids_are_refused(self):
        for session_id in ("../other", "a/b", "..", ".", "", "a\n"):
            with self.assertRaises(ValueError):
                asyncio.run(self.media.delete_session(session_id))
            with self.assertRaises(ValueError):
                asyncio.run(self.media.dehydrate(session_id, []))

    def test_deleting_a_session_leaves_other_sessions_alone(self):
        asyncio.run(self.media.dehydrate("ses_other", conversation()))
        self.dehydrate(conversation())
        asyncio.run(self.media.delete_session(SESSION))
        self.assertFalse((self.root / SESSION).exists())
        self.assertEqual(len(list((self.root / "ses_other").iterdir())), 3)


class FailureTests(Base):
    def test_a_failed_write_keeps_the_media_in_the_row(self):
        class Full(_LocalBlobs):
            def put(self, *_a):
                raise OSError("No space left on device")

        media = SessionMedia(Full(self.root))
        with self.assertLogs("ambient.session_media", "ERROR"):
            saved = asyncio.run(media.dehydrate(SESSION, conversation()))
        self.assertEqual(saved, conversation())

    def test_a_failed_write_leaves_no_temp_file(self):
        real = os.fsync
        os.fsync = lambda _fd: (_ for _ in ()).throw(OSError("disk error"))
        self.addCleanup(setattr, os, "fsync", real)
        with self.assertRaises(OSError):
            _LocalBlobs(self.root).put(SESSION, "a" * 64, b"x")
        self.assertEqual(self.files(), [])

    def test_a_read_error_fails_the_load(self):
        saved = self.dehydrate(conversation())

        class Broken(_LocalBlobs):
            def get(self, *_a):
                raise OSError("I/O error")

        with self.assertRaises(OSError):
            asyncio.run(SessionMedia(Broken(self.root)).hydrate(SESSION, saved))

    def test_a_missing_file_becomes_a_note_and_keeps_its_reference(self):
        original = conversation()
        saved = self.dehydrate(original)
        missing_ref = reference(data_url(b"frame-one"))
        (self.root / SESSION / missing_ref.split(":")[1]).unlink()

        with self.assertLogs("ambient.session_media", "ERROR"):
            loaded = self.hydrate(saved)

        self.assertEqual(loaded[1]["content"][1], {"type": "text", "text": MISSING_NOTE})
        self.assertEqual(loaded[1]["content"][2], image(data_url(b"frame-two")))
        self.assertEqual(loaded[1][ARCHIVE_KEY], [missing_ref])
        self.assertEqual(loaded[4], original[4])
        # The reference survives another save, and isn't sent to the provider.
        self.assertEqual(self.dehydrate(loaded)[1][ARCHIVE_KEY], [missing_ref])
        self.assertNotIn(missing_ref, json.dumps(_normalize_for_openai(loaded)))


class BudgetTests(Base):
    def reduced(self) -> tuple[list[dict], list[str]]:
        frames = [data_url(f"frame-{i}".encode()) for i in range(8)]
        messages = [
            {"role": "system", "content": "s"},
            {"role": "user", "content": [image(data_url(b"context"))]},
            {"role": "user", "content": [
                {"type": "text", "text": "focus_clip: between 0 seconds and 10 seconds"},
                *[image(f) for f in frames],
            ]},
            {"role": "assistant", "content": "noted"},
            {"role": "user", "content": [image(data_url(b"fresh"))]},
        ]
        self.dehydrate(messages)
        budget = MediaBudget(images=8, frames=10_000, inline_bytes=10**9, source="test")
        self.assertIsNotNone(fit_media_budget(messages, budget))
        return messages, frames

    def test_a_reduced_message_points_at_every_original_it_dropped(self):
        messages, frames = self.reduced()
        kept = {b["image_url"]["url"] for b in messages[2]["content"] if b["type"] == "image_url"}
        dropped = {reference(f) for f in frames if f not in kept}
        self.assertTrue(dropped)
        self.assertEqual(set(messages[2][ARCHIVE_KEY]), dropped)
        stored = {p.name for p in self.files()}
        self.assertTrue({r.split(":")[1] for r in dropped} <= stored)

    def test_a_restart_restores_the_reduced_view_without_the_originals(self):
        messages, _ = self.reduced()
        loaded = self.hydrate(self.dehydrate(messages))
        self.assertEqual(loaded, messages)
        self.assertEqual(sum(b["type"] == "image_url" for b in loaded[2]["content"]),
                         sum(b["type"] == "image_url" for b in messages[2]["content"]))

    def test_archive_entries_never_reach_the_provider(self):
        messages, _ = self.reduced()
        self.assertTrue(any(ARCHIVE_KEY in m for m in messages))
        self.assertNotIn(ARCHIVE_KEY, json.dumps(_normalize_for_openai(messages)))


class _FakeS3:
    class exceptions:
        class NoSuchKey(Exception):
            pass

    def __init__(self) -> None:
        self.objects: dict[str, bytes] = {}
        self.puts = 0

    def put_object(self, Bucket, Key, Body):
        self.puts += 1
        self.objects[Key] = Body

    def get_object(self, Bucket, Key):
        if Key not in self.objects:
            raise self.exceptions.NoSuchKey()

        class Body:
            @staticmethod
            def read():
                return self.objects[Key]

        return {"Body": Body}

    def get_paginator(self, _name):
        fake = self

        class Paginator:
            @staticmethod
            def paginate(Bucket, Prefix):
                return [{"Contents": [{"Key": k} for k in fake.objects if k.startswith(Prefix)]}]

        return Paginator

    def delete_objects(self, Bucket, Delete):
        for obj in Delete["Objects"]:
            del self.objects[obj["Key"]]


class S3Tests(unittest.TestCase):
    def setUp(self) -> None:
        self.s3 = _FakeS3()
        self.media = SessionMedia(_S3Blobs(self.s3, "bucket"))

    def test_round_trip_and_one_upload_per_item(self):
        saved = asyncio.run(self.media.dehydrate(SESSION, conversation()))
        asyncio.run(self.media.dehydrate(SESSION, conversation()))
        self.assertEqual(self.s3.puts, 3)
        self.assertNotIn(";base64,", json.dumps(saved))
        # A worker that didn't create the session reads the same objects.
        other_worker = SessionMedia(_S3Blobs(self.s3, "bucket"))
        self.assertEqual(asyncio.run(other_worker.hydrate(SESSION, saved)), conversation())

    def test_a_missing_object_is_not_a_read_error(self):
        saved = asyncio.run(self.media.dehydrate(SESSION, conversation()))
        self.s3.objects.clear()
        with self.assertLogs("ambient.session_media", "ERROR"):
            loaded = asyncio.run(self.media.hydrate(SESSION, saved))
        self.assertEqual(loaded[4]["content"], [{"type": "text", "text": MISSING_NOTE}])

    def test_delete_removes_only_that_session(self):
        asyncio.run(self.media.dehydrate(SESSION, conversation()))
        asyncio.run(self.media.dehydrate("ses_other", conversation()))
        asyncio.run(self.media.delete_session(SESSION))
        self.assertEqual(len(self.s3.objects), 3)
        self.assertTrue(all(k.startswith("session-media/ses_other/") for k in self.s3.objects))


class _FakePool:
    """Just enough of asyncpg for the messages column."""

    def __init__(self) -> None:
        self.messages: str | None = None

    async def execute(self, query, *args):
        if query.startswith("UPDATE sessions SET messages"):
            self.messages = args[0]
            return "UPDATE 1"
        return "DELETE 1"

    async def fetchrow(self, _query, _session_id):
        return {"messages": self.messages}


class StoreTests(Base):
    def setUp(self) -> None:
        super().setUp()
        self.store = Store(dsn="postgresql://unused")
        self.store._pool = _FakePool()
        self.store.media = self.media

    def test_the_row_holds_references_and_loading_restores_media(self):
        asyncio.run(self.store.save_messages(SESSION, conversation()))
        self.assertNotIn(";base64,", self.store._pool.messages)
        self.assertEqual(asyncio.run(self.store.get_messages(SESSION)), conversation())

    def test_deleting_the_session_deletes_its_media(self):
        asyncio.run(self.store.save_messages(SESSION, conversation()))
        self.assertTrue(asyncio.run(self.store.delete_session(SESSION)))
        self.assertEqual(self.files(), [])


if __name__ == "__main__":
    unittest.main()

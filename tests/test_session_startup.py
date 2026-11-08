"""Unit tests for the session-startup latency changes.

Covers the pure logic introduced by docs/session-startup-latency-plan.md:
  * SessionRunner seed/description-injection (Phase 1)
  * the yt-dlp height-capped format selector (Phase 2)

No server, DB, sandbox, or network — the SessionRunner is built with in-memory
fakes and only its message-shaping helpers are exercised.

Run: python -m unittest tests.test_session_startup
"""
import asyncio
import importlib.util
import os
import unittest

from ambient.server.runner import SessionRunner
from ambient.server.sandbox import SandboxLimits


class _FakeStore:
    """Minimal store: just get_file, returning a preset record."""

    def __init__(self, file_rec=None):
        self._file_rec = file_rec

    async def get_file(self, _video_id):
        return self._file_rec

    async def update_session(self, *_a, **_k):
        return {}


class _FakeBroker:
    pass


def _make_runner(*, video_id="vid_test", file_rec=None) -> SessionRunner:
    return SessionRunner(
        session_id="ses_test",
        video_id=video_id,
        model="test-model",
        max_turns_per_run=5,
        backend="inprocess",
        limits=SandboxLimits(),
        store=_FakeStore(file_rec),
        broker=_FakeBroker(),
    )


def _run(coro):
    return asyncio.run(coro)


class DescriptionInjectedTests(unittest.TestCase):
    def test_empty_conversation_not_injected(self):
        r = _make_runner()
        self.assertFalse(r._description_injected())

    def test_metadata_only_seed_not_injected(self):
        r = _make_runner()
        r.messages.append({"role": "user", "content": [{"type": "text", "text":
            "Video id: vid_test\nTitle: Foo\nDuration: 120 seconds\nNote: a detailed "
            "video description is being generated in the background."}]})
        self.assertFalse(r._description_injected())

    def test_full_seed_counts_as_injected(self):
        r = _make_runner()
        r.messages.append({"role": "user", "content": [{"type": "text", "text":
            "Video id: vid_test, Description: A long overview of the video."}]})
        self.assertTrue(r._description_injected())

    def test_system_update_marker_counts_as_injected(self):
        r = _make_runner()
        r.messages.append({"role": "user", "content": [{"type": "text", "text":
            f"{SessionRunner._DESC_INJECT_MARKER} now available:\nThe overview."}]})
        self.assertTrue(r._description_injected())

    def test_survives_rehydration_shape(self):
        # A rehydrated runner has video_description=None but the marker in messages.
        rec_messages = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": [{"type": "text", "text":
                f"{SessionRunner._DESC_INJECT_MARKER} now available:\nX"}]},
        ]
        r = _make_runner()
        r.messages = rec_messages
        self.assertIsNone(r.video_description)
        self.assertTrue(r._description_injected())


class BuildSeedTextTests(unittest.TestCase):
    def test_uses_full_description_when_ready(self):
        r = _make_runner()
        r.video_description = "The overview text."
        seed = _run(r._build_seed_text())
        self.assertIn("Video id: vid_test", seed)
        self.assertIn("Description: The overview text.", seed)

    def test_metadata_seed_from_youtube_row(self):
        rec = {"youtube": {"title": "My Clip", "duration": 321}, "filename": "x.mp4"}
        r = _make_runner(file_rec=rec)
        seed = _run(r._build_seed_text())
        self.assertIn("Title: My Clip", seed)
        self.assertIn("Duration: 321 seconds", seed)
        self.assertNotIn("Description:", seed)  # must not claim a description yet

    def test_metadata_seed_unknown_when_no_row(self):
        r = _make_runner(file_rec=None)
        seed = _run(r._build_seed_text())
        self.assertIn("Title: unknown", seed)
        self.assertIn("Duration: unknown seconds", seed)

    def test_seed_then_injection_are_consistent(self):
        # Metadata seed first (no description) -> not injected -> later injection sticks.
        rec = {"youtube": {"title": "T", "duration": 10}}
        r = _make_runner(file_rec=rec)
        seed = _run(r._build_seed_text())
        r.messages.append({"role": "user", "content": [{"type": "text", "text": seed}]})
        self.assertFalse(r._description_injected())
        r.video_description = "later overview"
        r.messages.append({"role": "user", "content": [{"type": "text", "text":
            f"{SessionRunner._DESC_INJECT_MARKER} now available:\n{r.video_description}"}]})
        self.assertTrue(r._description_injected())


class AwaitSourceReadyTests(unittest.TestCase):
    def test_non_youtube_is_immediately_ready(self):
        r = _make_runner(file_rec={"source_type": "upload"})
        _run(r._await_source_ready())
        self.assertTrue(r._source_ready)

    def test_missing_row_is_immediately_ready(self):
        r = _make_runner(file_rec=None)
        _run(r._await_source_ready())
        self.assertTrue(r._source_ready)

    def test_ready_youtube_source_no_wait(self):
        r = _make_runner(file_rec={"source_type": "youtube", "source_status": "ready"})
        called = []
        _run(r._await_source_ready(on_wait=lambda: called.append(1) or _noop()))
        self.assertTrue(r._source_ready)
        self.assertEqual(called, [])  # already ready -> on_wait never fired

    def test_failed_youtube_source_raises(self):
        r = _make_runner(file_rec={
            "source_type": "youtube", "source_status": "failed", "source_error": "boom",
        })
        with self.assertRaises(RuntimeError):
            _run(r._await_source_ready())


async def _noop():
    return None


class AwaitDescriptionTests(unittest.TestCase):
    def test_returns_immediately_when_no_task(self):
        r = _make_runner()
        self.assertIsNone(r._description_task)
        _run(r._await_description(timeout=0.01))  # must not hang or raise

    def test_returns_immediately_when_already_resolved(self):
        r = _make_runner()
        r.video_description = "ready"
        _run(r._await_description(timeout=0.01))

    def test_waits_for_pending_task_then_returns(self):
        async def scenario():
            r = _make_runner()

            async def resolver():
                await asyncio.sleep(0.02)
                r.video_description = "landed"

            r._description_task = asyncio.create_task(resolver())
            await r._await_description(timeout=5)
            return r.video_description

        self.assertEqual(_run(scenario()), "landed")

    def test_timeout_leaves_task_running(self):
        async def scenario():
            r = _make_runner()

            async def slow():
                await asyncio.sleep(5)
                r.video_description = "eventually"

            r._description_task = asyncio.create_task(slow())
            await r._await_description(timeout=0.02)  # times out
            running = not r._description_task.done()   # shielded -> still alive
            r._description_task.cancel()
            return running

        self.assertTrue(_run(scenario()))


class YoutubeFormatSelectorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        path = os.path.join(
            os.path.dirname(__file__), "..", "ambient", "sandboxes", "e2b",
            "templates", "video-analysis-v1", "main.py",
        )
        spec = importlib.util.spec_from_file_location("sbx_main", os.path.abspath(path))
        cls.main = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.main)

    def test_capped_includes_height_filter(self):
        fmt = self.main._youtube_format_selector(720)
        self.assertIn("[height<=720]", fmt)
        self.assertIn("vcodec^=avc1", fmt)          # avc1 preference preserved
        self.assertTrue(fmt.endswith("/b"))         # unfiltered last resort

    def test_uncapped_has_no_height_filter(self):
        fmt = self.main._youtube_format_selector(0)
        self.assertNotIn("height<=", fmt)
        self.assertTrue(fmt.endswith("/b"))

    def test_uncapped_matches_legacy_selector(self):
        # The original hardcoded selector, byte-for-byte (regression guard).
        legacy = "bv*[vcodec^=avc1][ext=mp4]+ba[ext=m4a]/bv*[ext=mp4]+ba[ext=m4a]/b[ext=mp4]/bv*+ba/b"
        self.assertEqual(self.main._youtube_format_selector(0), legacy)


if __name__ == "__main__":
    unittest.main()

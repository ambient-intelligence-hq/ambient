"""Unit tests for the `ambient` CLI helpers and in-memory adapters.

No LLM, network, DB, or ffmpeg — only the pure logic:
  * video-id sanitization, URL detection, local staging (symlink + collision hash)
  * environment layering (OpenRouter convenience, --model / --vision-model, forced
    local/hermetic backend)
  * output-schema normalization in ambient.agent
  * LiteStore / LiteBroker behavior

Run: python -m unittest tests.test_cli
"""
import asyncio
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from pydantic import BaseModel

from ambient import cli


def _run(coro):
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


class SanitizeAndUrlTests(unittest.TestCase):
    def test_sanitize_strips_glob_and_space_metacharacters(self):
        raw = "BALI： guide [eZ5] & more"
        out = cli._sanitize_id(raw)
        self.assertTrue(all(c.isalnum() or c in "_-" for c in out))
        self.assertNotIn(" ", out)
        self.assertNotIn("[", out)

    def test_sanitize_empty_falls_back(self):
        self.assertEqual(cli._sanitize_id("！！！"), "video")

    def test_is_url(self):
        self.assertTrue(cli._is_url("https://youtu.be/abc"))
        self.assertTrue(cli._is_url("http://example.com/v.mp4"))
        self.assertFalse(cli._is_url("/tmp/video.mp4"))
        self.assertFalse(cli._is_url("video.mp4"))


class StageLocalTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.ws = Path(self._tmp.name) / "ws"
        self._patch = mock.patch.object(cli, "WORKSPACE", self.ws)
        self._patch.start()

    def tearDown(self):
        self._patch.stop()
        self._tmp.cleanup()

    def _make_video(self, name: str) -> Path:
        p = Path(self._tmp.name) / name
        p.write_bytes(b"fake mp4 bytes")
        return p

    def test_stage_creates_symlink_with_preserved_ext(self):
        src = self._make_video("my clip.mp4")
        vid = cli._stage_local(src)
        staged = list(self.ws.glob(f"{vid}.*"))
        self.assertEqual(len(staged), 1)
        self.assertEqual(staged[0].suffix, ".mp4")
        self.assertTrue(staged[0].is_symlink() or staged[0].is_file())

    def test_same_path_stable_id_different_path_distinct(self):
        a = self._make_video("clip.mp4")
        sub = Path(self._tmp.name) / "sub"
        sub.mkdir()
        b = sub / "clip.mp4"
        b.write_bytes(b"other bytes")
        id_a1 = cli._stage_local(a)
        id_a2 = cli._stage_local(a)
        id_b = cli._stage_local(b)
        self.assertEqual(id_a1, id_a2)          # same source -> stable id
        self.assertNotEqual(id_a1, id_b)         # same stem, different path -> distinct

    def test_unsupported_extension_rejected(self):
        bad = self._make_video("notes.txt")
        with self.assertRaises(Exception):
            cli._stage_local(bad)

    def test_missing_file_rejected(self):
        with self.assertRaises(Exception):
            cli._stage_local(Path(self._tmp.name) / "nope.mp4")


class ApplyEnvTests(unittest.TestCase):
    _KEYS = ["OPENROUTER_API_KEY", "LLM_API_KEY", "AGENT_API_KEY", "LLM_BASE_URL",
             "AGENT_BASE_URL", "LLM_MODEL", "AGENT_MODEL", "SANDBOX_BACKEND",
             "SELF_VIDEO_ANALYSIS_TOOL", "S3_BUCKET", "VIDEO_FOLDER"]

    def setUp(self):
        self._saved = {k: os.environ.get(k) for k in self._KEYS}
        for k in self._KEYS:
            os.environ.pop(k, None)

    def tearDown(self):
        for k, v in self._saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v

    def test_openrouter_convenience(self):
        os.environ["OPENROUTER_API_KEY"] = "sk-or-test"
        cli._apply_env(None, None, Path("/tmp/ws"))
        self.assertEqual(os.environ["LLM_API_KEY"], "sk-or-test")
        self.assertEqual(os.environ["LLM_BASE_URL"], cli.OPENROUTER_BASE_URL)

    def test_explicit_base_url_not_overridden(self):
        os.environ["OPENROUTER_API_KEY"] = "sk-or-test"
        os.environ["LLM_BASE_URL"] = "http://localhost:8000/v1"
        cli._apply_env(None, None, Path("/tmp/ws"))
        self.assertEqual(os.environ["LLM_BASE_URL"], "http://localhost:8000/v1")

    def test_model_flag_sets_both_roles(self):
        cli._apply_env("org/model-x", None, Path("/tmp/ws"))
        self.assertEqual(os.environ["AGENT_MODEL"], "org/model-x")
        self.assertEqual(os.environ["LLM_MODEL"], "org/model-x")

    def test_vision_model_overrides_only_vision(self):
        cli._apply_env("org/agent", "org/vision", Path("/tmp/ws"))
        self.assertEqual(os.environ["AGENT_MODEL"], "org/agent")
        self.assertEqual(os.environ["LLM_MODEL"], "org/vision")

    def test_forces_hermetic_local_backend(self):
        cli._apply_env(None, None, Path("/tmp/ws"))
        self.assertEqual(os.environ["SANDBOX_BACKEND"], "host")
        self.assertNotEqual(os.environ["SANDBOX_BACKEND"], "e2b")
        self.assertEqual(os.environ["S3_BUCKET"], "")     # empty -> no-S3 frames fallback
        self.assertEqual(os.environ["VIDEO_FOLDER"], "/tmp/ws")

    def test_self_video_analysis_not_forced(self):
        # The CLI must honor the server's SELF_VIDEO_ANALYSIS_TOOL setting so the
        # agent workflow matches the UI — never override it here.
        os.environ["SELF_VIDEO_ANALYSIS_TOOL"] = "1"
        cli._apply_env(None, None, Path("/tmp/ws"))
        self.assertEqual(os.environ["SELF_VIDEO_ANALYSIS_TOOL"], "1")


class OutputSchemaTests(unittest.TestCase):
    def test_resolve_output_schema_variants(self):
        from ambient.agent import _resolve_output_schema

        class M(BaseModel):
            x: int

        self.assertIsNone(_resolve_output_schema(None))
        self.assertEqual(_resolve_output_schema({"type": "object"}), {"type": "object"})
        resolved = _resolve_output_schema(M)
        self.assertIsInstance(resolved, dict)
        self.assertIn("properties", resolved)


class LiteAdapterTests(unittest.TestCase):
    def test_store_session_event_file_flow(self):
        from ambient.lite import LiteStore

        async def scenario():
            s = LiteStore()
            await s.create_session({"session_id": "ses1", "usage": {}})
            self.assertIsNotNone(await s.get_session("ses1"))
            await s.update_session("ses1", lambda r: {**r, "status": "running"})
            self.assertEqual((await s.get_session("ses1"))["status"], "running")
            with self.assertRaises(KeyError):
                await s.update_session("missing", lambda r: r)

            e1 = await s.append_event("ses1", "run.started", {})
            e2 = await s.append_event("ses1", "turn.start", {})
            self.assertEqual(e2["seq"], e1["seq"] + 1)

            await s.put_file({"video_id": "v1", "source_status": "ready"})
            self.assertEqual((await s.get_file("v1"))["source_status"], "ready")
            self.assertIsNone(await s.get_file(None))

        _run(scenario())

    def test_broker_stops_only_on_close_not_run_completed(self):
        from ambient.lite import LiteBroker

        async def scenario():
            b = LiteBroker()
            # run.completed must NOT stop iteration — the trailing status change
            # after it still needs to drain before the queue is left empty.
            await b.publish("s", {"type": "run.started", "seq": 1})
            await b.publish("s", {"type": "run.completed", "seq": 2})
            await b.publish("s", {"type": "session.status_changed", "seq": 3})
            await b.publish("s", {"__close__": True, "seq": 0})
            seen = [ev["type"] async for ev in b.events()]
            self.assertEqual(seen, ["run.started", "run.completed", "session.status_changed"])

        _run(scenario())

    def test_broker_close_sentinel_stops_iteration(self):
        from ambient.lite import LiteBroker

        async def scenario():
            b = LiteBroker()
            await b.publish("s", {"type": "turn.start", "seq": 1})
            await b.publish("s", {"__close__": True, "seq": 0})
            seen = [ev.get("type") async for ev in b.events()]
            self.assertEqual(seen, ["turn.start"])

        _run(scenario())


if __name__ == "__main__":
    unittest.main()

"""Unit tests for the bash tool (host + e2b backend selection).

No LLM/network. Host tests shell out to real `/bin/sh`; the e2b path is exercised
with a fake media box to confirm routing + argument passing.

Run: python -m unittest tests.test_bash
"""
import asyncio
import tempfile
import unittest

from ambient.config import settings
from ambient.tools import bash_tool
from ambient.tools.bash_tool import bash
from ambient.tools.video_backend import current_media_box


def _run(coro):
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


class BashDisabledTests(unittest.TestCase):
    def test_disabled_by_default(self):
        prev = settings.enable_bash_tool
        settings.enable_bash_tool = False
        try:
            out, attachments = _run(bash("echo hi"))
            self.assertIn("disabled", out.lower())
            self.assertEqual(attachments, [])
        finally:
            settings.enable_bash_tool = prev


class BashHostTests(unittest.TestCase):
    def setUp(self):
        self._prev = settings.enable_bash_tool
        settings.enable_bash_tool = True
        # Commands run in a throwaway directory, not the configured video folder.
        self._prev_cwd = settings.bash_cwd
        self._cwd = tempfile.TemporaryDirectory()
        settings.bash_cwd = self._cwd.name

    def tearDown(self):
        settings.enable_bash_tool = self._prev
        settings.bash_cwd = self._prev_cwd
        self._cwd.cleanup()

    def test_echo_and_exit_code(self):
        out, _ = _run(bash("echo hello"))
        self.assertIn("exit_code: 0", out)
        self.assertIn("hello", out)

    def test_nonzero_exit_and_stderr(self):
        out, _ = _run(bash("ls /definitely/not/here"))
        self.assertNotIn("exit_code: 0", out)
        self.assertIn("stderr", out)

    def test_timeout_is_killed(self):
        out, _ = _run(bash("sleep 5", timeout=1))
        self.assertIn("exit_code: 124", out)
        self.assertIn("timed out", out)

    def test_output_is_byte_capped(self):
        prev = settings.bash_max_output_bytes
        settings.bash_max_output_bytes = 100
        try:
            out, _ = _run(bash("head -c 5000 /dev/zero | tr '\\0' 'a'"))
            self.assertIn("truncated", out)
        finally:
            settings.bash_max_output_bytes = prev


class _FakeBox:
    """Stands in for an active E2B media box."""
    def __init__(self):
        self.calls = []

    def run_shell(self, command, timeout=None):
        self.calls.append((command, timeout))
        return {"stdout": "from-sandbox", "stderr": "", "exit_code": 0}


class BashE2BRoutingTests(unittest.TestCase):
    def setUp(self):
        self._prev = settings.enable_bash_tool
        settings.enable_bash_tool = True

    def tearDown(self):
        settings.enable_bash_tool = self._prev

    def test_routes_to_active_media_box(self):
        box = _FakeBox()
        token = current_media_box.set(box)
        try:
            out, _ = _run(bash("uname -a", timeout=42))
        finally:
            current_media_box.reset(token)
        self.assertEqual(box.calls, [("uname -a", 42)])
        self.assertIn("from-sandbox", out)
        self.assertIn("exit_code: 0", out)


if __name__ == "__main__":
    unittest.main()

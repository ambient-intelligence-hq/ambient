"""E2BMediaSandbox gives each thread its own e2b Sandbox handle.

The e2b SDK keeps one HTTP/2 connection per thread, but a Sandbox object reuses
the connection of the thread that built it. Sharing one across the dispatcher's
worker threads made concurrent commands fail with
`httpcore.ReadError: [Errno 35] Resource temporarily unavailable`.

No network: the e2b Sandbox class is replaced with a fake.
"""
import threading
import unittest
from types import SimpleNamespace
from unittest import mock

from ambient.sandboxes.e2b import media_sandbox
from ambient.sandboxes.e2b.media_sandbox import RESULT_SENTINEL, E2BMediaSandbox


class _FakeSandbox:
    made: list = []

    def __init__(self, sandbox_id):
        self.sandbox_id = sandbox_id
        self.thread = threading.get_ident()
        self.commands = SimpleNamespace(run=self._run)
        _FakeSandbox.made.append(self)

    @classmethod
    def create(cls, template, timeout=None, envs=None):
        return cls("sbx_1")

    @classmethod
    def connect(cls, sandbox_id, timeout=None):
        return cls(sandbox_id)

    def set_timeout(self, _secs):
        pass

    def _run(self, cmd, **_kw):
        # A command must run on a handle built by the calling thread.
        assert self.thread == threading.get_ident(), "handle shared across threads"
        env = '{"ok": true, "data": {"thread": %d}}' % self.thread
        return SimpleNamespace(stdout=f"{RESULT_SENTINEL}{env}{RESULT_SENTINEL}", stderr="", exit_code=0)


class PerThreadHandleTests(unittest.TestCase):
    def setUp(self):
        _FakeSandbox.made = []
        patcher = mock.patch.object(media_sandbox, "Sandbox", _FakeSandbox)
        patcher.start()
        self.addCleanup(patcher.stop)
        self.box = E2BMediaSandbox()
        self.box.create(template="t")

    def test_creating_thread_reuses_its_handle(self):
        self.box.run(["probe"])
        self.box.run(["probe"])
        self.assertEqual(len(_FakeSandbox.made), 1)

    def test_each_thread_gets_its_own_handle_to_the_same_box(self):
        results, errors = [], []

        def work():
            try:
                results.append(self.box.run(["extract-frames"])["thread"])
                results.append(self.box.run(["extract-frames"])["thread"])
            except Exception as exc:  # noqa: BLE001
                errors.append(exc)

        threads = [threading.Thread(target=work) for _ in range(3)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        self.assertEqual(errors, [])
        self.assertEqual(len(results), 6)
        # One handle from create + one connect per worker thread, all to one box.
        self.assertEqual(len(_FakeSandbox.made), 4)
        self.assertEqual({s.sandbox_id for s in _FakeSandbox.made}, {"sbx_1"})

    def test_new_box_replaces_stale_thread_handles(self):
        self.box.run(["probe"])
        self.box.sandbox = _FakeSandbox("sbx_2")      # e.g. the box was recreated
        made_before = len(_FakeSandbox.made)
        self.box.run(["probe"])
        self.assertEqual(len(_FakeSandbox.made), made_before + 1)
        self.assertEqual(_FakeSandbox.made[-1].sandbox_id, "sbx_2")


if __name__ == "__main__":
    unittest.main()

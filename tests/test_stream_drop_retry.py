"""A provider that drops the stream mid-response (connection reset / truncated
chunked body) must not end the run: the runner re-requests that turn.

Seen on Jarvislabs: the gateway restarts every few minutes and cuts every open
stream at once ("Response payload is not completed"), which ended about half of
all multi-turn runs. llm_client only retries before the first chunk.
"""
from __future__ import annotations

import asyncio
import unittest
from unittest import mock

import aiohttp

from ambient.server import runner as runner_mod
from tests.test_run_budget import _answer, _runner, _tool_turn


def _run_with_drops(r, responses):
    """responses: list of chunk lists; an Exception instance in a list is raised
    at that point of the stream (after the chunks before it were yielded)."""
    calls = []

    async def stream(**kw):
        calls.append(kw)
        for c in responses.pop(0):
            if isinstance(c, BaseException):
                raise c
            yield c

    with mock.patch.object(runner_mod, "stream_chat_completion", stream), \
            mock.patch.object(runner_mod, "self_video_analysis_enabled", lambda _m: True), \
            mock.patch.object(runner_mod, "_STREAM_DROP_DELAYS", (0.0, 0.0)):
        asyncio.run(r._run_until_done(r.model))
    return calls


def _drop():
    return aiohttp.ClientPayloadError("Response payload is not completed")


class StreamDropRetryTests(unittest.TestCase):
    def test_dropped_turn_is_retried_and_the_run_finishes(self):
        r, events = _runner({})
        partial = _tool_turn(1, 1)[:1]   # a tool-call delta, then the connection dies
        calls = _run_with_drops(r, [partial + [_drop()], _tool_turn(1, 1), _answer('{"segments": []}')])
        self.assertEqual(len(calls), 3)           # dropped turn 1, its retry, the answer turn
        done = [p for t, p in events if t == "run.completed"]
        self.assertEqual(done[-1]["stop_reason"], "end_turn")
        self.assertEqual(done[-1]["answer"], '{"segments": []}')
        retries = [p for t, p in events if t == "tool.progress" and p["progress"].get("stream_retry")]
        self.assertEqual(len(retries), 1)
        # Only the retried attempt's assistant message is kept (one tool call, not two).
        tool_uses = [b for m in r.messages if m.get("role") == "assistant" and isinstance(m.get("content"), list)
                     for b in m["content"] if b.get("type") == "tool_use"]
        self.assertEqual(len(tool_uses), 1)

    def test_gives_up_after_repeated_drops(self):
        r, events = _runner({})
        _run_with_drops(r, [[_drop()], [_drop()], [_drop()]])
        done = [p for t, p in events if t == "run.completed"]
        self.assertEqual(done[-1]["stop_reason"], "error")
        self.assertTrue(any(t == "error" for t, _ in events))

    def test_other_errors_are_not_retried(self):
        r, events = _runner({})
        calls = _run_with_drops(r, [[RuntimeError("HTTP 400 from x: bad request")]])
        self.assertEqual(len(calls), 1)
        done = [p for t, p in events if t == "run.completed"]
        self.assertEqual(done[-1]["stop_reason"], "error")


if __name__ == "__main__":
    unittest.main()

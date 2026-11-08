"""Parallel tool calls must be answered contiguously in the OpenAI shape.

When a turn makes several tool calls and the tools attach media (clip frames),
the runner used to append each tool's result followed immediately by its media
user message: tool(A), user(media A), tool(B), user(media B). Strict providers
(Together) reject that, because every tool message of a parallel group has to
directly follow the assistant's tool_calls.
"""
from __future__ import annotations

import asyncio
import unittest
from types import SimpleNamespace

from ambient.server.runner import SessionRunner, _normalize_for_openai
from ambient.server.sandbox import SandboxLimits


class _FakeStore:
    async def get_file(self, _video_id):
        return None

    async def update_session(self, *_a, **_k):
        return {}


class _FakeDispatcher:
    async def call_tool(self, tool_use_id, name, tool_input):
        yield SimpleNamespace(type="result", data={
            "analysis": f"analysis for {tool_use_id}",
            "user_message_contents": [{"type": "image_url", "image_url": {"url": f"data:{tool_use_id}"}}],
        })


def _runner() -> SessionRunner:
    r = SessionRunner(
        session_id="ses_test", video_id="vid_test", model="test-model",
        max_turns_per_run=5, backend="inprocess", limits=SandboxLimits(),
        store=_FakeStore(), broker=object(),
    )
    r._dispatcher = _FakeDispatcher()

    async def _noop(*_a, **_k):
        return None

    r._emit = _noop
    r._record_usage = _noop
    r._await_source_ready = _noop
    return r


class ParallelToolResultOrderTests(unittest.TestCase):
    def test_tool_messages_contiguous_after_tool_calls(self):
        r = _runner()
        calls = [
            {"type": "tool_use", "id": "call_a", "name": "focus_clip", "input": {}},
            {"type": "tool_use", "id": "call_b", "name": "focus_clip", "input": {}},
        ]
        r.messages = [{"role": "user", "content": "q"}, {"role": "assistant", "content": calls}]
        asyncio.run(r._run_tool_calls("run_1", 1, calls))

        roles = [(m["role"], m.get("tool_call_id")) for m in _normalize_for_openai(r.messages)]
        self.assertEqual(roles[1:4], [("assistant", None), ("tool", "call_a"), ("tool", "call_b")])
        # Both tools' media still reach the model, after the tool messages.
        self.assertEqual([m for m, _ in roles[4:]], ["user", "user"])


if __name__ == "__main__":
    unittest.main()

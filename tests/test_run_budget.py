"""Per-run spend caps and the note that tells the agent how dense its frames are.

Runaway runs (WGO-Bench robot clips with the bash tool on) took 60-120 model
requests and up to 10M input tokens: the agent treated near-complete frame
coverage as a sparse overview, then cropped/upscaled frames with bash and opened
each one with read_image, one step per image. These tests cover the two engine
fixes: the density note on the context frames, and the per-run budget that ends
such a run with a real answer.

No network, DB or LLM: the store, broker, dispatcher and model stream are fakes.
"""
from __future__ import annotations

import asyncio
import json
import unittest
from types import SimpleNamespace
from unittest import mock

from ambient import fast_mode as fm
from ambient.server import runner as runner_mod
from ambient.server.runner import SessionRunner
from ambient.server.sandbox import SandboxLimits


class FramesContextNoteTests(unittest.TestCase):
    def test_dense_frames_at_source_resolution_say_answer_from_them(self):
        # robointer_droid_000030: 192 frames of a 23.2 s, 320x180, 10 fps clip.
        note = fm.frames_context_note(192, 23.2, {"width": 320, "height": 180, "fps": 10}, 768)
        self.assertIn("192 frames over 23.2 s", note)
        self.assertIn("320×180 at 10 fps", note)
        self.assertIn("83% of all its frames", note)
        self.assertIn("adds no detail", note)
        self.assertIn("Answer from these frames", note)

    def test_dense_frames_of_a_larger_source_do_not_rule_out_crops(self):
        note = fm.frames_context_note(192, 51.5, {"width": 1280, "height": 720, "fps": 30}, 768)
        self.assertIn("Answer from these frames", note)
        self.assertNotIn("adds no detail", note)

    def test_sparse_frames_say_use_tools(self):
        note = fm.frames_context_note(192, 3600.0, None, 768)
        self.assertIn("sparse", note)
        self.assertNotIn("Answer from these frames", note)

    def test_no_note_without_a_duration(self):
        self.assertIsNone(fm.frames_context_note(192, 0.0))

    def test_note_is_part_of_the_cached_frames_header(self):
        with mock.patch("ambient.llm.construct_payload", lambda frames: []):
            msg = fm.build_fast_frames_message([], note="N frames.")
        self.assertEqual(msg["content"][0]["text"],
                         "Frames sampled uniformly across the video (in time order):\nN frames.")


# --- runner budget -----------------------------------------------------------

class _Store:
    async def get_file(self, _video_id):
        return None

    async def update_session(self, *_a, **_k):
        return {}

    async def save_messages(self, *_a, **_k):
        return None


class _Broker:
    async def renew(self, *_a):
        return True

    async def release(self, *_a):
        return None


class _Dispatcher:
    def __init__(self):
        self.calls: list[str] = []

    async def call_tool(self, tool_use_id, name, tool_input):
        self.calls.append(name)
        yield SimpleNamespace(type="result", data={"analysis": f"saw {tool_use_id}"})


def _tool_turn(turn: int, n_calls: int, prompt_tokens: int = 100) -> list[dict]:
    calls = [{"index": i, "id": f"call_{turn}_{i}", "type": "function",
              "function": {"name": "grab_frames",
                           "arguments": json.dumps({"video_id": "v", "start_time": i, "end_time": i + 1})}}
             for i in range(n_calls)]
    return [{"choices": [{"index": 0, "delta": {"tool_calls": calls}, "finish_reason": None}]},
            {"choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls"}],
             "usage": {"prompt_tokens": prompt_tokens, "completion_tokens": 5}}]


def _answer(text: str) -> list[dict]:
    return [{"choices": [{"index": 0, "delta": {"content": text}, "finish_reason": None}]},
            {"choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]}]


def _runner(budget: dict) -> tuple[SessionRunner, list]:
    r = SessionRunner(
        session_id="ses_test", video_id="vid_test", model="m", max_turns_per_run=50,
        backend="inprocess", limits=SandboxLimits(), store=_Store(), broker=_Broker(),
        run_budget=budget,
    )
    events: list = []

    async def emit(t, p):
        events.append((t, p))

    async def noop(*_a, **_k):
        return None

    r._emit = emit
    for name in ("_record_usage", "_await_source_ready", "_set_status", "_seed_usage_once",
                 "_persist_usage", "_await_sandbox_ready", "_await_description", "_attach_video_context"):
        setattr(r, name, noop)
    r._has_video_context = lambda: True
    r._description_injected = lambda: True
    r._dispatcher = _Dispatcher()
    r.messages.append({"role": "user", "content": [{"type": "text", "text": "segment the video"}]})
    return r, events


def _run(r: SessionRunner, responses: list[list[dict]]) -> list[dict]:
    seen: list[dict] = []

    async def stream(**kw):
        seen.append(kw)
        for c in responses.pop(0):
            yield c

    with mock.patch.object(runner_mod, "stream_chat_completion", stream), \
            mock.patch.object(runner_mod, "self_video_analysis_enabled", lambda _m: True):
        asyncio.run(r._run_until_done(r.model))
    return seen


def _texts(r: SessionRunner) -> list[str]:
    return [b.get("text", "") for m in r.messages if m.get("role") == "user"
            and isinstance(m.get("content"), list) for b in m["content"] if b.get("type") == "text"]


class RunBudgetTests(unittest.TestCase):
    def test_tool_call_cap_forces_a_final_answer(self):
        r, events = _runner({"max_tool_calls_per_run": 4})
        # The agent would keep calling tools forever; the cap ends it after 4 calls.
        responses = [_tool_turn(t, 2) for t in range(1, 10)]
        responses.insert(2, _answer('{"segments": []}'))   # the forced, tool-free answer
        seen = _run(r, responses)
        self.assertEqual(len(r._dispatcher.calls), 4)
        self.assertEqual(len(seen), 3)
        self.assertEqual(seen[-1]["tools"], [])            # no tools on the forced turn
        done = [p for t, p in events if t == "run.completed"][-1]
        self.assertEqual(done["stop_reason"], "max_tool_calls")
        self.assertEqual(done["answer"], '{"segments": []}')

    def test_agent_is_warned_once_before_the_cap(self):
        r, _ = _runner({"max_tool_calls_per_run": 12})
        responses = [_tool_turn(t, 2) for t in range(1, 6)] + [_answer("done")]
        _run(r, responses)
        warnings = [t for t in _texts(r) if t.startswith("Run budget:")]
        self.assertEqual(len(warnings), 1)
        self.assertIn("10 of 12 tool calls used", warnings[0])   # 2 left <= max(3, 12 // 4)

    def test_input_token_cap_forces_a_final_answer(self):
        r, events = _runner({"max_tool_calls_per_run": 0, "max_input_tokens_per_run": 1000})
        responses = [_tool_turn(1, 1, 600), _tool_turn(2, 1, 600), _answer("final")]
        _run(r, responses)
        done = [p for t, p in events if t == "run.completed"][-1]
        self.assertEqual(done["stop_reason"], "max_input_tokens")
        self.assertEqual(done["answer"], "final")
        self.assertEqual(len(r._dispatcher.calls), 2)

    def test_no_caps_leaves_the_run_alone(self):
        r, events = _runner({"max_tool_calls_per_run": 0, "max_input_tokens_per_run": 0})
        responses = [_tool_turn(t, 3) for t in range(1, 6)] + [_answer("done")]
        _run(r, responses)
        self.assertEqual(len(r._dispatcher.calls), 15)
        done = [p for t, p in events if t == "run.completed"][-1]
        self.assertEqual(done["stop_reason"], "end_turn")
        self.assertFalse([t for t in _texts(r) if t.startswith("Run budget:")])

    def test_budget_counts_per_run_not_per_session(self):
        r, events = _runner({"max_tool_calls_per_run": 4})
        _run(r, [_tool_turn(1, 3), _answer("first")])
        r.messages.append({"role": "user", "content": [{"type": "text", "text": "follow-up"}]})
        _run(r, [_tool_turn(2, 3), _answer("second")])
        stops = [p["stop_reason"] for t, p in events if t == "run.completed"]
        self.assertEqual(stops, ["end_turn", "end_turn"])


if __name__ == "__main__":
    unittest.main()

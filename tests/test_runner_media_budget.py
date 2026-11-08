"""Runner wiring for the media budget, the kill switch, and the empty-answer
fallback when a turn runs out of output tokens."""
from __future__ import annotations

import asyncio
import importlib
import unittest
from types import SimpleNamespace
from unittest import mock

from ambient import media_budget as mb
from ambient.config import settings
from ambient.media_plan import image_block_cap
from ambient.server import runner as runner_mod
from ambient.server.runner import SessionRunner
from ambient.server.sandbox import SandboxLimits

FIREWORKS = "https://api.fireworks.ai/inference/v1"
JARVISLABS = "https://models.jarvislabs.net/v1"


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
    def __init__(self, n_frames: int):
        self.n = n_frames

    async def call_tool(self, tool_use_id, name, tool_input):
        yield SimpleNamespace(type="result", data={
            "analysis": f"notes {tool_use_id}",
            "user_message_contents": [{"type": "image_url", "image_url": {"url": "https://s3/f.jpg"}}] * self.n,
        })


def _runner(model="accounts/fireworks/models/kimi-k3", base_url=FIREWORKS) -> tuple[SessionRunner, list]:
    r = SessionRunner(
        session_id="ses_test", video_id="vid_test", model=model, max_turns_per_run=5,
        backend="inprocess", limits=SandboxLimits(), store=_Store(), broker=_Broker(),
        base_url=base_url,
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
    return r, events


def _chunks(content: str = "", reasoning: str = "", finish: str = "stop") -> list[dict]:
    delta = {}
    if reasoning:
        delta["reasoning"] = reasoning
    if content:
        delta["content"] = content
    return [{"choices": [{"index": 0, "delta": delta, "finish_reason": None}]},
            {"choices": [{"index": 0, "delta": {}, "finish_reason": finish}]}]


def _fake_stream(responses: list[list[dict]], seen: list):
    async def stream(**kw):
        seen.append(kw)
        for c in responses.pop(0):
            yield c
    return stream


class LengthFallbackTests(unittest.TestCase):
    def test_reasoning_only_length_turn_gets_an_answer(self):
        r, events = _runner()
        r.messages.append({"role": "user", "content": [{"type": "text", "text": "List key moments"}]})
        seen: list = []
        stream = _fake_stream([_chunks(reasoning="thinking " * 50, finish="length"),
                               _chunks(content="1. 00:10 intro")], seen)
        with mock.patch.object(runner_mod, "stream_chat_completion", stream):
            asyncio.run(r._run_until_done(r.model))
        done = [p for t, p in events if t == "run.completed"][-1]
        self.assertEqual(done["answer"], "1. 00:10 intro")
        self.assertEqual(done["stop_reason"], "end_turn")
        self.assertEqual(seen[1]["tools"], [])
        self.assertFalse(seen[1]["reasoning_enabled"])

    def test_length_with_text_is_kept(self):
        r, events = _runner()
        r.messages.append({"role": "user", "content": [{"type": "text", "text": "q"}]})
        seen: list = []
        with mock.patch.object(runner_mod, "stream_chat_completion",
                               _fake_stream([_chunks(content="partial answer", finish="length")], seen)):
            asyncio.run(r._run_until_done(r.model))
        self.assertEqual(len(seen), 1)
        self.assertEqual([p for t, p in events if t == "run.completed"][-1]["answer"], "partial answer")


class RunCompletionOrderTests(unittest.TestCase):
    def test_lease_released_before_run_completed(self):
        """Clients send the next message as soon as they see run.completed
        (session.status_idle); the run lease must already be free by then."""
        r, events = _runner()
        order: list = []

        async def release(*_a):
            order.append("release")

        async def emit(t, p):
            if t == "run.completed":
                order.append("completed")
        r.broker.release = release
        r._emit = emit
        r.messages.append({"role": "user", "content": [{"type": "text", "text": "q"}]})
        with mock.patch.object(runner_mod, "stream_chat_completion",
                               _fake_stream([_chunks(content="answer")], [])):
            asyncio.run(r._run_until_done(r.model))
        self.assertEqual(order, ["release", "completed"])


class UnexpectedFailureTests(unittest.TestCase):
    def test_run_that_raises_still_completes(self):
        """A run that raises (e.g. a DB timeout under load) used to die silently,
        leaving the client on session.status_running forever."""
        r, events = _runner()

        async def boom(*_a, **_k):
            raise TimeoutError("statement timeout")

        async def save_fails(*_a, **_k):
            raise ConnectionError("db gone")
        r._prepare_messages = boom
        r.store.save_messages = save_fails                 # cleanup fails too
        r.messages.append({"role": "user", "content": [{"type": "text", "text": "q"}]})
        with self.assertLogs("ambient.server.runner", level="ERROR"):
            asyncio.run(r._run_until_done(r.model))
        types = [t for t, _ in events]
        self.assertIn("error", types)
        done = [p for t, p in events if t == "run.completed"]
        self.assertEqual(len(done), 1)
        self.assertEqual(done[0]["stop_reason"], "error")


class AgentEndpointTests(unittest.TestCase):
    """Clips that go into the agent's conversation are sized for the agent's endpoint,
    not the LLM_* vision endpoint (they differ with AGENT_* or a per-agent base_url)."""

    def test_clip_cap_follows_the_session_endpoint(self):
        sfc = importlib.import_module("ambient.tools.self_focus_clip")
        seen = {}

        async def produce(video_tools, step, start, end, *, model=None, base_url=None):
            seen.update(step=step, model=model, base_url=base_url)
            return "frames", [], start, None

        r, _ = _runner()                                   # agent on Fireworks; LLM_* untouched
        real_call_tool = r._dispatcher.call_tool

        async def tool(video_id, start_time, end_time):
            return await sfc.self_focus_clip(video_id, start_time, end_time)
        registry = {"focus_clip": tool}

        async def go():
            r._dispatcher.agent_endpoint = (r.model, FIREWORKS)
            async for ev in real_call_tool("c0", "focus_clip", {"start_time": 0, "end_time": 300}):
                if ev.type == "failed":
                    self.fail(ev.data["message"])

        with mock.patch.dict("ambient.tools.TOOL_REGISTRY", registry), \
                mock.patch("ambient.tools.clip_media.produce_window_media", produce), \
                mock.patch.object(sfc, "resolve_video_tools", lambda *a, **k: None):
            asyncio.run(go())
        self.assertEqual(seen, {"step": "self_focus_clip", "model": r.model, "base_url": FIREWORKS})

    def test_run_publishes_its_endpoint_to_the_dispatcher(self):
        r, _ = _runner()
        r.messages.append({"role": "user", "content": [{"type": "text", "text": "q"}]})
        with mock.patch.object(runner_mod, "stream_chat_completion",
                               _fake_stream([_chunks(content="answer")], [])):
            asyncio.run(r._run_until_done("some/override-model"))
        self.assertEqual(r._dispatcher.agent_endpoint, ("some/override-model", FIREWORKS))

    def test_without_a_session_the_vision_endpoint_is_used(self):
        from ambient.tools.video_backend import current_agent_endpoint
        self.assertIsNone(current_agent_endpoint.get())

    def test_endpoint_sets_the_clip_cap(self):
        from ambient.media_plan import plan_media
        fw = plan_media("self_focus_clip", FIREWORKS, "accounts/fireworks/models/kimi-k3", 300)
        modal = plan_media("self_focus_clip", "https://x--ep-kimi-k3-server.us-west.modal.direct/v1",
                           "moonshotai/Kimi-K3", 300)
        self.assertEqual(fw.max_frames, 47)
        self.assertEqual(modal.max_frames, 192)


SELF_HOSTED = "http://10.0.0.5:8000/v1"


class VideoCountLimitTests(unittest.TestCase):
    """Some deployments take one video per request (Jarvislabs Qwen-fp8, Gemma): a second
    is rejected outright, so clips over the limit must not reach a request."""
    MODEL = "qwen-3.8-27b-fp8"

    def _runner(self, context=None):
        r, events = _runner(model=self.MODEL, base_url=JARVISLABS)
        r._run_model = self.MODEL
        r._dispatcher = _VideoClips()
        r.messages.insert(1, context or {"role": "user", "content": [
            {"type": "text", "text": "Frames"}, {"type": "image_url", "image_url": {"url": "u"}}]})
        return r, events

    def _clips(self, *starts):
        return [{"type": "tool_use", "id": f"c{s}", "name": "focus_clip",
                 "input": {"start_time": s, "end_time": s + 60}} for s in starts]

    def _videos(self, r):
        return sum(mb.message_cost(m).videos for m in r.messages)

    def test_budget_knows_the_limit(self):
        self.assertEqual(mb.budget_for(JARVISLABS, self.MODEL).videos, 1)
        self.assertIsNone(mb.budget_for(FIREWORKS, "accounts/fireworks/models/kimi-k3").videos)

    def test_only_the_first_clip_runs_and_the_rest_say_why(self):
        r, events = self._runner()
        calls = self._clips(0, 300, 600)
        r.messages.append({"role": "assistant", "content": calls})
        asyncio.run(r._run_tool_calls("run_1", 1, calls))
        self.assertEqual([c[0] for c in r._dispatcher.calls], ["c0"])
        results = {p["tool_use_id"]: p["analysis"] for t, p in events if t == "tool.result"}
        self.assertEqual(results["c0"], "clip")
        self.assertIn("300-360 s in a later turn", results["c300"])
        self.assertEqual(self._videos(r), 1)
        # Every tool call still gets its answer, in the shape strict providers require.
        roles = [m["role"] for m in runner_mod._normalize_for_openai(r.messages)]
        self.assertEqual(roles[-4:], ["tool", "tool", "tool", "user"])

    def test_next_clip_replaces_the_previous_one(self):
        r, _ = self._runner()
        for s in (0, 300, 600):
            calls = self._clips(s)
            r.messages.append({"role": "assistant", "content": calls})
            asyncio.run(r._run_tool_calls("run_1", 1, calls))
            asyncio.run(r._prepare_messages("run_1", 2, r.model))
            self.assertEqual(self._videos(r), 1)
            self.assertEqual(mb.message_cost(r.messages[-1]).videos, 1)     # the newest one stays

    def test_video_context_fills_the_only_slot(self):
        ctx = {"role": "user", mb.FRAMES_KEY: 32, "content": [
            {"type": "text", "text": "The full video"}, {"type": "video_url", "video_url": {"url": "u"}}]}
        r, events = self._runner(ctx)
        calls = self._clips(0)
        r.messages.append({"role": "assistant", "content": calls})
        asyncio.run(r._run_tool_calls("run_1", 1, calls))
        self.assertEqual(r._dispatcher.calls, [])
        self.assertIn("grab_frames", [p for t, p in events if t == "tool.result"][0]["analysis"])
        self.assertEqual(self._videos(r), 1)

    def test_routes_without_a_limit_run_every_clip(self):
        r, _ = _runner()
        self.assertEqual(r._deferred_clips(self._clips(0, 300, 600)), {})


class _VideoClips:
    def __init__(self):
        self.calls = []

    async def call_tool(self, tool_use_id, name, tool_input):
        self.calls.append((tool_use_id, name))
        s, e = tool_input["start_time"], tool_input["end_time"]
        yield SimpleNamespace(type="result", data={"analysis": "clip", "user_message_contents": [
            {"type": "text", "text": f"Clip of the video between {s:.1f} and {e:.1f} seconds"},
            {"type": "video_url", "video_url": {"url": f"https://s3/{s}.mp4"}}]})


class VideoContextCountTests(unittest.TestCase):
    """Fix #4: the budget counts a whole-video context as what it really adds."""

    def test_self_hosted_walkthrough_stays_within_budget(self):
        from ambient.fast_mode import build_fast_video_message
        from ambient.media_plan import plan_media
        model = "qwen/qwen3.8-27b"
        budget = mb.budget_for(SELF_HOSTED, model)
        plan = plan_media("context", SELF_HOSTED, model, 1276)
        ctx = build_fast_video_message("https://s3/x.mp4", video_id="vid")
        ctx[mb.FRAMES_KEY] = mb.video_context_frames(SELF_HOSTED, model, plan, 1276, budget)
        self.assertEqual(mb.message_cost(ctx).frames, 500)
        r, _ = _runner(model=model, base_url=SELF_HOSTED)
        r.messages.insert(1, ctx)
        r.messages.append({"role": "user", "content": [{"type": "text", "text": "q"}]})
        for i in range(4):                                # one 60 s clip a turn (120 frames)
            r.messages.append({"role": "assistant", "content": [{"type": "text", "text": "t"}]})
            r.messages.append({"role": "user", "content": [
                {"type": "text", "text": f"Clip of the video between {i * 60}.0 and {i * 60 + 60}.0 seconds"},
                {"type": "video_url", "video_url": {"url": "https://s3/c.mp4"}}]})
            mb.fit_media_budget(r.messages, budget)
            total = sum(mb.message_cost(m).frames for m in r.messages)
            self.assertEqual(total, 620)                                      # context + only the new clip
            self.assertEqual(mb.message_cost(r.messages[-1]).frames, 120)     # the new clip survives
            self.assertLess(total * 768 * 432 * 3, 715_827_880)               # vLLM's decoded limit

    def test_uses_the_measured_rate_when_there_is_one(self):
        from ambient.capabilities import Capabilities
        from ambient.media_plan import MediaPlan
        plan = MediaPlan("video", fps=0.4, max_frames=500)
        budget = mb.MediaBudget(images=500, frames=600, inline_bytes=10**9)
        capped = Capabilities(entry_id="x", measured_at="", capped=True, frames_per_video=32, frames_per_min=6.2)
        steady = Capabilities(entry_id="y", measured_at="", frames_per_min=12.0)
        with mock.patch("ambient.capabilities.resolve", lambda *_: capped):
            self.assertEqual(mb.video_context_frames("u", "m", plan, 1276, budget), 32)
        with mock.patch("ambient.capabilities.resolve", lambda *_: steady):
            self.assertEqual(mb.video_context_frames("u", "m", plan, 1200, budget), 240)
        with mock.patch("ambient.capabilities.resolve", lambda *_: None):
            self.assertEqual(mb.video_context_frames("u", "m", plan, 1276, budget), 500)

    def test_a_huge_context_still_leaves_clips_some_room(self):
        from ambient.capabilities import Capabilities
        from ambient.media_plan import MediaPlan
        budget = mb.MediaBudget(images=500, frames=600, inline_bytes=10**9)
        dense = Capabilities(entry_id="x", measured_at="", frames_per_min=120.0)
        with mock.patch("ambient.capabilities.resolve", lambda *_: dense):
            n = mb.video_context_frames("u", "m", MediaPlan("video", fps=2, max_frames=500), 1276, budget)
        self.assertEqual(n, 510)                          # not 2,552

    def test_bookkeeping_field_never_reaches_the_provider(self):
        msg = {"role": "user", mb.FRAMES_KEY: 500, "content": [{"type": "video_url", "video_url": {"url": "u"}}]}
        out = runner_mod._normalize_for_openai([msg])
        self.assertNotIn(mb.FRAMES_KEY, out[0])
        self.assertIn(mb.FRAMES_KEY, msg)                 # still on the persisted message

    def test_removed_clip_stops_counting(self):
        msg = {"role": "user", mb.FRAMES_KEY: 60, "content": [
            {"type": "text", "text": "Clip of the video between 0.0 and 60.0 seconds"},
            {"type": "video_url", "video_url": {"url": "u"}}]}
        self.assertEqual(mb.message_cost(msg).frames, 60)
        self.assertEqual(mb.message_cost(mb._placeholder(msg)).frames, 0)

    def test_small_image_limit_does_not_bound_video_frames(self):
        b = mb.budget_for(JARVISLABS, "qwen-3.8-27b-fp8")  # 4 images, but it takes video
        self.assertEqual((b.images, b.frames), (3, 600))


class RunnerBudgetTests(unittest.TestCase):
    def _session(self, n_frames: int, rounds: int) -> tuple[SessionRunner, list]:
        r, events = _runner()
        r._dispatcher = _Dispatcher(n_frames)
        r.messages.insert(1, {"role": "user", "content": [{"type": "text", "text": "Frames sampled"},
                                                          *[{"type": "image_url", "image_url": {"url": "u"}}] * 47]})
        for i in range(rounds):
            call = {"type": "tool_use", "id": f"c{i}", "name": "focus_clip",
                    "input": {"start_time": i * 60, "end_time": i * 60 + 30}}
            r.messages.append({"role": "assistant", "content": [call]})
            asyncio.run(r._run_tool_calls("run_1", i + 1, [call]))
        return r, events

    def test_inline_frames_stay_out_of_events(self):
        r, events = _runner()

        class D:
            async def call_tool(self, tool_use_id, name, tool_input):
                yield SimpleNamespace(type="result", data={
                    "analysis": "a", "user_message_contents": [],
                    "attachments": [{"type": "image_url", "url": "data:image/jpeg;base64,AAAA"},
                                    {"type": "image_url", "url": "https://s3/f.jpg"}]})
        r._dispatcher = D()
        call = {"type": "tool_use", "id": "c0", "name": "focus_clip", "input": {}}
        asyncio.run(r._run_tool_calls("run_1", 1, [call]))
        res = [p for t, p in events if t == "tool.result"][0]
        self.assertEqual(res["attachments"], [{"type": "image_url", "url": "https://s3/f.jpg"}])

    def test_tool_media_is_labeled_with_its_range(self):
        r, _ = self._session(5, 1)
        self.assertEqual(r.messages[-1]["content"][0]["text"],
                         "Media from focus_clip between 0.0 and 30.0 seconds:")

    def test_prepare_fits_and_records(self):
        r, events = self._session(40, 3)                    # 47 + 3 x 40 images vs 95
        prepared = asyncio.run(r._prepare_messages("run_1", 4, r.model))
        images = sum(1 for m in prepared if isinstance(m.get("content"), list)
                     for b in m["content"] if b.get("type") == "image_url")
        self.assertLessEqual(images, mb.budget_for(FIREWORKS, r.model).images)
        self.assertEqual(r.usage["media_budget"]["reductions"], 1)
        progress = [p for t, p in events if t == "tool.progress"]
        self.assertIn("media_budget", progress[-1]["progress"])
        # Reductions are persisted: the next turn sends the same messages, no new edit.
        asyncio.run(r._prepare_messages("run_1", 5, r.model))
        self.assertEqual(r.usage["media_budget"]["reductions"], 1)

    def test_churn_nudge_once_per_run(self):
        r, _ = self._session(40, 2)
        for i in range(12):                                 # a new 40-frame window every turn
            call = {"type": "tool_use", "id": f"n{i}", "name": "focus_clip",
                    "input": {"start_time": 200 + i * 30, "end_time": 220 + i * 30}}
            r.messages.append({"role": "assistant", "content": [call]})
            asyncio.run(r._run_tool_calls("run_1", i + 3, [call]))
            asyncio.run(r._prepare_messages("run_1", i + 3, r.model))
        nudges = [m for m in r.messages if m["role"] == "user" and isinstance(m["content"], list)
                  and m["content"][0].get("text") == runner_mod._CHURN_NUDGE]
        self.assertGreaterEqual(r.usage["media_budget"]["reductions"], runner_mod._CHURN_REDUCTIONS)
        self.assertEqual(len(nudges), 1)

    def test_kill_switch_restores_old_behavior(self):
        with mock.patch.object(settings, "media_budget_enabled", False):
            r, events = self._session(40, 3)
            before = [dict(m) for m in r.messages]
            asyncio.run(r._prepare_messages("run_1", 4, r.model))
            self.assertEqual(r.messages, before)            # nothing rewritten
            self.assertNotIn("media_budget", r.usage)
            self.assertEqual(image_block_cap(JARVISLABS, "deepseek-v4.1-flash-1017"), 100)
        self.assertEqual(image_block_cap(JARVISLABS, "deepseek-v4.1-flash-1017"), 384)


if __name__ == "__main__":
    unittest.main()

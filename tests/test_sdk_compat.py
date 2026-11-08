"""SDK compatibility regression tests for the Managed Agents route.

The server speaks the Anthropic **Managed Agents** protocol; the official SDK
(`client.beta.agents` / `client.beta.sessions`) drives it directly. These tests
pin our route's output to the *installed* anthropic SDK's own models, so they
fail the moment either side drifts out of the contract:

  * our mapping changes      -> `_to_managed_events` / `_*_resource` emit a shape
                                the SDK can't decode;
  * the pinned SDK bumps     -> a member leaves the stream-event union, or a
                                resource model gains a required field.

Everything here is offline — no server, DB, network, or LLM. It exercises the
pure mapping functions and parses the results with the SDK's own constructors.

The one thing these can't see is a change to the *wire framing* in
`stream_events` (the SSE `event:` name). That contract — "SSE event name ==
managed event `type`, and the SDK silently drops names outside its allowlist" —
is covered structurally by `test_stream_replay.py`; here we guarantee the
`type` values themselves stay SDK-decodable.
"""
from __future__ import annotations

import asyncio
import re
import typing
import unittest
from pathlib import Path

from anthropic._models import construct_type
from anthropic.types.beta import BetaManagedAgentsSession, BetaManagedAgentsAgent
from anthropic.types.beta.sessions.beta_managed_agents_stream_session_events import (
    BetaManagedAgentsStreamSessionEvents as EVENT_UNION,
)

from ambient.config import settings
from ambient.server import runner as runner_mod
from ambient.server.errors import APIError
from ambient.server.routes import managed_agents as ma
from ambient.server.routes.managed_agents import (
    _to_managed_events,
    _session_resource,
    _agent_resource,
)


def _sdk_event_types() -> set[str]:
    """Discriminator literals the installed SDK can decode into a stream event.

    `EVENT_UNION` is `Annotated[Union[...], PropertyInfo(discriminator="type")]`;
    each member fixes `type` to a `Literal[...]`.
    """
    union = typing.get_args(EVENT_UNION)[0]
    out: set[str] = set()
    for member in typing.get_args(union):
        field = member.model_fields.get("type")
        if field is None:
            continue
        lit = typing.get_args(field.annotation)
        out.add(lit[0] if lit else field.annotation)
    return out


SDK_EVENT_TYPES = _sdk_event_types()

# Managed event types we emit on purpose that are NOT in the SDK union. The SDK's
# stream decoder (`Stream.__stream__`) silently drops SSE event names outside its
# allowlist — no else/raise — so these are safe extras for richer non-SDK clients
# (the Studio/demo UI renders them for token-by-token streaming).
CUSTOM_NON_SDK_TYPES = {"agent.reasoning.delta", "agent.message.delta"}

_TS = "2026-10-26T00:00:00Z"

# One representative runner event per type that `_to_managed_events` MAPS to
# output. Keep this in sync with the branches in `_to_managed_events`; the
# coverage test below fails if a newly emitted runner event type is missing here
# (or from INTENTIONALLY_DROPPED).
MAPPED_SAMPLES: dict[str, dict] = {
    "run.started": {"run_id": "run_1", "model": "m"},
    "agent.reasoning": {"reasoning": "thinking"},
    "span.model_request_end": {
        "run_id": "run_1", "model": "m", "source": "agent", "cost": 0.01,
        "usage": {"input_tokens": 1, "output_tokens": 1,
                  "cache_read_input_tokens": 0, "cache_creation_input_tokens": 0},
    },
    "error": {"message": "bad"},
    "tool.scheduled": {"tool_use_id": "tu_1", "name": "search_clip", "input": {}},
    "tool.result": {"tool_use_id": "tu_1", "analysis": "ok", "attachments": []},
    "tool.failed": {"tool_use_id": "tu_2", "error": {"message": "boom"}},
    "run.completed": {"run_id": "run_1", "stop_reason": "end_turn", "answer": "a", "usage": {}},
    "user.message": {"content": "hi"},
    "chat.completion.chunk": {"choices": [{"delta": {"content": "t", "reasoning": "r"}}]},
}

# Runner event types `_to_managed_events` intentionally DROPS (no SDK counterpart,
# nothing SDK-visible to carry them).
INTENTIONALLY_DROPPED = {
    "session.status_changed",
    "tool.progress",
    "tool.started",
    "turn.start",
    "turn.end",
}


class TestStreamEventSDKCompat(unittest.TestCase):
    def test_mapped_events_parse_into_sdk_union(self):
        """Each SDK-targeted managed event must construct into the SDK stream-event
        union with a matching discriminator."""
        for rtype, payload in MAPPED_SAMPLES.items():
            produced = _to_managed_events({"type": rtype, "payload": payload, "ts": _TS})
            self.assertTrue(produced, f"{rtype} produced no managed events")
            for me in produced:
                mtype = me["type"]
                if mtype in CUSTOM_NON_SDK_TYPES:
                    continue
                self.assertIn(
                    mtype, SDK_EVENT_TYPES,
                    f"{rtype} -> {mtype!r} is not an SDK-decodable event type",
                )
                obj = construct_type(type_=EVENT_UNION, value=dict(me))
                self.assertEqual(
                    getattr(obj, "type", None), mtype,
                    f"{mtype!r} did not round-trip through the SDK union discriminator",
                )

    def test_no_unknown_managed_types(self):
        """Every managed type we emit is either SDK-known or an explicitly
        whitelisted custom extra — never an accidental unknown."""
        for rtype, payload in MAPPED_SAMPLES.items():
            for me in _to_managed_events({"type": rtype, "payload": payload, "ts": _TS}):
                self.assertIn(
                    me["type"], SDK_EVENT_TYPES | CUSTOM_NON_SDK_TYPES,
                    f"{rtype} emits {me['type']!r}, neither SDK-known nor whitelisted "
                    f"(add to CUSTOM_NON_SDK_TYPES only if the SDK should drop it)",
                )

    def test_internal_error_is_remapped_not_raw(self):
        """Our internal `error` must surface as `session.error`, never a bare
        `error` — the SDK's `error` SSE branch *raises* an APIError."""
        [me] = _to_managed_events({"type": "error", "payload": {"message": "x"}, "ts": _TS})
        self.assertEqual(me["type"], "session.error")

    def test_dropped_events_produce_nothing(self):
        for rtype in INTENTIONALLY_DROPPED:
            self.assertEqual(
                _to_managed_events({"type": rtype, "payload": {}, "ts": _TS}), [],
                f"{rtype} was expected to be dropped but produced output",
            )


class TestRunnerEmitCoverage(unittest.TestCase):
    """Forcing function: every event type the runner/route can put on the stream
    must be accounted for — mapped (MAPPED_SAMPLES) or dropped
    (INTENTIONALLY_DROPPED). Adding a new `_emit("x.y")` without updating
    `_to_managed_events` and this file fails here, before it can silently ship an
    SDK-undecodable event."""

    def _emitted_types(self) -> set[str]:
        # Runner emits via `self._emit("type", ...)`; the route appends one event
        # type directly to the log (`append_event(session_id, "user.message", ...)`).
        runner_src = Path(runner_mod.__file__).read_text()
        route_src = Path(ma.__file__).read_text()
        emitted = set(re.findall(r'_emit\(\s*"([a-z0-9_.]+)"', runner_src))
        emitted |= set(re.findall(r'append_event\([^,]+,\s*"([a-z0-9_.]+)"', route_src))
        return emitted

    def test_all_emitted_types_handled(self):
        emitted = self._emitted_types()
        known = set(MAPPED_SAMPLES) | INTENTIONALLY_DROPPED
        unhandled = emitted - known
        self.assertFalse(
            unhandled,
            f"emitted event types with no handling in _to_managed_events — map them "
            f"(+ MAPPED_SAMPLES) or drop them (+ INTENTIONALLY_DROPPED): {sorted(unhandled)}",
        )

    def test_no_stale_known_types(self):
        """Nothing in our bookkeeping that the code no longer emits (keeps the
        samples honest)."""
        emitted = self._emitted_types()
        stale = (set(MAPPED_SAMPLES) | INTENTIONALLY_DROPPED) - emitted
        self.assertFalse(stale, f"tracked event types no longer emitted anywhere: {sorted(stale)}")


class TestResourceSDKCompat(unittest.TestCase):
    def test_session_resource_parses(self):
        rec = {
            "session_id": "ses_1", "status": "ready",
            "created_at": "t", "updated_at": "t",
            "ma_agent": {"type": "agent", "id": "ambient_v1", "version": 1},
            "environment_id": "default", "usage": {},
            "sandbox": {"backend": "inprocess", "status": "ready", "id": "ip_x"},
        }
        rec["mode"] = "agent"
        s = construct_type(type_=BetaManagedAgentsSession, value=_session_resource(rec))
        self.assertEqual(s.id, "ses_1")
        self.assertEqual(s.agent.id, "ambient_v1")
        self.assertEqual(s.mode, "agent")  # extra field survives SDK parsing

    def test_agent_resource_parses(self):
        rec = {
            "id": "ambient_v1", "version": 1, "name": "n", "description": "d",
            "model": "m", "system": "s", "tools": [{"type": "video_agent_20260825"}],
            "skills": [], "mcp_servers": [], "metadata": {},
            "created_at": "t", "updated_at": "t",
        }
        a = construct_type(type_=BetaManagedAgentsAgent, value=_agent_resource(rec))
        self.assertEqual(a.id, "ambient_v1")
        self.assertEqual(a.model.id, "m")
        self.assertEqual(a.mode, "agent")


class _FakeAgentStore:
    def __init__(self):
        self.agents: dict[str, dict] = {}

    async def put_agent(self, rec):
        self.agents[rec["id"]] = rec

    async def get_agent(self, agent_id):
        return self.agents.get(agent_id)


class _FakeRequest:
    def __init__(self, store):
        self.app = type("App", (), {"state": type("State", (), {"store": store})()})()


class TestAgentModeSelection(unittest.TestCase):
    """agents.create with no mode marker must give the tool loop, not fast mode."""

    def setUp(self):
        self.store = _FakeAgentStore()

    def _create(self, **body):
        return asyncio.run(ma.create_agent(_FakeRequest(self.store), body))

    def _session(self, agent_id, **meta):
        body = {"agent": agent_id, "metadata": {"video_id": "vid_x", **meta}}
        return asyncio.run(ma.build_session_record(body, self.store))

    def test_no_tools_defaults_to_agent_mode_and_agent_model(self):
        a = self._create(name="n", system="s")
        self.assertEqual(a["mode"], "agent")
        self.assertEqual(a["model"]["id"], settings.agent_model)
        self.assertEqual(self._session(a["id"])["mode"], "agent")

    def test_fast_marker_selects_fast_mode_and_vision_model(self):
        a = self._create(name="n", tools=[{"type": ma.FAST_MODE_TOOL}])
        self.assertEqual(a["mode"], "fast")
        self.assertEqual(a["model"]["id"], settings.llm_model)
        self.assertEqual(self._session(a["id"])["mode"], "fast")

    def test_explicit_model_wins(self):
        a = self._create(name="n", model="x/y", tools=[{"type": ma.FAST_MODE_TOOL}])
        self.assertEqual(a["model"]["id"], "x/y")

    def test_legacy_marker_is_agent_mode(self):
        a = self._create(name="n", tools=[{"type": "agent_toolset_20260401"}])
        self.assertEqual(a["mode"], "agent")

    def test_metadata_mode_overrides_agent(self):
        a = self._create(name="n")
        self.assertEqual(self._session(a["id"], mode="fast")["mode"], "fast")

    def test_unknown_tool_type_rejected(self):
        for tools in ([{"type": "video_agent"}], [{"type": "bash_20250124"}], ["video_agent_20260825"]):
            with self.assertRaises(APIError) as cm:
                self._create(name="n", tools=tools)
            self.assertEqual(cm.exception.status_code, 400)
            self.assertIn(ma.AGENT_MODE_TOOL, str(cm.exception.detail))

    def test_conflicting_markers_rejected(self):
        with self.assertRaises(APIError):
            self._create(name="n", tools=[{"type": ma.AGENT_MODE_TOOL}, {"type": ma.FAST_MODE_TOOL}])

    def test_bad_metadata_mode_rejected(self):
        a = self._create(name="n")
        with self.assertRaises(APIError):
            self._session(a["id"], mode="slow")


class TestSDKSurfacePresent(unittest.TestCase):
    def test_beta_managed_agents_surface_present(self):
        """A downgrade below the Managed Agents beta surface should fail loudly
        here rather than at runtime against a real client."""
        from anthropic import Anthropic

        c = Anthropic(api_key="x")
        self.assertTrue(hasattr(c.beta, "sessions"), "SDK lacks client.beta.sessions")
        self.assertTrue(hasattr(c.beta, "agents"), "SDK lacks client.beta.agents")
        self.assertTrue(SDK_EVENT_TYPES, "could not introspect SDK stream-event union")


if __name__ == "__main__":
    unittest.main()

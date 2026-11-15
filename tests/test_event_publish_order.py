"""Runner events are published in seq order, even when emitted concurrently.

Parallel tool calls emit their events (e.g. tool.scheduled -> agent.tool_use)
from concurrent coroutines. The live SSE tail skips any event at or below the
highest seq it has sent, so an event published after a higher-seq one never
reached live clients: Studio then got a tool result with no tool call
(AI_UIMessageStreamError "No tool invocation found") and replayed the turn.

No DB/Redis: the store assigns seqs and stalls, the broker records order.
"""
import asyncio
import unittest

from ambient.server.runner import SessionRunner
from ambient.server.sandbox import SandboxLimits


class _SlowStore:
    """Assigns seqs in call order, then finishes earlier inserts *later*, the
    way concurrent DB round trips can complete out of order."""

    def __init__(self):
        self.seq = 0

    async def append_event(self, _sid, type, payload):
        self.seq += 1
        seq = self.seq
        await asyncio.sleep(0.01 * (10 - seq))
        return {"seq": seq, "type": type, **payload}


class _Broker:
    def __init__(self):
        self.published: list[int] = []

    async def publish(self, _sid, ev):
        self.published.append(ev["seq"])


class _LiveTail:
    """Mirrors stream_events' live loop: drop seq <= highest sent."""

    def __init__(self):
        self.seen_max = 0
        self.sent: list[int] = []

    def feed(self, seqs):
        for s in seqs:
            if s <= self.seen_max:
                continue
            self.seen_max = s
            self.sent.append(s)


class EmitOrderTests(unittest.TestCase):
    def _runner(self):
        broker = _Broker()
        r = SessionRunner(
            session_id="ses_test", video_id="vid_test", model="m", max_turns_per_run=1,
            backend="inprocess", limits=SandboxLimits(), store=_SlowStore(), broker=broker,
        )
        return r, broker

    def test_concurrent_emits_publish_in_seq_order(self):
        r, broker = self._runner()

        async def go():
            await asyncio.gather(*(
                r._emit("tool.scheduled", {"tool_use_id": f"call_{i}"}) for i in range(5)
            ))
        asyncio.run(go())
        self.assertEqual(broker.published, sorted(broker.published))

    def test_live_tail_sees_every_tool_use(self):
        r, broker = self._runner()

        async def go():
            await asyncio.gather(*(
                r._emit("tool.scheduled", {"tool_use_id": f"call_{i}"}) for i in range(5)
            ))
        asyncio.run(go())
        tail = _LiveTail()
        tail.feed(broker.published)
        self.assertEqual(tail.sent, [1, 2, 3, 4, 5])


if __name__ == "__main__":
    unittest.main()

"""Regression tests for the SSE replay in `stream_events` (managed_agents route).

A run persists one event per streamed LLM chunk, so a single agent turn is often
tens of thousands of events. Two bugs broke replay once a run got that long:

  * the fresh-stream anchor scanned only the first 10,000 events, so a follow-up
    turn's stream anchored on turn 1 and replayed it before the new turn
    (Studio: "the follow-up repeats the first turn");
  * the replay read a single 10,000-event page, so resuming a long run stopped
    mid-run and fell into a live tail that never re-sends history (Studio: a
    reloaded session loads only partially and the stream never closes).

These run the real helpers against the in-memory LiteStore — no Postgres/Redis.

Run: python -m unittest tests.test_stream_replay
"""
import asyncio
import contextlib
import unittest

from ambient.lite import LiteStore
from ambient.server.routes.managed_agents import _iter_event_log, _replay_anchor

SID = "ses_test"


def _run(coro):
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


async def _add_run(store: LiteStore, chunks: int) -> dict:
    """Append one engine run: user.message, run.started, `chunks` token chunks,
    run.completed. Returns the boundary seqs."""
    um = await store.append_event(SID, "user.message", {"content": "q"})
    rs = await store.append_event(SID, "run.started", {})
    for _ in range(chunks):
        await store.append_event(SID, "chat.completion.chunk", {})
    rc = await store.append_event(SID, "run.completed", {"answer": "a"})
    return {"user": um["seq"], "started": rs["seq"], "completed": rc["seq"]}


async def _collect_run(store: LiteStore, after_seq: int, page_size: int) -> list[dict]:
    """What stream_events replays: everything after `after_seq` up to and
    including the first run.completed."""
    out = []
    async with contextlib.aclosing(_iter_event_log(store, SID, after_seq, page_size=page_size)) as events:
        async for ev in events:
            out.append(ev)
            if ev["type"] == "run.completed":
                break
    return out


class ReplayAnchorTests(unittest.TestCase):
    def test_follow_up_anchors_on_latest_run_past_10k_events(self):
        async def scenario():
            store = LiteStore()
            await _add_run(store, chunks=12_000)          # turn 1: > 10k events
            turn2 = await _add_run(store, chunks=5)       # turn 2 (the follow-up)
            anchor = await _replay_anchor(store, SID)
            # The fresh stream must start at turn 2, not turn 1.
            self.assertEqual(anchor, turn2["started"] - 1)
            replayed = await _collect_run(store, anchor, page_size=2000)
            self.assertEqual(replayed[0]["seq"], turn2["started"])
            self.assertEqual(replayed[-1]["seq"], turn2["completed"])

        _run(scenario())

    def test_anchor_before_run_started_is_emitted(self):
        # The user's message is persisted before the run task emits run.started;
        # a stream opened in that window must still anchor on the new turn.
        async def scenario():
            store = LiteStore()
            await _add_run(store, chunks=3)
            pending = await store.append_event(SID, "user.message", {"content": "q2"})
            self.assertEqual(await _replay_anchor(store, SID), pending["seq"] - 1)

        _run(scenario())

    def test_empty_session_anchors_at_zero(self):
        self.assertEqual(_run(_replay_anchor(LiteStore(), SID)), 0)


class ReplayPagingTests(unittest.TestCase):
    def test_long_run_replays_completely_to_run_completed(self):
        async def scenario():
            store = LiteStore()
            turn1 = await _add_run(store, chunks=12_000)
            replayed = await _collect_run(store, turn1["user"] - 1, page_size=2000)
            seqs = [e["seq"] for e in replayed]
            # Every event of the run, contiguous, ending on run.completed.
            self.assertEqual(seqs, list(range(turn1["user"], turn1["completed"] + 1)))
            self.assertEqual(replayed[-1]["type"], "run.completed")

        _run(scenario())

    def test_exact_page_multiple_terminates(self):
        async def scenario():
            store = LiteStore()
            for _ in range(4000):                          # exactly 2 pages of 2000
                await store.append_event(SID, "chat.completion.chunk", {})
            got = [e async for e in _iter_event_log(store, SID, 0, page_size=2000)]
            self.assertEqual(len(got), 4000)

        _run(scenario())


if __name__ == "__main__":
    unittest.main()

"""Per-request media budget (ambient.media_budget).

Pins the behaviors that keep caching and the agent's knowledge intact: nothing
changes below the high-water mark, the opening context and this turn's results
are protected, media thins before it disappears, and placeholders name the range
and how to re-open it.
"""
from __future__ import annotations

import asyncio
import copy
import unittest
from types import SimpleNamespace

from ambient import media_budget as mb
from ambient.media_plan import plan_media

FIREWORKS = "https://api.fireworks.ai/inference/v1"
JARVISLABS = "https://models.jarvislabs.net/v1"
MODAL_1017 = "https://example--ep-deepseek-v4-1-flash-1017-server.us-west.modal.direct/v1"
SELF_HOSTED = "http://10.0.0.5:8000/v1"

def img(nbytes: int = 0) -> dict:
    url = "data:image/jpeg;base64," + "A" * nbytes if nbytes else "https://s3.example/f.jpg"
    return {"type": "image_url", "image_url": {"url": url}}


def frames_msg(start: float, end: float, n: int, nbytes: int = 0) -> dict:
    return {"role": "user", "content": [
        {"type": "text", "text": mb.media_label("focus_clip", {"start_time": start, "end_time": end})},
        *[img(nbytes) for _ in range(n)],
    ]}


def clip_msg(start: float, end: float) -> dict:
    return {"role": "user", "content": [
        {"type": "text", "text": f"Clip of the video between {start:.1f} and {end:.1f} seconds"},
        {"type": "video_url", "video_url": {"url": "https://s3.example/c.mp4"}},
    ]}


def convo(context_frames: int, windows: list[tuple[float, float, int]], nbytes: int = 0) -> list[dict]:
    """system, opening context, question, then one tool round per window (the last
    window is this turn's fresh result)."""
    msgs = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": [{"type": "text", "text": "Frames sampled uniformly across the video (in time order):"},
                                     *[img(nbytes) for _ in range(context_frames)]]},
        {"role": "user", "content": [{"type": "text", "text": "List the key moments"}]},
    ]
    for i, (s, e, n) in enumerate(windows):
        msgs.append({"role": "assistant", "content": [{"type": "tool_use", "id": f"t{i}", "name": "focus_clip",
                                                        "input": {"start_time": s, "end_time": e}}]})
        msgs.append({"role": "user", "content": [{"type": "tool_result", "tool_use_id": f"t{i}", "content": f"notes {i}"}]})
        msgs.append(frames_msg(s, e, n, nbytes))
    return msgs


B = mb.MediaBudget(images=100, frames=100, inline_bytes=10_000_000)


class BudgetTests(unittest.TestCase):
    def test_measured_budgets(self):
        fw = mb.budget_for(FIREWORKS, "accounts/fireworks/models/kimi-k3")
        self.assertEqual(fw.images, 95)                     # 112 real-size frames x 0.85
        self.assertEqual(fw.source, "measured")
        jl = mb.budget_for(JARVISLABS, "deepseek-v4.1-flash-1017")
        self.assertEqual(jl.inline_bytes, int(12_000_000 * 0.85))      # a limit the probe hit

    def test_at_least_values_are_used_in_full(self):
        # Modal: the probe stopped at 384 frames, 32 MB inline and 128 MB by URL.
        b = mb.budget_for(MODAL_1017, "deepseek-ai/DeepSeek-V4.1-Flash-1017")
        self.assertEqual((b.images, b.frames), (384, 384))
        self.assertEqual(b.inline_bytes, 32_000_000)
        self.assertEqual(b.url_bytes, 128_000_000)

    def test_unmeasured_defaults(self):
        b = mb.budget_for(SELF_HOSTED, "qwen/qwen3.8-27b")
        self.assertEqual(b.source, "default")
        self.assertLessEqual(b.frames, 600)                # under vLLM's default decode cap

    def test_clip_frames_count_toward_frames(self):
        c = mb.message_cost(clip_msg(0, 30))
        self.assertEqual((c.images, c.frames), (0, 60))     # 30 s at 2 fps

    def test_inline_bytes_only_for_data_urls(self):
        self.assertEqual(mb.message_cost(frames_msg(0, 10, 3)).inline_bytes, 0)
        self.assertGreater(mb.message_cost(frames_msg(0, 10, 3, nbytes=1000)).inline_bytes, 3000)


class FitTests(unittest.TestCase):
    def test_under_high_water_is_untouched(self):
        msgs = convo(40, [(0, 30, 20), (60, 90, 20)])       # 80 images < 85
        before = copy.deepcopy(msgs)
        self.assertIsNone(mb.fit_media_budget(msgs, B))
        self.assertEqual(msgs, before)                     # byte-identical: cache keeps hitting

    def test_oldest_reduced_first_down_to_low_water(self):
        msgs = convo(40, [(0, 30, 20), (60, 90, 20), (120, 150, 20)])   # 100 images
        red = mb.fit_media_budget(msgs, B)
        self.assertEqual(red.after.images, 60)             # down to the 60% low-water mark
        self.assertFalse(mb.is_media_message(msgs[5]))     # both older windows -> notes
        self.assertFalse(mb.is_media_message(msgs[8]))
        self.assertEqual(mb.message_cost(msgs[-1]).images, 20)   # this turn's result: full
        self.assertEqual(red.fresh_touched, 0)
        self.assertEqual(msgs[4]["content"][0]["content"], "notes 0")   # tool text kept

    def test_context_is_never_reduced(self):
        msgs = convo(50, [(0, 30, 40), (60, 90, 40)])
        mb.fit_media_budget(msgs, B)
        self.assertEqual(mb.message_cost(msgs[1]).images, 50)

    def test_removed_media_leaves_a_reopenable_note(self):
        msgs = convo(50, [(130, 200, 4), (300, 330, 40)])
        msgs[5] = clip_msg(130, 200)                       # an old clip can't be thinned
        mb.fit_media_budget(msgs, mb.MediaBudget(images=100, frames=150, inline_bytes=10**9))
        note = msgs[5]["content"][0]["text"]
        self.assertEqual(len(msgs[5]["content"]), 1)
        self.assertIn("02:10–03:20", note)
        self.assertIn("focus_clip(start_time=130, end_time=200)", note)

    def test_thin_before_remove(self):
        msgs = convo(40, [(0, 30, 28), (60, 90, 28), (120, 150, 4)])   # 100 images
        red = mb.fit_media_budget(msgs, B)
        self.assertEqual((red.thumbnailed, red.removed), (2, 0))
        self.assertTrue(msgs[5]["content"][0]["text"].startswith(mb.REDUCED_PREFIX))
        self.assertIn("focus_clip(start_time=0, end_time=30)", msgs[5]["content"][0]["text"])
        self.assertEqual(mb.message_cost(msgs[5]).images, 7)

    def test_reduction_is_stable_on_the_next_turn(self):
        msgs = convo(40, [(0, 30, 20), (60, 90, 20), (120, 150, 20)])
        mb.fit_media_budget(msgs, B)
        snapshot = copy.deepcopy(msgs)
        self.assertIsNone(mb.fit_media_budget(msgs, B))    # hysteresis: no churn
        self.assertEqual(msgs, snapshot)

    def test_fresh_results_reduced_only_past_the_hard_limit(self):
        msgs = convo(50, [(0, 30, 70)])                    # 120 > 100 with only fresh media
        red = mb.fit_media_budget(msgs, B)
        self.assertEqual(red.fresh_touched, 1)
        self.assertLessEqual(red.after.images, 100)

    def test_inline_bytes_trigger(self):
        msgs = convo(10, [(0, 30, 10), (60, 90, 10)], nbytes=400_000)   # ~12 MB inline
        red = mb.fit_media_budget(msgs, B)
        self.assertIsNotNone(red)
        self.assertLessEqual(red.after.inline_bytes, B.inline_bytes)


class UrlBytesTests(unittest.TestCase):
    """Providers that download hosted images cap the total (Alibaba via OpenRouter:
    "Downloaded image content cannot exceed 30MB"); e2b frames are hosted URLs."""

    def hosted(self, start, end, n, tag):
        return {"role": "user", "content": [
            {"type": "text", "text": mb.media_label("focus_clip", {"start_time": start, "end_time": end})},
            *[{"type": "image_url", "image_url": {"url": f"https://s3.example/{tag}{i}.jpg"}} for i in range(n)]]}

    def test_budget_reads_measured_download_limit(self):
        b = mb.budget_for("https://openrouter.ai/api/v1", "qwen/qwen3.8-27b")   # pinned to alibaba
        self.assertEqual(b.url_bytes, int(28_000_000 * 0.85))
        self.assertIsNone(mb.budget_for(SELF_HOSTED, "qwen/qwen3.8-27b").url_bytes)

    def test_hosted_bytes_trigger_reduction(self):
        msgs = convo(0, [])
        msgs.append({"role": "assistant", "content": [{"type": "text", "text": "looking"}]})
        msgs.append(self.hosted(0, 30, 40, "a"))
        msgs.append({"role": "assistant", "content": [{"type": "text", "text": "more"}]})
        msgs.append(self.hosted(60, 90, 40, "b"))
        sizes = {u: 200_000 for u in mb.hosted_urls(msgs)}            # 80 x 200 KB = 16 MB
        b = mb.MediaBudget(images=500, frames=500, inline_bytes=10**9, url_bytes=10_000_000)
        red = mb.fit_media_budget(msgs, b, sizes)
        self.assertIsNotNone(red)
        self.assertLessEqual(red.after.url_bytes, b.url_bytes)
        self.assertEqual(red.fresh_touched, 0)

    def test_no_download_limit_means_no_url_cost(self):
        msgs = convo(0, [])
        msgs.append({"role": "assistant", "content": [{"type": "text", "text": "x"}]})
        msgs.append(self.hosted(0, 30, 80, "a"))
        self.assertIsNone(mb.fit_media_budget(msgs, mb.MediaBudget(500, 500, 10**9)))

    def test_hosted_video_does_not_count(self):
        self.assertEqual(mb.message_cost(clip_msg(0, 30)).url_bytes, 0)

    def test_context_frames_fit_download_limit(self):
        frames = [SimpleNamespace(frame_url=f"https://s3.example/f{i}.jpg") for i in range(100)]
        sizes = {f.frame_url: 300_000 for f in frames}
        b = mb.MediaBudget(images=200, frames=200, inline_bytes=10**9, url_bytes=12_000_000)
        kept = mb.fit_context_frames(frames, b, 0.5, sizes)
        self.assertLessEqual(len(kept) * 300_000, 6_000_000)
        self.assertIs(kept[-1], frames[-1])

    def test_url_sizes_probe(self):
        import functools, http.server, tempfile, threading
        d = tempfile.mkdtemp()
        with open(f"{d}/f.jpg", "wb") as f:
            f.write(b"x" * 12345)
        srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), functools.partial(
            http.server.SimpleHTTPRequestHandler, directory=d))
        srv.RequestHandlerClass.log_message = lambda *a: None
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        try:
            base = f"http://127.0.0.1:{srv.server_address[1]}"
            got = asyncio.run(mb.url_sizes([f"{base}/f.jpg", f"{base}/missing.jpg"]))
        finally:
            srv.shutdown()
        self.assertEqual(got, {f"{base}/f.jpg": 12345})       # unmeasurable -> left out


def unique_frames_msg(start: float, end: float, n: int) -> dict:
    return {"role": "user", "content": [
        {"type": "text", "text": mb.media_label("focus_clip", {"start_time": start, "end_time": end})},
        *[{"type": "image_url", "image_url": {"url": f"https://s3.example/{start:g}_{i}.jpg"}} for i in range(n)],
    ]}


def images(msg: dict) -> int:
    return mb.message_cost(msg).images


class FitMediaBudgetSpecTests(unittest.TestCase):
    """fit_media_budget, rule by rule. B allows 100 images: reductions start above 85
    and aim for 60. In convo(), msgs[1] is the opening context, msgs[5] the first
    window, and the last window is this turn's fresh result."""

    def test_no_change_at_or_below_85_percent(self):
        msgs = convo(45, [(0, 30, 20), (60, 90, 20)])                  # exactly 85
        before = copy.deepcopy(msgs)
        self.assertIsNone(mb.fit_media_budget(msgs, B))
        self.assertEqual(msgs, before)

    def test_reduction_starts_above_85_percent(self):
        msgs = convo(46, [(0, 30, 20), (60, 90, 20)])                  # 86
        red = mb.fit_media_budget(msgs, B)
        self.assertIsNotNone(red)
        self.assertEqual(red.before.images, 86)
        self.assertLess(red.after.images, 86)

    def test_old_images_thin_from_80_to_20(self):
        msgs = convo(10, [(100, 130, 5)])
        msgs.insert(3, {"role": "assistant", "content": [{"type": "text", "text": "looking"}]})
        msgs.insert(4, unique_frames_msg(0, 60, 80))                   # 95 in total
        urls = [b["image_url"]["url"] for b in msgs[4]["content"] if b["type"] == "image_url"]
        red = mb.fit_media_budget(msgs, B)
        kept = [b["image_url"]["url"] for b in msgs[4]["content"] if b["type"] == "image_url"]
        self.assertEqual(kept, urls[::4])                              # 20 evenly spaced frames
        self.assertEqual(len(kept), 20)
        self.assertEqual((red.thumbnailed, red.removed), (1, 0))
        note = msgs[4]["content"][0]["text"]
        self.assertTrue(note.startswith(mb.REDUCED_PREFIX))
        self.assertIn("80 to 20 frames", note)

    def test_reduction_stops_once_at_60_percent(self):
        msgs = convo(10, [(0, 30, 60), (60, 90, 20), (120, 150, 10)])  # 100
        red = mb.fit_media_budget(msgs, B)
        self.assertEqual(images(msgs[5]), 15)                          # oldest thinned: 55 total
        self.assertEqual(images(msgs[8]), 20)                          # next one left alone
        self.assertEqual(red.after.images, 55)
        self.assertEqual(red.thumbnailed, 1)

    def test_reduction_continues_to_notes_to_reach_60_percent(self):
        msgs = convo(40, [(0, 30, 20), (60, 90, 20), (120, 150, 20)])  # 100; thinning alone reaches 70
        red = mb.fit_media_budget(msgs, B)
        self.assertEqual(red.after.images, 60)
        self.assertEqual((red.thumbnailed, red.removed), (2, 2))
        self.assertFalse(mb.is_media_message(msgs[5]))
        self.assertFalse(mb.is_media_message(msgs[8]))

    def test_opening_context_remains_protected(self):
        msgs = convo(70, [(0, 30, 30), (60, 90, 10)])                  # context alone is above 60%
        before = copy.deepcopy(msgs[1])
        red = mb.fit_media_budget(msgs, B)
        self.assertEqual(msgs[1], before)
        self.assertEqual(red.removed, 1)                               # the old window went instead

    def test_fresh_media_protected_below_the_hard_limit(self):
        msgs = convo(50, [(0, 30, 30), (60, 90, 18)])                  # 98
        fresh = copy.deepcopy(msgs[-1])
        red = mb.fit_media_budget(msgs, B)
        self.assertEqual(msgs[-1], fresh)
        self.assertEqual(red.fresh_touched, 0)
        self.assertEqual(red.after.images, 68)                         # stays above 60%, under 100%

    def test_fresh_only_conversation_at_the_hard_limit_is_untouched(self):
        msgs = convo(40, [(0, 30, 60)])                                # exactly 100
        before = copy.deepcopy(msgs)
        self.assertIsNone(mb.fit_media_budget(msgs, B))
        self.assertEqual(msgs, before)

    def test_fresh_media_reduced_above_100_percent(self):
        msgs = convo(50, [(0, 30, 70)])                                # 120
        red = mb.fit_media_budget(msgs, B)
        self.assertEqual(red.fresh_touched, 1)
        self.assertEqual(images(msgs[-1]), 18)
        self.assertLessEqual(red.after.images, 100)

    def test_newest_fresh_media_is_never_removed(self):
        msgs = convo(90, [])
        msgs.append({"role": "assistant", "content": [{"type": "text", "text": "looking"}]})
        msgs.append(clip_msg(0, 60))                                   # 120 frames: can't be thinned
        with self.assertLogs("ambient.media_budget", level="WARNING"):
            self.assertIsNone(mb.fit_media_budget(msgs, B))
        self.assertTrue(mb.is_media_message(msgs[-1]))

    def test_earlier_fresh_media_goes_before_the_newest(self):
        msgs = convo(50, [])
        msgs.append({"role": "assistant", "content": [{"type": "text", "text": "looking"}]})
        msgs.append(clip_msg(0, 30))
        msgs.append(clip_msg(100, 130))                                # 170 frame-equivalents
        red = mb.fit_media_budget(msgs, B)
        self.assertEqual(red.fresh_touched, 1)
        self.assertFalse(mb.is_media_message(msgs[-2]))
        self.assertTrue(mb.is_media_message(msgs[-1]))

    def test_clips_go_directly_to_placeholders(self):
        msgs = convo(10, [])
        msgs.append({"role": "assistant", "content": [{"type": "text", "text": "looking"}]})
        msgs.append(clip_msg(0, 30))                                   # 60 frame-equivalents
        msgs.append({"role": "assistant", "content": [{"type": "text", "text": "more"}]})
        msgs.append(frames_msg(200, 215, 30))                          # 100 in total
        red = mb.fit_media_budget(msgs, B)
        self.assertEqual((red.thumbnailed, red.removed), (0, 1))
        note = msgs[4]["content"]
        self.assertEqual(len(note), 1)
        self.assertTrue(note[0]["text"].startswith(mb.REMOVED_PREFIX))
        self.assertIn("(clip)", note[0]["text"])

    def test_windows_too_small_to_thin_go_directly_to_placeholders(self):
        msgs = convo(82, [(0, 30, 4), (60, 90, 4)])                    # 90; 4 frames can't be thinned
        red = mb.fit_media_budget(msgs, B)
        self.assertEqual((red.thumbnailed, red.removed), (0, 1))

    def _trigger(self, msgs, budget, sizes=None, *, unit):
        red = mb.fit_media_budget(msgs, budget, sizes)
        self.assertIsNotNone(red, unit)
        over = {u: getattr(red.before, u) > (getattr(budget, u) or float("inf")) * mb.HIGH
                for u in ("images", "frames", "inline_bytes", "url_bytes")}
        self.assertEqual([u for u, v in over.items() if v], [unit])

    def test_image_limit_alone_triggers(self):
        self._trigger(convo(40, [(0, 30, 30), (60, 90, 20)]),
                      mb.MediaBudget(images=100, frames=10_000, inline_bytes=10**9), unit="images")

    def test_frame_limit_alone_triggers(self):
        msgs = convo(0, [(200, 215, 1)])
        msgs[5] = clip_msg(0, 60)                                      # 120 frames from one image-free clip
        msgs.append({"role": "assistant", "content": [{"type": "text", "text": "more"}]})
        msgs.append(frames_msg(300, 310, 1))
        self._trigger(msgs, mb.MediaBudget(images=1_000, frames=100, inline_bytes=10**9), unit="frames")

    def test_inline_byte_limit_alone_triggers(self):
        self._trigger(convo(2, [(0, 30, 2), (60, 90, 2)], nbytes=200_000),   # ~1.2 MB
                      mb.MediaBudget(images=1_000, frames=1_000, inline_bytes=1_000_000), unit="inline_bytes")

    def test_url_byte_limit_alone_triggers(self):
        msgs = convo(0, [(200, 215, 1)])
        msgs[5] = unique_frames_msg(0, 30, 40)
        msgs.append({"role": "assistant", "content": [{"type": "text", "text": "more"}]})
        msgs.append(unique_frames_msg(60, 90, 40))
        sizes = {u: 200_000 for u in mb.hosted_urls(msgs)}             # 16 MB
        self._trigger(msgs, mb.MediaBudget(images=1_000, frames=1_000, inline_bytes=10**9, url_bytes=10_000_000),
                      sizes, unit="url_bytes")

    def test_video_limit_alone_triggers_without_the_percent_marks(self):
        b = mb.MediaBudget(images=1_000, frames=10_000, inline_bytes=10**9, videos=2)
        msgs = convo(0, [])
        msgs.append({"role": "assistant", "content": [{"type": "text", "text": "a"}]})
        msgs.append(clip_msg(0, 30))
        msgs.append({"role": "assistant", "content": [{"type": "text", "text": "b"}]})
        msgs.append(clip_msg(60, 90))
        self.assertIsNone(mb.fit_media_budget(msgs, b))                # 2 of 2 is allowed
        msgs.append({"role": "assistant", "content": [{"type": "text", "text": "c"}]})
        msgs.append(clip_msg(120, 150))
        red = mb.fit_media_budget(msgs, b)
        self.assertEqual(red.after.videos, 2)                          # only as many as needed go
        self.assertEqual(red.ranges, ["00:00–00:30"])
        self.assertTrue(mb.is_media_message(msgs[6]))

    def test_counters_costs_and_time_ranges(self):
        msgs = convo(40, [(130, 200, 20), (300, 330, 20), (400, 430, 20)])
        red = mb.fit_media_budget(msgs, B)
        self.assertEqual((red.before.images, red.after.images), (100, 60))
        self.assertEqual((red.thumbnailed, red.removed, red.fresh_touched), (2, 2, 0))
        self.assertEqual(red.ranges, ["02:10–03:20", "05:00–05:30"])
        d = red.as_dict()
        self.assertEqual(d["before"]["images"], 100)
        self.assertEqual(d["budget"]["images"], 100)

    def test_warns_when_protected_media_cannot_fit(self):
        msgs = convo(150, [])                                          # the opening context alone is 150
        before = copy.deepcopy(msgs)
        with self.assertLogs("ambient.media_budget", level="WARNING") as logs:
            self.assertIsNone(mb.fit_media_budget(msgs, B))
        self.assertIn("still over budget", logs.output[0])
        self.assertEqual(msgs, before)


class TiledClipRangeTests(unittest.TestCase):
    """A window can arrive as several consecutive tiles, each labeled separately."""

    def tiled(self):
        content = []
        for s, e in ((0, 60), (60, 120), (120, 180)):
            content += clip_msg(s, e)["content"]
        return {"role": "user", "content": content}

    def test_note_names_the_whole_window(self):
        note = mb._placeholder(self.tiled())["content"][0]["text"]
        self.assertIn("00:00–03:00", note)
        self.assertIn("focus_clip(start_time=0, end_time=180)", note)

    def test_reduction_reports_the_whole_window(self):
        msgs = convo(10, [(300, 310, 1)])
        msgs[5] = self.tiled()                                         # 3 x 120 frames
        msgs.append({"role": "assistant", "content": [{"type": "text", "text": "more"}]})
        msgs.append(frames_msg(400, 410, 1))
        red = mb.fit_media_budget(msgs, mb.MediaBudget(images=1_000, frames=200, inline_bytes=10**9))
        self.assertEqual(red.ranges, ["00:00–03:00"])

    def test_thinned_frames_keep_their_range_for_a_later_note(self):
        thinned = mb._thumbnail(frames_msg(130, 200, 40))
        self.assertIn("02:10–03:20", mb._placeholder(thinned)["content"][0]["text"])


class ContextTests(unittest.TestCase):
    def test_agent_window_leaves_room_for_context(self):
        p = plan_media("self_focus_clip", FIREWORKS, "accounts/fireworks/models/kimi-k3", 300)
        self.assertEqual(p.method, "frames")
        self.assertEqual(p.max_frames, 47)                 # 95 - 50% context share

    def test_context_frames_fit_bytes(self):
        frames = [SimpleNamespace(frame_url="data:image/jpeg;base64," + "A" * 100_000) for _ in range(100)]
        kept = mb.fit_context_frames(frames, mb.MediaBudget(images=200, frames=200, inline_bytes=2_000_000), 0.5)
        self.assertLessEqual(sum(mb.frame_inline_bytes(f) for f in kept), 1_000_000)
        self.assertGreaterEqual(len(kept), 9)
        self.assertIs(kept[0], frames[0])
        self.assertIs(kept[-1], frames[-1])                # still spans the whole video

    def test_hosted_url_frames_cost_no_bytes(self):
        frames = [SimpleNamespace(frame_url="https://s3.example/f.jpg") for _ in range(100)]
        self.assertEqual(len(mb.fit_context_frames(frames, B, 0.5)), 50)


if __name__ == "__main__":
    unittest.main()

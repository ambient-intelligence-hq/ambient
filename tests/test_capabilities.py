"""Measured capabilities (vendored vlm.fyi data) in media planning.

Cases are real measurements from the provider matrix (2026-10-31): they pin
that the data resolves for the endpoints the engine uses, and that it only
vetoes the video path where a video would fail or be wasted.
"""
from __future__ import annotations

import unittest

from ambient.capabilities import MARGIN, Capabilities, resolve
from ambient.media_plan import image_block_cap, plan_media

OPENROUTER = "https://openrouter.ai/api/v1"
FIREWORKS = "https://api.fireworks.ai/inference/v1"
JARVISLABS = "https://models.jarvislabs.net/v1"
MODAL_KIMI = "https://example--ep-kimi-k3-server.us-west.modal.direct/v1"


class ResolveTests(unittest.TestCase):
    def test_openrouter_pinned_upstream(self):
        c = resolve(OPENROUTER, "qwen/qwen3.8-27b")          # pinned to alibaba
        self.assertIsNotNone(c)
        self.assertEqual(c.entry_id, "openrouter-alibaba/qwen3.8-27b")
        self.assertEqual(c.images_per_request, 250)

    def test_direct_provider_with_vendor_specific_id(self):
        c = resolve(FIREWORKS, "accounts/fireworks/models/kimi-k3")
        self.assertEqual(c.entry_id, "fireworks/kimi-k3")
        self.assertEqual(c.images_per_request, 120)

    def test_quantized_served_id(self):
        c = resolve(JARVISLABS, "qwen-3.8-27b-fp8")
        self.assertEqual(c.entry_id, "jarvislabs/qwen3.8-27b")
        self.assertTrue(c.capped)
        self.assertLessEqual(c.frames_per_video, 40)

    def test_unmeasured_is_none(self):
        self.assertIsNone(resolve("http://10.0.0.5:8000/v1", "qwen/qwen3.8-27b"))
        self.assertIsNone(resolve(OPENROUTER, "some/unmeasured-model"))


class ImageCapTests(unittest.TestCase):
    def test_measured_cap_with_margin(self):
        # Fireworks: 120 images, 112 real-size frames -> the lower, with margin.
        self.assertEqual(image_block_cap(FIREWORKS, "accounts/fireworks/models/kimi-k3"), int(112 * MARGIN))

    def test_measured_cap_can_raise_it(self):
        # Jarvislabs deepseek fits at least 384 real-size frames. Its ~12 MB inline body
        # limit is enforced on the frames' real size (media_budget), not here.
        self.assertEqual(image_block_cap(JARVISLABS, "deepseek-v4.1-flash-1017"), 384)

    def test_unmeasured_falls_back_to_profile(self):
        # Self-hosted profile allows 1000; the per-request frame ceiling (500) binds.
        self.assertEqual(image_block_cap("http://10.0.0.5:8000/v1", "qwen/qwen3.8-27b"), 500)


class MarginTests(unittest.TestCase):
    """A limit the probe hit gets the margin; an "at least" value is used in full."""

    def test_usable(self):
        c = Capabilities(entry_id="x", measured_at="", images_per_request=120, frames_768x432=384,
                         lower_bounds=frozenset({"frames_768x432"}))
        self.assertEqual(c.usable("images_per_request"), int(120 * MARGIN))
        self.assertEqual(c.usable("frames_768x432"), 384)
        self.assertIsNone(c.usable("inline_bytes"))

    def test_image_cap_applies_the_rule_per_limit(self):
        c = Capabilities(entry_id="x", measured_at="", images_per_request=400, frames_768x432=384,
                         lower_bounds=frozenset({"frames_768x432"}))
        self.assertEqual(c.image_cap(), int(400 * MARGIN))             # 340 < 384

    def test_inconclusive_value_keeps_the_margin(self):
        from ambient.capabilities import _from_entry
        c = _from_entry({"id": "x", "limits": {
            "images_per_request": {"status": "inconclusive", "at_least": 512},
            "frames_768x432": {"status": "at_least", "value": 384},
            "url_image_bytes": {"status": "measured", "value": 80_000_000},
        }})
        self.assertEqual(c.usable("images_per_request"), int(512 * MARGIN))
        self.assertEqual(c.usable("frames_768x432"), 384)
        self.assertEqual(c.usable("url_image_bytes"), int(80_000_000 * MARGIN))

    def test_video_file_at_least_is_not_vetoed_up_to_its_value(self):
        c = Capabilities(entry_id="x", measured_at="", video="yes", video_file_bytes=512_000_000,
                         lower_bounds=frozenset({"video_file_bytes"}))
        self.assertIsNone(c.video_vetoed(10, video_bytes=500_000_000))
        self.assertIn("MB", c.video_vetoed(10, video_bytes=513_000_000))


class VideoVetoTests(unittest.TestCase):
    def test_capped_video_vetoed_when_frames_deliver_more(self):
        # e.g. Qwen3.8 on Reka via OpenRouter: video capped at ~32 frames, but 170
        # images per request -> frames win for a 5-minute context.
        c = Capabilities(entry_id="x", measured_at="", video="yes", capped=True, frames_per_video=32)
        self.assertIn("caps video", c.video_vetoed(300, frames_alternative=144))
        self.assertIsNone(c.video_vetoed(300, frames_alternative=3))
        self.assertIsNone(c.video_vetoed(20, frames_alternative=144))   # short clip fits the cap

    def test_capped_deployment_with_tiny_image_cap_keeps_video(self):
        # Qwen-fp8 on Jarvislabs: ~32-frame video cap, 4 images per request.
        self.assertEqual(plan_media("context", JARVISLABS, "qwen-3.8-27b-fp8", 300).method, "video")

    def test_cap_veto_only_when_frames_would_deliver_more(self):
        # Gemma on Jarvislabs: video capped at ~32 frames, but only 4 images per
        # request, so the video (32 frames) still beats frames (3).
        self.assertEqual(plan_media("context", JARVISLABS, "gemma-4-31b-it", 300).method, "video")

    def test_capped_provider_keeps_video_for_short_clips(self):
        # A 10 s focus clip at 2 fps (20 frames) fits under the cap.
        self.assertEqual(plan_media("focus_clip", JARVISLABS, "qwen-3.8-27b-fp8", 10).method, "video")

    def test_image_only_deployment_never_gets_video(self):
        p = plan_media("focus_clip", MODAL_KIMI, "moonshotai/Kimi-K3", 10)
        self.assertEqual(p.method, "frames")
        self.assertIn("rejects video", p.meta["reason"])

    def test_oversized_file_falls_back_to_frames(self):
        # Gemini on AI Studio accepts video files up to ~12 MB.
        big = plan_media("context", OPENROUTER, "google/gemini-3.7-flash", 300, video_bytes=40_000_000)
        small = plan_media("context", OPENROUTER, "google/gemini-3.7-flash", 300, video_bytes=5_000_000)
        self.assertEqual(big.method, "frames")
        self.assertEqual(small.method, "video")

    def test_unmeasured_endpoint_unchanged(self):
        self.assertEqual(plan_media("context", "http://10.0.0.5:8000/v1", "qwen/qwen3.8-27b", 300).method, "video")


if __name__ == "__main__":
    unittest.main()

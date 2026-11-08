"""Model-id matching against the catalog, and endpoint classification.

Served model ids differ from catalog ids by org prefix, quant suffix, Fireworks'
`p`-for-dot versions and hyphenation; a miss silently sends a vision model down
the text-only path, so these pin the cases seen in the provider matrix.
"""
from __future__ import annotations

import unittest

from ambient.config import match_model
from ambient.media_plan import classify_service, plan_media


class MatchModelTests(unittest.TestCase):
    def test_matches(self):
        for served, catalog in [
            ("unsloth/Qwen3.8-27B-NVFP4", "qwen/qwen3.8-27b"),
            ("deepseek-v4.1-flash-1017", "deepseek/deepseek-v4.1-flash-1017"),
            ("accounts/fireworks/models/deepseek-v4p1-flash-1017", "deepseek/deepseek-v4.1-flash-1017"),
            ("accounts/fireworks/models/glm-5p3-flash", "z-ai/glm-5.3-flash"),
            ("qwen-3.8-27b-fp8", "qwen/qwen3.8-27b"),
            ("zai-org/GLM-5.3-Flash", "z-ai/glm-5.3-flash"),
        ]:
            with self.subTest(served=served):
                self.assertTrue(match_model(served, catalog))

    def test_non_matches(self):
        for served, catalog in [
            ("openai/gpt-5.6-luna", "openai/gpt-5.6-luna-pro"),   # tier kept
            ("qwen/qwen3.8-27b", "qwen/qwen3.8-2.4t-a95b"),
            ("z-ai/glm-5.3-flash", "z-ai/glm-5.3"),
            ("deepseek-v4p1-flash", "deepseek/deepseek-v4-flash"),
        ]:
            with self.subTest(served=served):
                self.assertFalse(match_model(served, catalog))


class EndpointProfileTests(unittest.TestCase):
    def test_hosted_apis_classified(self):
        self.assertEqual(classify_service("https://api.fireworks.ai/inference/v1"), "fireworks")
        self.assertEqual(classify_service("https://api.together.ai/v1"), "together")
        self.assertEqual(classify_service("https://api.openai.com/v1"), "openai")
        self.assertEqual(classify_service("http://10.0.0.5:8000/v1"), "self_hosted")

    def test_hosted_apis_use_capped_frames(self):
        for url in ("https://api.fireworks.ai/inference/v1", "https://api.together.ai/v1", "https://api.openai.com/v1"):
            with self.subTest(url=url):
                # An unmeasured model, so the endpoint profile's cap applies (measured
                # pairs use their own limit; see test_capabilities).
                p = plan_media("context", url, "moonshotai/kimi-k2-unmeasured", 1800)
                self.assertEqual(p.method, "frames")
                self.assertLessEqual(p.max_frames, 100)


if __name__ == "__main__":
    unittest.main()

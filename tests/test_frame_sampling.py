"""The whole-video context never samples more frames than the source has.

Asking a 10.4 s, 10 fps clip (104 frames) for 192 frames made the sampler pick
fps = ceil(192 / 10.4) = 19; the extractor then repeated source frames to fill
19 fps, so ~45% of the attached images were duplicates (token cost, no content).
The source frame rate is now probed at upload and caps the sample.
"""
import asyncio
import hashlib
import os
import shutil
import subprocess
import tempfile
import unittest
from types import SimpleNamespace
from unittest import mock

from ambient import fast_mode as fm


class _FakeTools:
    def __init__(self, dur):
        self._duration_sec = dur
        self.calls = []

    def fetch_frames(self, **kw):
        self.calls.append(kw)
        return []


def _sample(dur, source_fps, n=192):
    tools = _FakeTools(dur)
    with mock.patch("ambient.tools.video_backend.make_video_tools", lambda *a, **k: tools):
        asyncio.run(fm.sample_fast_frames("v", None, SimpleNamespace(max_frames=n, max_dim=768),
                                          duration=dur, source_fps=source_fps))
    return tools.calls[-1]


class SamplePlanTests(unittest.TestCase):
    def test_short_clip_is_capped_at_its_frame_count(self):
        call = _sample(10.4, 10)
        self.assertEqual(call["max_frames"], 104)
        self.assertEqual(call["fps"], 10)

    def test_unknown_source_rate_keeps_the_old_plan(self):
        call = _sample(10.4, None)
        self.assertEqual((call["max_frames"], call["fps"]), (192, 19))

    def test_long_video_is_unaffected(self):
        call = _sample(600.0, 30)
        self.assertEqual((call["max_frames"], call["fps"]), (192, 1))

    def test_frame_count(self):
        self.assertEqual(fm.source_frame_count(10.4, 10), 104)
        self.assertIsNone(fm.source_frame_count(10.4, None))
        self.assertIsNone(fm.source_frame_count(0, 10))


@unittest.skipUnless(shutil.which("ffmpeg") and shutil.which("ffprobe"), "needs ffmpeg")
class RangeProxyDuplicateFrameTests(unittest.TestCase):
    """Real ffmpeg on a 10.4 s, 10 fps testsrc clip whose every frame differs."""

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.mkdtemp()
        cls.vid = "vid_dupcheck"
        cls.path = os.path.join(cls.tmp, f"{cls.vid}.mp4")
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", "testsrc=size=320x180:rate=10",
                        "-t", "10.4", "-pix_fmt", "yuv420p", "-c:v", "libx264", cls.path], check=True)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def _frames(self, source_fps):
        from ambient.config import settings
        with mock.patch.object(settings, "video_folder", self.tmp), \
                mock.patch.object(settings, "sandbox_backend", "inprocess"):
            frames, _ = asyncio.run(fm.sample_fast_frames(
                self.vid, None, SimpleNamespace(max_frames=192, max_dim=768), duration=10.4,
                source_fps=source_fps))
        return frames

    @staticmethod
    def _dupes(frames):
        hashes = [hashlib.md5(f.frame_url.encode()).hexdigest() for f in frames]
        return len(hashes) - len(set(hashes))

    def test_without_the_rate_frames_repeat(self):
        frames = self._frames(None)
        self.assertGreater(self._dupes(frames), 50)   # documents the old behaviour

    def test_with_the_rate_every_frame_is_distinct(self):
        info = fm.probe_source_video(self.path)
        self.assertEqual((info["width"], info["height"], info["fps"]), (320, 180, 10.0))
        frames = self._frames(info["fps"])
        self.assertEqual(len(frames), 104)
        self.assertEqual(self._dupes(frames), 0)

    def test_upload_records_the_source_info(self):
        from ambient.server import video_store
        with mock.patch.object(video_store, "VIDEO_FOLDER", self.tmp), \
                mock.patch.object(video_store, "get_s3_client", lambda: SimpleNamespace(bucket=None)):
            with open(self.path, "rb") as fh:
                meta = video_store.store_video(fh.read(), "clip.mp4", "video/mp4")
        self.assertEqual(meta["video_info"], {"width": 320, "height": 180, "fps": 10.0})


if __name__ == "__main__":
    unittest.main()

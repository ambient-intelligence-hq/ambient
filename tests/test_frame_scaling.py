"""Media frames and clips fit inside max_dim without upscaling smaller sources.

A fixed `scale=768:-2` turned a 320x180 source into 768x432 frames (2.5x the
bytes, no new detail) and let portrait sources exceed 768 on the long side.
Runs real ffmpeg on synthetic sources; skipped when ffmpeg is missing.
"""
import os
import shutil
import subprocess
import tempfile
import unittest

from ambient.tools.rangeproxy_video_tools import _fit_filter


def _probe(path: str) -> tuple[int, int]:
    out = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0",
                          "-show_entries", "stream=width,height", "-of", "csv=p=0", path],
                         capture_output=True, text=True, check=True).stdout.strip()
    w, h = out.split(",")
    return int(w), int(h)


@unittest.skipUnless(shutil.which("ffmpeg") and shutil.which("ffprobe"), "needs ffmpeg")
class FitFilterTests(unittest.TestCase):
    CASES = {
        (320, 180): (320, 180),     # small: kept at source size
        (321, 181): (320, 180),     # odd: snapped to even (libx264 needs it)
        (1280, 720): (768, 432),    # large: scaled down
        (720, 1280): (432, 768),    # portrait: long side capped, not the width
        (768, 432): (768, 432),     # exact fit: unchanged
    }

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.mkdtemp()

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def _source(self, w: int, h: int) -> str:
        path = os.path.join(self.tmp, f"src_{w}x{h}.mp4")
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", f"testsrc=size={w}x{h}:rate=10",
                        "-t", "1", "-pix_fmt", "yuv444p", "-c:v", "libx264", path], check=True)
        return path

    def test_frames(self):
        for (w, h), want in self.CASES.items():
            with self.subTest(source=(w, h)):
                out = os.path.join(self.tmp, f"f_{w}x{h}.jpg")
                subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", self._source(w, h), "-frames:v", "1",
                                "-vf", _fit_filter(768), "-pix_fmt", "yuvj420p", out], check=True)
                self.assertEqual(_probe(out), want)

    def test_h264_clips(self):
        for (w, h), want in self.CASES.items():
            with self.subTest(source=(w, h)):
                out = os.path.join(self.tmp, f"c_{w}x{h}.mp4")
                subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", self._source(w, h),
                                "-vf", f"{_fit_filter(768)},fps=2", "-c:v", "libx264", "-an", out], check=True)
                self.assertEqual(_probe(out), want)


if __name__ == "__main__":
    unittest.main()

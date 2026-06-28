import os
import json
import subprocess
from typing import List, Optional, Tuple
from ambient import Frame
import glob
import importlib
from ambient.llm import llm_call
from ambient.config import settings
from ambient import Clip

VIDEO_FOLDER = settings.video_folder

# Videos longer than this (seconds) use the ffmpeg backend unconditionally.
# decord's VideoReader builds a full seek index in RAM; for a long high-res video
# that index alone can exceed available memory and trigger an OOM kill. Mirrors
# ambient/sandboxes/e2b/video-analysis-v1/main.py.
DECORD_MAX_DURATION_SECS = int(os.environ.get("DECORD_MAX_DURATION_SECS", "300"))


class VideoFrameTools:
    FPS = 1
    SKIM_FPS = 1
    SKIM_MAX_FRAMES = 25
    FOCUS_FPS = 2
    FOCUS_MAX_FRAMES = 30
    OVERVIEW_MAX_FRAMES = 75
    OVERVIEW_FPS = 2
    MAX_CLIP_DURATION_SEC = 600
    # Maximum frames decoded and held in RAM at once by the decord backend.
    # At 1080p a single RGB frame is ~6 MB; 32 frames ~= 200 MB, well within budget.
    DECORD_BATCH_SIZE = 32
    PROMPT = """You are video analysis expert, your are given frames from a video (with equal interval sampling), 
    Please describe the content of the viewed video frames in detail with their timestamps (each frame with ~25 words). If query related content is found, please highlight the timestamps of the video frames that are relevant to the question and explain why (each timestamp with additional ~50 words). Do not answer the question directly.
    
    You should cite the frames in the analysis in the following format:
    (timestamp)[Frame {frame_index}]
    

    <analysis>
    {analysis}
    </analysis>
    """

    def __init__(self, video_id: str, max_frame_dimention: Optional[int] = None):
        self.video_id = video_id
        self.video_path = self.get_video_path()
        self.frame_dir = os.path.join(VIDEO_FOLDER, video_id, "frames")
        os.makedirs(self.frame_dir, exist_ok=True)
        self.frames: List[Frame] = []
        self._vr: Optional[object] = None
        self._backend: Optional[str] = None
        self._avg_fps: Optional[float] = None
        self._duration_sec: Optional[float] = None
        self.max_frame_dimention = max_frame_dimention

    def get_video_path(self) -> str:
        # Search for the video with common extensions.
        glob_pattern = os.path.join(VIDEO_FOLDER, f"{self.video_id}*")
        video_paths = [
            path
            for path in glob.glob(glob_pattern)
            if path.endswith(
                (
                    ".mp4",
                    ".mov",
                    ".avi",
                    ".mkv",
                    ".webm",
                    ".flv",
                    ".wmv",
                    ".mpeg",
                    ".mpg",
                    ".m4v",
                    ".3gp",
                    ".3g2",
                    ".mj2",
                )
            )
        ]
        if len(video_paths) == 0:
            raise FileNotFoundError(f"Video not found: {glob_pattern}")
        return video_paths[0]

    def _probe_video_info(self) -> tuple[float, float]:
        cmd = [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=avg_frame_rate,duration:format=duration",
            "-of",
            "default=noprint_wrappers=1",
            self.video_path,
        ]
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        avg_frame_rate = ""
        duration = 0.0
        for line in result.stdout.splitlines():
            if line.startswith("avg_frame_rate="):
                avg_frame_rate = line.split("=", 1)[1].strip()
            elif line.startswith("duration="):
                try:
                    duration = float(line.split("=", 1)[1].strip())
                except ValueError:
                    pass

        fps = 1.0
        if avg_frame_rate and avg_frame_rate != "0/0":
            if "/" in avg_frame_rate:
                num, den = avg_frame_rate.split("/", 1)
                fps = float(num) / float(den) if float(den) != 0 else 1.0
            else:
                fps = float(avg_frame_rate)
        if fps <= 0:
            fps = 1.0

        return duration, fps

    def _ensure_backend(self) -> None:
        if self._backend is not None:
            return

        # Probe duration cheaply via ffprobe first so we can decide whether
        # decord is safe to use before loading it (decord builds a full seek
        # index in RAM; on long videos that alone can trigger an OOM kill).
        duration, fps = self._probe_video_info()

        if duration > DECORD_MAX_DURATION_SECS:
            print(
                f"[backend] ffmpeg  fps={fps:.2f}  duration={duration:.2f}s"
                f"  (decord skipped: video > {DECORD_MAX_DURATION_SECS}s threshold)"
            )
            self._duration_sec = duration
            self._avg_fps = fps
            self._backend = "ffmpeg"
            return

        try:
            decord = importlib.import_module("decord")
            self._vr = decord.VideoReader(self.video_path, ctx=decord.cpu(0))
            self._avg_fps = float(self._vr.get_avg_fps()) or 1.0
            self._duration_sec = len(self._vr) / self._avg_fps
            self._backend = "decord"
        except ModuleNotFoundError:
            # macOS arm64 fallback when decord wheels are unavailable
            self._duration_sec = duration
            self._avg_fps = fps
            self._backend = "ffmpeg"

    def _get_video_reader(self):
        self._ensure_backend()
        return self._vr

    def _video_duration_seconds(self) -> float:
        self._ensure_backend()
        return self._duration_sec

    def _normalize_time_window(
        self, start_time_ms: float, end_time_ms: Optional[float]
    ) -> tuple[float, Optional[float]]:
        """
        Normalize start/end timestamps to seconds. Accepts milliseconds too.
        """
        video_duration = self._video_duration_seconds()

        start = float(start_time_ms) / 1000.0
        end = float(end_time_ms) / 1000.0 if end_time_ms is not None else None

        # Clamp within valid bounds.
        start = max(0.0, min(start, video_duration))
        if end is not None:
            end = max(start, min(end, video_duration))

        return start, end

    def _frame_cache_path(self, frame_index: int) -> str:
        return os.path.join(self.frame_dir, f"frame_{frame_index:09d}.png")

    def _fps_cache_dir(
        self,
        fps: int,
        start_time: Optional[float] = None,
        duration_sec: Optional[float] = None,
        max_frames: Optional[int] = None,
    ) -> str:
        dim_label = (
            str(self.max_frame_dimention)
            if self.max_frame_dimention is not None
            else "orig"
        )
        folder_name = f"fps_{fps}_dim_{dim_label}"
        if start_time is not None and duration_sec is not None:
            folder_name += f"_start_{start_time}_duration_{duration_sec}"
        if max_frames is not None:
            folder_name += f"_max{max_frames}"
        cache_dir = os.path.join(self.frame_dir, folder_name)

        os.makedirs(cache_dir, exist_ok=True)
        return cache_dir

    def _uniform_sample_indices(self, values: List[int], target_size: int) -> List[int]:
        if target_size <= 0:
            return []
        if target_size >= len(values):
            return values
        if target_size == 1:
            return [values[0]]
        last = len(values) - 1
        return [values[int(i * last / (target_size - 1))] for i in range(target_size)]

    def _ensure_frames_cache(
        self,
        fps: int,
        start_time: Optional[float] = None,
        duration_sec: Optional[float] = None,
        max_frames: Optional[int] = None,
    ) -> List[Tuple[str, float]]:
        """Return [(frame_path, timestamp_sec), ...] for the requested window.

        Timestamps are the frames' true positions in the source video. When the
        window is capped to `max_frames`, frames are sampled *uniformly across the
        whole window* (not truncated to the first N), so the timestamps are
        persisted alongside the PNGs (frames are renumbered sequentially and can't
        be recovered from filenames alone).
        """
        self._ensure_backend()
        cache_dir = self._fps_cache_dir(fps, start_time, duration_sec, max_frames)
        done_marker = os.path.join(cache_dir, ".done")
        times_path = os.path.join(cache_dir, "times.json")

        if os.path.exists(done_marker):
            cached = sorted(glob.glob(os.path.join(cache_dir, "frame_*.png")))
            if cached:
                return list(zip(cached, self._load_times(times_path, cached, fps, start_time)))

        # Rebuild cache for this fps bucket.
        for existing in glob.glob(os.path.join(cache_dir, "frame_*.png")):
            try:
                os.remove(existing)
            except OSError:
                pass
        if os.path.exists(done_marker):
            os.remove(done_marker)

        if self._backend == "decord" and self._vr is not None:
            timestamps = self._extract_frames_decord(fps, start_time, duration_sec, cache_dir, max_frames)
        else:
            timestamps = self._extract_frames_ffmpeg(fps, start_time, duration_sec, cache_dir, max_frames)

        paths = sorted(glob.glob(os.path.join(cache_dir, "frame_*.png")))
        # Guard against any count drift (e.g. ffmpeg emitting +/-1 frame).
        if len(timestamps) != len(paths):
            base = start_time or 0.0
            timestamps = [base + i / float(fps) for i in range(len(paths))]
        with open(times_path, "w", encoding="utf-8") as f:
            json.dump(timestamps, f)
        with open(done_marker, "w", encoding="utf-8") as f:
            f.write("ok")
        return list(zip(paths, timestamps))

    @staticmethod
    def _load_times(
        times_path: str, paths: List[str], fps: int, start_time: Optional[float]
    ) -> List[float]:
        try:
            with open(times_path, encoding="utf-8") as f:
                times = json.load(f)
            if len(times) == len(paths):
                return times
        except (OSError, ValueError):
            pass
        base = start_time or 0.0
        return [base + i / float(fps) for i in range(len(paths))]

    def _extract_frames_decord(
        self,
        fps: int,
        start_time: Optional[float],
        duration_sec: Optional[float],
        cache_dir: str,
        max_frames: Optional[int] = None,
    ) -> List[float]:
        print(
            f"Extracting frames for {fps} fps, start_time: {start_time}, "
            f"duration_sec: {duration_sec}, max_frames: {max_frames} using decord backend"
        )
        vr = self._vr
        native_fps = self._avg_fps
        total_frames = len(vr)
        if total_frames == 0:
            return []
        step_sec = 1.0 / float(fps)
        t = start_time if start_time is not None else 0.0
        end_sec = (
            t + duration_sec if duration_sec is not None else self._duration_sec
        )
        # Keep each sample's true time paired with its frame index, so capping
        # below preserves accurate timestamps.
        seen: set = set()
        pairs: List[Tuple[float, int]] = []
        while t < end_sec:
            idx = int(round(t * native_fps))
            idx = max(0, min(idx, total_frames - 1))
            if idx not in seen:
                seen.add(idx)
                pairs.append((t, idx))
            t += step_sec

        # Apply max_frames early (uniformly across the window) so we never
        # decode/hold more than needed.
        if max_frames is not None and len(pairs) > max_frames:
            pairs = self._uniform_sample_indices(pairs, max_frames)

        timestamps = [p[0] for p in pairs]
        frame_indices = [p[1] for p in pairs]

        pil_image = importlib.import_module("PIL.Image")
        out_idx = 0
        # Decode in small batches to keep peak memory bounded.
        for batch_start in range(0, len(frame_indices), self.DECORD_BATCH_SIZE):
            batch_indices = frame_indices[batch_start: batch_start + self.DECORD_BATCH_SIZE]
            batch = vr.get_batch(batch_indices).asnumpy()
            for rgb_frame in batch:
                image = pil_image.fromarray(rgb_frame)
                if self.max_frame_dimention is not None:
                    image.thumbnail(
                        (self.max_frame_dimention, self.max_frame_dimention),
                        pil_image.Resampling.LANCZOS,
                    )
                image.save(
                    os.path.join(cache_dir, f"frame_{out_idx:09d}.png"), format="PNG"
                )
                out_idx += 1
            del batch  # release the numpy array before the next batch
        return timestamps

    def _extract_frames_ffmpeg(
        self,
        fps: int,
        start_time: Optional[float],
        duration_sec: Optional[float],
        cache_dir: str,
        max_frames: Optional[int] = None,
    ) -> List[float]:
        base = start_time or 0.0
        # Window length, used both to cap frames uniformly and to time them.
        if duration_sec is not None and duration_sec > 0:
            window = duration_sec
        elif self._duration_sec is not None:
            window = max(0.0, self._duration_sec - base)
        else:
            window = None

        # Cap by *lowering the sampling fps* so frames stay uniform across the
        # whole window, instead of `-frames:v N` which truncates to the first N.
        effective_fps = float(fps)
        if max_frames is not None and window and window > 0:
            effective_fps = min(effective_fps, max_frames / window)
            if effective_fps <= 0:
                effective_fps = float(fps)

        print(
            f"Extracting frames for {effective_fps:.4f} fps (requested {fps}), "
            f"start_time: {start_time}, duration_sec: {duration_sec}, "
            f"max_frames: {max_frames} using ffmpeg backend"
        )
        output_pattern = os.path.join(cache_dir, "frame_%09d.png")
        cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y"]
        if start_time is not None and start_time > 0:
            cmd += ["-ss", str(start_time)]
        if duration_sec is not None and duration_sec > 0:
            cmd += ["-t", str(duration_sec)]
        cmd += ["-i", self.video_path]
        vf_parts = [f"fps={effective_fps}"]
        if self.max_frame_dimention is not None:
            vf_parts.append(
                f"scale={self.max_frame_dimention}:{self.max_frame_dimention}:"
                "force_original_aspect_ratio=decrease"
            )
        cmd += ["-vf", ",".join(vf_parts)]
        cmd.append(output_pattern)
        subprocess.run(cmd, check=True, capture_output=True, text=True)

        paths = sorted(glob.glob(os.path.join(cache_dir, "frame_*.png")))
        return [base + i / effective_fps for i in range(len(paths))]

    def fetch_frames(
        self,
        fps: int = FPS,
        start_time_sec: float = 0,
        end_time_sec: float = None,
        max_frames: Optional[int] = None,
    ) -> List[Frame]:
        """
        Fetch sampled frames and persist them to cache for reuse.
        """
        if fps <= 0:
            raise ValueError("fps must be > 0")
        if start_time_sec < 0:
            raise ValueError("start_time must be >= 0")
        if end_time_sec is not None and end_time_sec <= start_time_sec:
            raise ValueError("end_time must be greater than start_time")

        # start_sec, end_sec = self._normalize_time_window(start_time_ms, end_time_ms)
        duration_sec = (
            end_time_sec - start_time_sec if end_time_sec is not None else None
        )
        # Pass max_frames into the cache layer so extraction is capped at the
        # source, preventing OOM on long videos (e.g. 2400s @ 1fps with no limit
        # would otherwise decode ~2400 frames). Returns (path, true_timestamp).
        frame_items = self._ensure_frames_cache(
            fps, start_time_sec, duration_sec, max_frames
        )
        if len(frame_items) == 0:
            return []

        frames: List[Frame] = []
        for idx, (frame_path, timestamp) in enumerate(frame_items):
            frames.append(
                Frame(
                    frame_file_path=frame_path,
                    timestamp=timestamp,
                    video_id=self.video_id,
                    id=f"{self.video_id}_fps{fps}_frame_{idx}",
                )
            )

        return frames

    def fetch_frames_between_timestamps(
        self, start_time: float, end_time: float, max_frames: Optional[int] = None
    ) -> List[Frame]:
        """
        Fetch sampled frames between start and end times.
        """
        return self.fetch_frames(
            fps=self.FPS,
            start_time=start_time,
            end_time=end_time,
            max_frames=max_frames,
        )

    def _fetch_clip(
        self, start_time: float, duration_sec: float, fps: int = 5, crf=None
    ) -> Clip:
        """
        Fetch the clip between the start and end times using ffmpeg.
        """
        max_dimentions = self.max_frame_dimention
        duration_sec = (
            duration_sec
            if duration_sec <= self.MAX_CLIP_DURATION_SEC
            else self.MAX_CLIP_DURATION_SEC
        )
        clip_path = os.path.join(
            self.frame_dir, f"clip_{start_time}_{duration_sec}secs.mp4"
        )

        cmd = [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-ss",
            str(start_time),
            "-t",
            str(duration_sec),
            "-i",
            self.video_path,
        ]
        if crf is not None:
            cmd.extend(["-c:v", "libx264", "-crf", str(crf), "-preset", "fast"])
        vf_parts = []
        if max_dimentions is not None:
            vf_parts.append(
                f"scale={max_dimentions}:{max_dimentions}:force_original_aspect_ratio=decrease"
            )
        if fps is not None:
            vf_parts.append(f"fps={fps}")
        if len(vf_parts) > 0:
            cmd.append("-vf")
            cmd.append(",".join(vf_parts))

        cmd.append(clip_path)
        subprocess.run(cmd, check=True, capture_output=True, text=True)
        return Clip(
            clip_file_path=clip_path,
            start_time=start_time,
            end_time=start_time + duration_sec,
            video_id=self.video_id,
            id=f"{self.video_id}_clip_{start_time}_{duration_sec}secs",
        )

    async def skim_frames(
        self,
        query: str,
        start_time: float,
        end_time: float,
        max_frames: Optional[int] = None,
    ) -> List[Frame]:
        """
        Skim the frames between the start and end times.

        Args:
            start_time: The start time of the frames to skim in milliseconds.
            end_time: The end time of the frames to skim in milliseconds.
            max_frames: The maximum number of frames to skim. If not provided, the default is 50.
        """
        duration_ms = end_time - start_time
        num_frames = int((duration_ms * self.SKIM_FPS) // 1000)
        capped_max = min(num_frames, self.SKIM_MAX_FRAMES)

        if max_frames is not None:
            capped_max = min(capped_max, max_frames)

        frames = self.fetch_frames(
            fps=self.SKIM_FPS,
            start_time=start_time,
            end_time=end_time,
            max_frames=capped_max,
        )
        analysis = await self.analyze_frames(query, frames)
        print(f"Analysis: {analysis}")
        return analysis, []

    async def focus_frames(
        self,
        query: str,
        start_time: float,
        end_time: float,
        max_frames: Optional[int] = None,
    ) -> List[Frame]:
        """
        Focus the frames between the start and end times.
        """
        duration_ms = end_time - start_time
        num_frames = int((duration_ms * self.FOCUS_FPS) // 1000)
        capped_max = min(num_frames, self.FOCUS_MAX_FRAMES)
        if max_frames is not None:
            capped_max = min(capped_max, max_frames)
        frames = self.fetch_frames(
            fps=self.FOCUS_FPS,
            start_time=start_time,
            end_time=end_time,
            max_frames=capped_max,
        )

        analysis = await self.analyze_frames(query, frames)
        print(f"Analysis: {analysis}")
        return analysis, []

    def get_overview_frames(self) -> List[Frame]:
        """
        Get the overview frames of the video.
        """
        print(f"Getting overview frames for {self.video_id}")
        frames = self.fetch_frames(
            fps=self.OVERVIEW_FPS, max_frames=self.OVERVIEW_MAX_FRAMES
        )
        return frames

    async def fetch_clip(
        self,
        start_time_sec: float,
        end_time_sec: float,
        fps: int = 5,
        crf: int = None,
        max_size_mb: Optional[int] = None,
    ) -> str:
        # start_time, end_time = self._normalize_time_window(start_time_ms, end_time_ms)
        duration_sec = end_time_sec - start_time_sec
        if duration_sec > self.MAX_CLIP_DURATION_SEC:
            duration_sec = self.MAX_CLIP_DURATION_SEC
        clip = self._fetch_clip(start_time_sec, duration_sec, fps=fps, crf=crf)
        print(f"Clip size: {os.path.getsize(clip.clip_file_path) / 1024 / 1024}MB")
        print(f"Max size: {max_size_mb}MB")
        if (
            max_size_mb is not None
            and os.path.getsize(clip.clip_file_path) / 1024 / 1024 > max_size_mb
        ):
            print(
                f"Clip size is too large > {max_size_mb}MB, reducing encoding quality"
            )
            self.max_frame_dimention = 480
            clip = self._fetch_clip(start_time_sec, duration_sec, fps=1, crf=28)
            print(
                f"Reduced clip size to {os.path.getsize(clip.clip_file_path) / 1024 / 1024}MB"
            )
        if (
            max_size_mb is not None
            and os.path.getsize(clip.clip_file_path) / 1024 / 1024 > max_size_mb
        ):
            print(f"Clip size is still too large > {max_size_mb}MB, returning None")
            raise ValueError(f"Clip size is still too large > {max_size_mb}MB")
        return clip

    async def fetch_clips(
        self,
        start_time_sec: float,
        end_time_sec: float,
        fps: int = 5,
        crf: int = None,
        max_size_mb: Optional[int] = None,
    ) -> Tuple[List["Clip"], float]:
        """Return ``([clip], global_offset)`` — interface parity with the e2b backend.

        The inprocess backend has no tile manifest, so it always returns a single
        on-demand clip; its global start is the citation offset.
        """
        clip = await self.fetch_clip(start_time_sec, end_time_sec, fps=fps, crf=crf, max_size_mb=max_size_mb)
        offset = clip.start_time if clip.start_time is not None else start_time_sec
        return [clip], offset

    async def analyze_frames(self, query: str, frames: List[Frame]) -> str:
        llm_response = await llm_call(
            prompt=self.PROMPT,
            query=f"Analyze the frames of the video for the query: {query}",
            model=settings.llm_model,
            base_url=settings.llm_base_url,
            api_key=settings.llm_api_key,
            video_frames=frames,
        )
        return llm_response["choices"][0]["message"]["content"]


if __name__ == "__main__":
    import time

    start_time = time.time()
    video_tools = VideoFrameTools("uk_tv_show", max_frame_dimention=768)
    frames = video_tools.get_overview_frames()
    print([frame.frame_file_path for frame in frames])
    end_time = time.time()
    print(f"Time taken: {end_time - start_time} seconds")

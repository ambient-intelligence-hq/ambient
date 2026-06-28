from typing import List, Dict
import re
from ambient.tools.inprocess_video_tools import VideoFrameTools
from ambient.llm import video_to_data_url


def _format_timestamp(timestamp: float) -> str:
    hours = int(timestamp // 3600)
    minutes = int((timestamp % 3600) // 60)
    seconds = int(timestamp % 60)
    if hours > 0:
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"
    else:
        return f"{minutes:02d}:{seconds:02d}"

def _extract_temporal_citations(text: str) -> List[Dict[str, float]]:
    """
    Parse citations from model text.
    Supported forms inside <citation>...</citation>:
    - 12:05
    - 12:05::18:20
    - 00:00:58::00:01:02 (hh:mm:ss::hh:mm:ss)
    """
    citations: List[Dict[str, float]] = []
    for raw in re.findall(r"<citation>\s*(.*?)\s*</citation>", text, flags=re.IGNORECASE | re.DOTALL):
        content = raw.strip().lower()

        frame_match = re.match(r"(\d+):(\d+)$", content)
        if frame_match:
            ts = float(frame_match.group(1)) * 60 + float(frame_match.group(2))
            if ts is not None:
                citations.append({"type": "frame", "start": ts, "end": ts,"raw": content})
            continue

        # hour frame match
        hour_frame_match = re.match(r"(\d+):(\d+):(\d+)$", content)
        if hour_frame_match:
            ts = float(hour_frame_match.group(1)) * 3600 + float(hour_frame_match.group(2)) * 60 + float(hour_frame_match.group(3))
            if ts is not None:
                citations.append({"type": "frame", "start": ts, "end": ts,"raw": content})
            continue

        range_match = re.match(r"(\d+):(\d+)::(\d+):(\d+)$", content)
        if range_match:
            start = float(range_match.group(1)) * 60 + float(range_match.group(2))
            end = float(range_match.group(3)) * 60 + float(range_match.group(4))
            if start is not None and end is not None:
                citations.append({"type": "range", "start": start, "end": end,"raw": content})
            continue

        # hour match
        hour_match = re.match(r"(\d+):(\d+):(\d+)::(\d+):(\d+):(\d+)$", content)
        if hour_match:
            start = float(hour_match.group(1)) * 3600 + float(hour_match.group(2)) * 60 + float(hour_match.group(3))
            end = float(hour_match.group(4)) * 3600 + float(hour_match.group(5)) * 60 + float(hour_match.group(6))
            if start is not None and end is not None:
                citations.append({"type": "range", "start": start, "end": end,"raw": content})
            continue

    return citations

def  _replace_citations_with_global_video_timestamps(text: str, citations: List[Dict[str, float]], start_time: float) -> str:
    for citation in citations:
        if citation["type"] == "frame":
            text = text.replace(citation["raw"], f"Local Clip: {citation['raw']}, Global Video: {_format_timestamp(citation['start']+start_time)}")
        elif citation["type"] == "range":
            text = text.replace(citation["raw"], f"Local Clip: {citation['raw']}, Global Video: {_format_timestamp(citation['start']+start_time)}::{_format_timestamp(citation['end']+start_time)}")
    return text


def _build_user_message_contents_from_citations(
    video_tools: VideoFrameTools,
    citations: List[Dict[str, float]],
    start_time: float,
    end_time: float,
) -> List[Dict]:
    user_message_contents: List[Dict] = []
    if len(citations) == 0:
        return user_message_contents

    seen_frame_ids = set()
    for citation in citations:
        ctype = citation["type"]
        start = max(0.0, float(citation["start"]))
        end = max(start, float(citation["end"]))

        if ctype == "frame":
            # Use a tiny window around the cited timestamp to capture one representative frame.
            frames = video_tools.fetch_frames(
                fps=2,
                start_time_sec=start,
                end_time_sec=start + 1.0,
                max_frames=1,
            )
        else:
            duration = max(1.0, end - start)
            print(
                f"Fetching frames for duration: {duration} seconds from {start} to {start + duration} seconds"
            )
            # Uniformly sampled frames for cited temporal ranges.
            frames = video_tools.fetch_frames(
                fps=2,
                start_time_sec=start,
                end_time_sec=start + duration,
                max_frames=6,
            )

        for frame in frames:
            if frame.id in seen_frame_ids:
                continue
            seen_frame_ids.add(frame.id)
            user_message_contents.append(
                {
                    "type": "text",
                    "text": f"Citation frame at Video Timestamp: {start_time + frame.timestamp:.3f} seconds ( Clip time stamp : {frame.timestamp:.3f} seconds ) ",
                }
            )
            user_message_contents.append(
                {
                    "type": "image_url",
                    "image_url": {
                        "url": video_to_data_url(frame.frame_file_path, "image/jpeg")
                    },
                }
            )

    return user_message_contents

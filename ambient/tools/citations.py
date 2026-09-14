from typing import Any, List, Dict
import re
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
    for raw in re.findall(
        r"<citation>\s*(.*?)\s*</citation>", text, flags=re.IGNORECASE | re.DOTALL
    ):
        content = raw.strip().lower()

        frame_match = re.match(r"(\d+):(\d+)$", content)
        if frame_match:
            ts = float(frame_match.group(1)) * 60 + float(frame_match.group(2))
            if ts is not None:
                citations.append(
                    {"type": "frame", "start": ts, "end": ts, "raw": content}
                )
            continue

        # hour frame match
        hour_frame_match = re.match(r"(\d+):(\d+):(\d+)$", content)
        if hour_frame_match:
            ts = (
                float(hour_frame_match.group(1)) * 3600
                + float(hour_frame_match.group(2)) * 60
                + float(hour_frame_match.group(3))
            )
            if ts is not None:
                citations.append(
                    {"type": "frame", "start": ts, "end": ts, "raw": content}
                )
            continue

        range_match = re.match(r"(\d+):(\d+)::(\d+):(\d+)$", content)
        if range_match:
            start = float(range_match.group(1)) * 60 + float(range_match.group(2))
            end = float(range_match.group(3)) * 60 + float(range_match.group(4))
            if start is not None and end is not None:
                citations.append(
                    {"type": "range", "start": start, "end": end, "raw": content}
                )
            continue

        # hour match
        hour_match = re.match(r"(\d+):(\d+):(\d+)::(\d+):(\d+):(\d+)$", content)
        if hour_match:
            start = (
                float(hour_match.group(1)) * 3600
                + float(hour_match.group(2)) * 60
                + float(hour_match.group(3))
            )
            end = (
                float(hour_match.group(4)) * 3600
                + float(hour_match.group(5)) * 60
                + float(hour_match.group(6))
            )
            if start is not None and end is not None:
                citations.append(
                    {"type": "range", "start": start, "end": end, "raw": content}
                )
            continue

    return citations


def parse_frame_annotations(text, clip_start: float):
    """Return [{'timestamp': float, 'global_timestamp': float, 'local_timestamp': str, 'bounding_box': [y,x,y,x], 'label': str|None}, ...] from the model text."""
    blocks = re.findall(r"<annotation>(.*?)</annotation>", text, re.S | re.I) or [text]
    out = []
    for blk in blocks:
        ts_match = re.search(
            r"<timestamp>\s*(.*?)\s*</timestamp>",
            blk,
            flags=re.IGNORECASE | re.DOTALL,
        )
        if not ts_match:
            continue

        timestamp = _parse_timestamp(ts_match.group(1).strip())
        m = re.search(r"<bounding_box>\s*(.*?)\s*</bounding_box>", blk, re.S | re.I)
        target = m.group(1) if m else blk
        nums = [int(n) for n in re.findall(r"-?\d+", target)]
        lbl = re.search(r"<label>\s*(.*?)\s*</label>", blk, re.S | re.I)
        if len(nums) == 4:
            out.append(
                {
                    "timestamp": timestamp,
                    "global_timestamp": timestamp,
                    "local_timestamp": ts_match.group(1).strip(),
                    "bounding_box": nums,
                    "label": lbl.group(1).strip() if lbl else None,
                }
            )
    return out


def _parse_timestamp(content: str) -> float:
    """
    Parse a mm:ss or hh:mm:ss timestamp into seconds. Returns None if unparseable.
    """
    content = content.strip().lower()

    hour_match = re.match(r"(\d+):(\d+):(\d+)$", content)
    if hour_match:
        return (
            float(hour_match.group(1)) * 3600
            + float(hour_match.group(2)) * 60
            + float(hour_match.group(3))
        )

    frame_match = re.match(r"(\d+):(\d+)$", content)
    if frame_match:
        return float(frame_match.group(1)) * 60 + float(frame_match.group(2))

    return None


def _extract_frame_annotations(text: str, clip_start: float) -> List[Dict]:
    """
    Parse frames and annotations from model text produced with GRAB_FRAMES_TOOL_PROMPT.

    Expected form inside <frames>...</frames>:
        <frame>
            <timestamp>12:05</timestamp>
            <annotation>
                <bounding_box>[y_min, x_min, y_max, x_max]</bounding_box>
                <label>description of the annotation</label>
            </annotation>
        </frame>

    The <annotation> block is optional. Returns a list of dicts, each shaped like:
        {
            "timestamp": 725.0,          # seconds, None if unparseable
            "raw_timestamp": "12:05",
            "annotations": [
                {"bounding_box": [y_min, x_min, y_max, x_max], "label": "..."},
                ...
            ],
        }
    """
    frames: List[Dict] = []

    for frame_body in re.findall(
        r"<frame>\s*(.*?)\s*</frame>", text, flags=re.IGNORECASE | re.DOTALL
    ):
        ts_match = re.search(
            r"<timestamp>\s*(.*?)\s*</timestamp>",
            frame_body,
            flags=re.IGNORECASE | re.DOTALL,
        )
        if not ts_match:
            continue

        raw_timestamp = ts_match.group(1).strip()
        timestamp = _parse_timestamp(raw_timestamp)

        annotations: List[Dict] = []
        for annotation_body in re.findall(
            r"<annotation>\s*(.*?)\s*</annotation>",
            frame_body,
            flags=re.IGNORECASE | re.DOTALL,
        ):
            box_match = re.search(
                r"<bounding_box>\s*(.*?)\s*</bounding_box>",
                annotation_body,
                flags=re.IGNORECASE | re.DOTALL,
            )
            label_match = re.search(
                r"<label>\s*(.*?)\s*</label>",
                annotation_body,
                flags=re.IGNORECASE | re.DOTALL,
            )

            bounding_box = None
            if box_match:
                bounding_box = [
                    int(n) for n in re.findall(r"-?\d+", box_match.group(1))
                ] or None

            label = label_match.group(1).strip() if label_match else None

            if bounding_box is None and label is None:
                continue

            annotations.append({"bounding_box": bounding_box, "label": label})

        frames.append(
            {
                "global_timestamp": timestamp + clip_start,
                "local_timestamp": raw_timestamp,
                "annotations": annotations,
            }
        )

    return frames


def _replace_citations_with_global_video_timestamps(
    text: str, citations: List[Dict[str, float]], start_time: float
) -> str:
    for citation in citations:
        if citation["type"] == "frame":
            text = text.replace(
                citation["raw"],
                f"Local Clip: {citation['raw']}, Global Video: {_format_timestamp(citation['start']+start_time)}",
            )
        elif citation["type"] == "range":
            text = text.replace(
                citation["raw"],
                f"Local Clip: {citation['raw']}, Global Video: {_format_timestamp(citation['start']+start_time)}::{_format_timestamp(citation['end']+start_time)}",
            )
    return text


def _build_user_message_contents_from_citations(
    video_tools: Any,
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
                        "url": frame.frame_url or video_to_data_url(frame.frame_file_path, "image/jpeg")
                    },
                }
            )

    return user_message_contents

"""Capture real Ambient sessions into static fixtures for the read-only Replay Gallery.

For each demo session this:
  1. pulls the mapped Managed-Agents event log (GET /v1/sessions/{id}/events),
  2. downloads the source video (GET /v1/files/{id}/content), web-optimises it,
     and extracts a poster frame,
  3. extracts the frames the agent looked at (tool-result `attachments`, which are
     either base64 data strings or short-lived presigned R2 URLs),
  4. uploads source + poster + frames to R2 under `replay/<slug>/`, rewriting each
     attachment URL in the event log to the durable relative key, and
  5. writes `ambient/studio/replay/src/fixtures/<slug>.json`.

The fixture stores *relative* media keys (e.g. `replay/<slug>/source.mp4`); the
replay app prepends `VITE_MEDIA_BASE` (the bucket's public r2.dev / custom-domain
URL) at build time. A sidecar `fixtures/_presigned.json` with 7-day URLs is also
written so the app can be previewed before the bucket is made public.

Run from the repo root (needs the local engine on :8080 and ffmpeg on PATH):
    uv run python scripts/capture_replay.py                 # all sessions
    uv run python scripts/capture_replay.py safety pose      # a subset (by slug)
    uv run python scripts/capture_replay.py --skip-media     # events only (fast)
"""

from __future__ import annotations

import base64
import json
import os
import re
import subprocess
import sys
import tempfile
import urllib.request
from pathlib import Path
from typing import Any

from ambient.utils.s3 import get_s3_client

ENGINE_URL = os.environ.get("ENGINE_URL", "http://127.0.0.1:8080").rstrip("/")
ENGINE_API_KEY = os.environ.get("ENGINE_API_KEY", "dev-token")

REPO_ROOT = Path(__file__).resolve().parent.parent
# The replay app now lives in the standalone website repo. Override with
# REPLAY_FIXTURE_DIR; defaults to the sibling `ambient-website` checkout.
FIXTURE_DIR = Path(
    os.environ.get("REPLAY_FIXTURE_DIR")
    or REPO_ROOT.parent / "ambient-website" / "replay" / "src" / "fixtures"
)
MEDIA_PREFIX = "replay"  # R2 key prefix for all replay media
PRESIGN_TTL = 7 * 24 * 3600  # SigV4 max — for local preview only

# Curated metadata (session titles are null in the DB). `poster_at` is the source
# timestamp (seconds) used for the gallery thumbnail.
# Order here = gallery order (index.json is built by iterating this dict).
SESSIONS: dict[str, dict] = {
    "swing-pose-estimation": {
        "session_id": "ses_f79fa5c206f446cf",
        "title": "Golf-swing body-pose overlay",
        "summary": "Adds body-pose skeleton points to a swing and renders a new video.",
        "poster_at": 2.0,
    },
    "highlight-reel": {
        "session_id": "ses_bc902ec2bd8a4a6f",
        "title": "30-second social highlight reel",
        "summary": "Builds a story-driven highlight reel with HyperFrames from a long video.",
        "poster_at": 2.0,
    },
    "video-qa-walkthrough": {
        "session_id": "ses_d2eb9690f85d409d",
        "title": "Tracking a person across frames in a video",
        "summary": "Tracks a person over the full video with YOLOv8 + ByteTrack, exporting an annotated video and per-frame tracks.",
        "poster_at": 2.0,
    },
    "video-to-documentation": {
        "session_id": "ses_c2863c5ab78b4899",
        "title": "Walkthrough → product documentation",
        "summary": "Converts a product demo video into step-by-step docs with screenshots.",
        "poster_at": 5.0,
    },
    "recipe-blog-post": {
        "session_id": "ses_1e90c58b5ed24e36",
        "title": "Cooking video → recipe blog post",
        "summary": "Writes a recipe blog post with step frames pulled from the video.",
        "poster_at": 4.0,
    },
    "flow-diagram-gif": {
        "session_id": "ses_578b260c86834109",
        "title": "Flow-diagram explainer GIF",
        "summary": "Finds the flow-diagram moment and cuts it into a looping GIF.",
        "poster_at": 2.0,
    },
    "safety-inspection-report": {
        "session_id": "ses_ea9194d0c1cc494d",
        "title": "Industrial safety inspection report",
        "summary": "Turns a factory-floor walkthrough into a hazard report with evidence frames.",
        "poster_at": 3.0,
    },
}


def _req(path: str) -> dict:
    r = urllib.request.Request(
        f"{ENGINE_URL}/v1{path}", headers={"x-api-key": ENGINE_API_KEY}
    )
    with urllib.request.urlopen(r, timeout=120) as resp:
        return json.load(resp)


def fetch_events(session_id: str) -> list[dict]:
    """All events for a session, paged by seq so a server-side limit can't truncate."""
    out: list[dict] = []
    after = 0
    while True:
        batch = _req(f"/sessions/{session_id}/events?after_seq={after}").get("data", [])
        if not batch:
            break
        out.extend(batch)
        last = batch[-1].get("seq")
        if last is None or last == after:
            break
        after = last
        if len(batch) < 2:  # tail reached
            break
    return out


def session_meta(session_id: str) -> dict:
    return _req(f"/sessions/{session_id}")


def download(url_or_path: str, dest: Path, is_engine_file: bool = False) -> None:
    headers = {"x-api-key": ENGINE_API_KEY} if is_engine_file else {}
    r = urllib.request.Request(url_or_path, headers=headers)
    with urllib.request.urlopen(r, timeout=600) as resp, open(dest, "wb") as f:
        while True:
            chunk = resp.read(1 << 20)
            if not chunk:
                break
            f.write(chunk)


def ffprobe_duration(path: Path) -> float:
    try:
        out = subprocess.check_output(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1", str(path)],
            text=True,
        )
        return round(float(out.strip()), 2)
    except Exception:
        return 0.0


def web_optimise(src: Path, dst: Path) -> None:
    """Downscale to <=1280w, H.264 + AAC, faststart — demo-friendly size."""
    subprocess.check_call([
        "ffmpeg", "-y", "-i", str(src),
        "-vf", "scale='min(1280,iw)':-2",
        "-c:v", "libx264", "-crf", "28", "-preset", "veryfast",
        "-c:a", "aac", "-b:a", "128k",
        "-movflags", "+faststart",
        str(dst),
    ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def extract_poster(src: Path, dst: Path, at: float) -> None:
    subprocess.check_call([
        "ffmpeg", "-y", "-ss", str(at), "-i", str(src),
        "-frames:v", "1", "-vf", "scale='min(1280,iw)':-2", str(dst),
    ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


_ARTIFACT_RX = re.compile(r"/video-analysis-tests/(artifacts/[^\s\"'\)\]<>?]+)")
_ARTIFACT_S3 = re.compile(r"s3://video-analysis-tests/(artifacts/[^\s\"'\)\]<>?]+)")
_MIME = {
    ".gif": "image/gif", ".mp4": "video/mp4", ".webm": "video/webm",
    ".json": "application/json", ".html": "text/html", ".htm": "text/html",
    ".zip": "application/zip", ".png": "image/png", ".jpg": "image/jpeg",
    ".csv": "text/csv", ".pdf": "application/pdf", ".md": "text/markdown",
}


# Deliverables whose link lives only in the console's saved answer (the engine
# run was truncated / resumed, so the artifact URL isn't in the engine event log).
# Captured from the Studio Message store; keys verified to exist in R2.
EXTRA_ARTIFACTS: dict[str, list[str]] = {
    "safety-inspection-report": [
        "artifacts/vid_200bca4a43ae0fa7/Industrial_Safety_Report_1791530978.html",
        "artifacts/vid_200bca4a43ae0fa7/Safety_Report_and_Evidence_Frames_1791530978.zip",
    ],
    "highlight-reel": [
        "artifacts/vid_f2c1386d8b046f54/canva-30s-social_1791541292.mp4",
        "artifacts/vid_f2c1386d8b046f54/canva-hyperframes-project_1791541292.zip",
    ],
    "video-to-documentation": [
        "artifacts/vid_f2c1386d8b046f54/canva-product-guide_1791539654.html",
    ],
    "video-qa-walkthrough": [
        "artifacts/vid_5e8d39727c286702/tracked_person_yolov8_bytetrack_1792930921.mp4",
        "artifacts/vid_5e8d39727c286702/tracks_final_1792930921.csv",
    ],
}


def _content_text(block: Any) -> str:
    if isinstance(block, list):
        return "".join(b.get("text", "") for b in block if isinstance(b, dict))
    return str(block or "")


def scan_artifact_keys(events: list[dict]) -> list[str]:
    """Deliverable object keys parsed from the answer text (R2 URLs / s3:// paths)."""
    keys: list[str] = []
    seen: set[str] = set()
    for e in events:
        if e.get("type") not in ("agent.message.delta", "agent.message"):
            continue
        t = _content_text(e.get("content"))
        for m in _ARTIFACT_RX.findall(t) + _ARTIFACT_S3.findall(t):
            k = m.rstrip(".,);\"' ")
            if k and k not in seen:
                seen.add(k)
                keys.append(k)
    return keys


def materialize_artifacts(slug: str, src_keys: list[str], s3, presigned: dict) -> list[dict]:
    """Copy each existing artifact object into the replay app's own durable prefix
    (`replay/<slug>/artifacts/<name>`, a server-side R2 copy — no download) so the
    links stay valid indefinitely via the public bucket base, independent of the
    original `artifacts/` object and the agent's long-expired presigned URL. Also
    presigns the durable key for dev preview."""
    out: list[dict] = []
    seen: set[str] = set()
    for src in src_keys:
        if src in seen or not s3.object_exists(src):
            continue
        seen.add(src)
        name = src.split("/")[-1]
        dest = f"{MEDIA_PREFIX}/{slug}/artifacts/{name}"
        try:
            s3.s3.copy_object(
                Bucket=s3.bucket,
                CopySource={"Bucket": s3.bucket, "Key": src},
                Key=dest,
                MetadataDirective="COPY",
            )
        except Exception as e:  # noqa: BLE001
            print(f"   ! copy failed for {name}: {e}")
            continue
        ext = os.path.splitext(name)[1].lower()
        out.append({"key": dest, "name": name, "mime": _MIME.get(ext, "application/octet-stream")})
        presigned[dest] = s3.get_presigned_url(dest, expires_in=PRESIGN_TTL)
    # Primary first: prefer a viewable deliverable (video/html/gif) over an archive.
    out.sort(key=lambda a: (a["mime"] in ("application/zip", "application/json", "text/csv"), a["name"]))
    return out


def attachment_bytes(url: str) -> bytes | None:
    """An attachment `url` is either a data: / raw-base64 string or an http(s) URL."""
    try:
        if url.startswith("http"):
            with urllib.request.urlopen(url, timeout=120) as resp:
                return resp.read()
        if url.startswith("data:"):
            url = url.split(",", 1)[1]
        return base64.b64decode(url)
    except Exception as e:  # noqa: BLE001
        print(f"      ! attachment fetch failed: {e}")
        return None


def capture(
    slug: str, meta: dict, skip_media: bool, upload_frames: bool, no_source: bool, s3, presigned: dict
) -> None:
    sid = meta["session_id"]
    print(f"\n== {slug}  ({sid}) ==")
    events = fetch_events(sid)
    sess = session_meta(sid)
    video_id = (sess.get("metadata") or {}).get("video_id")
    print(f"   events={len(events)} video={video_id}")

    out = FIXTURE_DIR / f"{slug}.json"
    existing = json.loads(out.read_text()) if out.exists() else {}

    media = {"source": None, "poster": None}
    frame_i = 0

    # --no-source: reuse already-uploaded source/poster + duration from the prior
    # fixture, so re-captures that only touch events/artifacts skip the heavy
    # video download + transcode + upload.
    if no_source and not skip_media:
        media = existing.get("media") or {
            "source": f"{MEDIA_PREFIX}/{slug}/source.mp4",
            "poster": f"{MEDIA_PREFIX}/{slug}/poster.jpg",
        }
        meta["_duration"] = existing.get("durationSec", 0.0)
        # Refresh the source/poster presigned URLs too, so a --no-source run
        # extends *every* R2 link (not just artifacts).
        for k in (media.get("source"), media.get("poster")):
            if k and s3.object_exists(k):
                presigned[k] = s3.get_presigned_url(k, expires_in=PRESIGN_TTL)

    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)

        if not skip_media and not no_source and video_id:
            raw = tmp / "raw.mp4"
            print("   downloading source…")
            download(f"{ENGINE_URL}/v1/files/{video_id}/content", raw, is_engine_file=True)
            dur = ffprobe_duration(raw)
            meta["_duration"] = dur

            opt = tmp / "source.mp4"
            print("   web-optimising source…")
            try:
                web_optimise(raw, opt)
            except Exception:  # noqa: BLE001 - fall back to raw if transcode fails
                opt = raw
            src_key = f"{MEDIA_PREFIX}/{slug}/source.mp4"
            s3.upload_file(str(opt), src_key, {"ContentType": "video/mp4"})
            media["source"] = src_key
            presigned[src_key] = s3.get_presigned_url(src_key, expires_in=PRESIGN_TTL)

            poster = tmp / "poster.jpg"
            try:
                extract_poster(opt, poster, meta.get("poster_at", 2.0))
                poster_key = f"{MEDIA_PREFIX}/{slug}/poster.jpg"
                s3.upload_file(str(poster), poster_key, {"ContentType": "image/jpeg"})
                media["poster"] = poster_key
                presigned[poster_key] = s3.get_presigned_url(poster_key, expires_in=PRESIGN_TTL)
            except Exception as e:  # noqa: BLE001
                print(f"   ! poster failed: {e}")

        # Rewrite attachment URLs in the event log to durable relative keys.
        for e in events:
            atts = e.get("attachments")
            if not atts:
                continue
            for a in atts:
                url = a.get("url")
                if not url or not isinstance(url, str):
                    continue
                if skip_media or not upload_frames:
                    # Drop the heavy base64 / soon-to-expire presigned URL. The
                    # console ignores attachments (tool results render as text),
                    # so this keeps fixtures small with no loss of fidelity.
                    a["url"] = None
                    continue
                data = attachment_bytes(url)
                if data is None:
                    a["url"] = None
                    continue
                key = f"{MEDIA_PREFIX}/{slug}/frames/{frame_i:04d}.jpg"
                fp = tmp / "f.jpg"
                fp.write_bytes(data)
                s3.upload_file(str(fp), key, {"ContentType": "image/jpeg"})
                presigned[key] = s3.get_presigned_url(key, expires_in=PRESIGN_TTL)
                a["url"] = key
                frame_i += 1
            if frame_i and frame_i % 50 == 0:
                print(f"   …{frame_i} frames uploaded")

    # First user message = the question shown in the gallery + chat.
    question = ""
    for e in events:
        if e.get("type") == "user.message":
            c = e.get("content")
            if isinstance(c, list):
                question = " ".join(b.get("text", "") for b in c if isinstance(b, dict))
            else:
                question = str(c or "")
            break

    usage = sess.get("usage") or {}
    model = next(iter((usage.get("by_model") or {}).keys()), None)

    # Deliverable artifacts: keys parsed from the answer plus any explicit ones
    # (whose link lived only in the console store). Copied into the replay app's
    # durable prefix so downloads never expire.
    artifacts = []
    if s3:
        src_keys = scan_artifact_keys(events) + EXTRA_ARTIFACTS.get(slug, [])
        artifacts = materialize_artifacts(slug, src_keys, s3, presigned)
        if artifacts:
            print(f"   artifacts: {[a['name'] for a in artifacts]}")

    fixture = {
        "slug": slug,
        "title": meta["title"],
        "summary": meta["summary"],
        "question": question.strip(),
        "model": model,
        "durationSec": meta.get("_duration", 0.0),
        "usage": usage,
        "media": media,
        "artifacts": artifacts,
        # `{event, data}` mirrors what the console's parseSSE yields, so the replay
        # engine can reuse the provider's event→UI-part translation verbatim.
        "events": [{"event": e.get("type"), "data": e} for e in events],
    }
    FIXTURE_DIR.mkdir(parents=True, exist_ok=True)
    out = FIXTURE_DIR / f"{slug}.json"
    out.write_text(json.dumps(fixture, indent=1))
    print(f"   wrote {out.relative_to(REPO_ROOT)}  ({out.stat().st_size // 1024} KB)")


def main() -> None:
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    flags = {a for a in sys.argv[1:] if a.startswith("--")}
    skip_media = "--skip-media" in flags
    upload_frames = "--frames" in flags
    no_source = "--no-source" in flags
    slugs = args or list(SESSIONS.keys())

    s3 = None if skip_media else get_s3_client()
    presigned: dict[str, str] = {}

    # merge with any existing presigned sidecar so partial runs accumulate
    sidecar = FIXTURE_DIR / "_presigned.json"
    if sidecar.exists():
        presigned.update(json.loads(sidecar.read_text()))

    for slug in slugs:
        if slug not in SESSIONS:
            print(f"!! unknown slug: {slug}")
            continue
        capture(slug, dict(SESSIONS[slug]), skip_media, upload_frames, no_source, s3, presigned)

    if not skip_media:
        FIXTURE_DIR.mkdir(parents=True, exist_ok=True)
        sidecar.write_text(json.dumps(presigned, indent=1))
        print(f"\nwrote {sidecar.relative_to(REPO_ROOT)}  ({len(presigned)} urls)")

    # A manifest the gallery index reads to list demos in a fixed order.
    manifest = [
        {"slug": s, "title": SESSIONS[s]["title"], "summary": SESSIONS[s]["summary"]}
        for s in SESSIONS
        if (FIXTURE_DIR / f"{s}.json").exists()
    ]
    (FIXTURE_DIR / "index.json").write_text(json.dumps(manifest, indent=1))
    print(f"wrote index.json  ({len(manifest)} demos)")


if __name__ == "__main__":
    main()

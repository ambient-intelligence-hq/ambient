# Spec: Stream the source video instead of downloading it

**Status:** Implemented — see **§11 (as-built)** for the final design and decisions, which supersede §3–§5 where they differ.
**Scope:** `ambient/sandboxes/e2b/templates/video-analysis-v1/main.py` (primary), with light touch on the host tools that invoke it.
**Author:** —
**Last updated:** 2026-06-26

---

## 1. Background

Every media operation in the E2B sandbox today begins by **downloading the entire
source video to local disk** via `resolve_video_path()`
([main.py:134](../ambient/sandboxes/e2b/templates/video-analysis-v1/main.py#L134)):
it probes extensions with `head_object`, calls `boto3 download_file`, then ffmpeg/decord
read the local copy.

There are two distinct read patterns, and **both pay the full download** regardless of
how little of the file they actually need:

| Caller | Sandbox command | What it actually needs | What it reads today |
|---|---|---|---|
| `search_clip` / `focus_clip` | `fetch-clip --start S --end E` | a single window (`-ss S -t (E-S)`) | the **whole file** |
| `get_video_description` (ingestion) | `extract-frames` (overview) | ~50 frames sampled across the whole duration | the **whole file** |

For a 40-min / ~2 GB video, a `search_clip` over a 30s window needs ~1–2% of the bytes,
and an overview needs ~50 keyframes — yet each sandbox downloads the full ~2 GB first.

> **Out of scope / already done:** the **output** frames and clips the sandbox produces
> are already uploaded to S3 and returned as presigned URLs (`--upload-s3`), so the host
> never handles those bytes. This spec is **only** about the *source-video read*.

### Why now

The ingestion redesign moved the overview/description to **ingestion time** (precomputed,
injected into session context, `get_video_description` tool disabled). Consequently the
**session/clip sandbox now does clip-only work** — there is no longer an overview pass to
amortize the full download against. So the full download is pure overhead on the
interactive clip path, and a one-time overhead on the ingestion path.

---

## 2. Goals / Non-goals

**Goals**
- G1. `fetch-clip` reads only the bytes around the requested window (no full download).
- G2. Overview extraction reads only the sampled keyframes (no full download).
- G3. No new infrastructure (no FUSE/s3fs mount, no `SYS_ADMIN`, no extra services).
- G4. Graceful fallback to the current download path on failure or for small inputs.

**Non-goals**
- N1. Changing the output upload/presign path (already optimal).
- N2. Frame-accurate overview timestamps (approximate is acceptable — see §6).
- N3. Replacing ffmpeg with a managed media service (tracked separately; see §8).

---

## 3. Core mechanism: ffmpeg reads a presigned URL

ffmpeg's `http`/`https` protocol issues **HTTP Range requests** for seeking on a seekable
source. S3/R2 support `Range` GETs, so `ffmpeg -ss T -i "<presigned-url>"` transfers only
the bytes near `T` (the mp4 index `moov` + the target keyframe's GOP), not the whole file.

This is the same byte-savings as an s3fs FUSE mount, but with **zero new infra** — it's a
protocol change on the ffmpeg input only.

**Source URL resolution.** The sandbox already has S3 credentials (it presigns *output*
frames/clips). It therefore self-presigns the *source* object and feeds that URL to ffmpeg
— no host change required. Today's `resolve_video_path()` becomes `resolve_video_url()`:

```python
def resolve_video_url(video_id: str) -> str:
    # existing extension probe (head_object) to find <base>/<video_id>.<ext>
    s3 = _make_s3_client()
    key = _probe_source_key(s3, video_id)          # e.g. videos/<video_id>.mp4
    return s3.get_presigned_url(key, expires_in=SOURCE_URL_TTL)
```

**Hard requirement: ffmpeg backend only.** `decord` builds an in-RAM seek index via
random access ([main.py:381](../ambient/sandboxes/e2b/templates/video-analysis-v1/main.py#L381));
over HTTP that is a storm of tiny range GETs and is pathological. When the source is a URL,
**force the ffmpeg backend** and skip decord unconditionally.

**Standard ffmpeg flags for a remote input:**
- `-ss T` **before** `-i` → fast *input* seeking (range-read to nearest keyframe before T).
- `-reconnect 1 -reconnect_streamed 1 -reconnect_delay_max 5` → survive transient drops.
- (`+faststart` at upload time keeps `moov` at the front; see §6.)

---

## 4. Optimization 1 — Presigned-URL clips (`search_clip` / `focus_clip`)

### Flow
`search_clip`/`focus_clip` → `SandboxVideoFrameTools.fetch_clip` → `fetch-clip` →
`_fetch_clip` runs `ffmpeg -ss start -t dur -i <input>`
([main.py:664](../ambient/sandboxes/e2b/templates/video-analysis-v1/main.py#L664)).

### Change
- `fetch-clip` resolves the source as a **presigned URL** (§3) instead of downloading.
- `_fetch_clip` passes that URL as `-i`, with `-ss` before it and the reconnect flags.
- ffmpeg reads `moov` + the window's GOP range → only the clip's bytes cross the network.

### Expected result
A 30s clip from a 2 GB video: transfer ≈ tens of MB instead of ~2 GB. The user-visible
latency drops from *"download whole file, then cut"* to *"range-read the window, then cut."*

### Host impact
Minimal/none: the host already receives an output clip presigned URL back. No change to
`search_clip`/`focus_clip` request/response shape.

---

## 5. Optimization 2 — Seek-sampled overview frames (ingestion)

The overview samples uniformly across the **whole** duration with `-vf fps=N`
([main.py:593](../ambient/sandboxes/e2b/templates/video-analysis-v1/main.py#L593)), which
demuxes the entire file even though it emits ~50 frames. Range reads alone don't help a
full-file pass — so restructure the pass into **N independent single-frame seeks**.

### Flow
`get_video_description` → `get_overview_frames` → `extract-frames` (overview preset:
`OVERVIEW_FPS`, `max_frames=OVERVIEW_MAX_FRAMES`).

### Change
When the source is a URL and the request is a whole-video overview (no explicit window):

```python
duration = ffprobe(url)                          # 1 small range read (moov)
timestamps = linspace(0, duration, N=OVERVIEW_MAX_FRAMES)
def grab(t):                                     # each seek ≈ one GOP over HTTP
    ffmpeg -ss t -reconnect 1 -i url -frames:v 1 -vf scale=... frame_t.png
frames = thread_pool.map(grab, timestamps)       # parallel — see below
```

- **Parallelize the seeks.** They are independent; run them on a thread pool (reuse the
  `UPLOAD_CONCURRENCY` pattern at
  [main.py:711](../ambient/sandboxes/e2b/templates/video-analysis-v1/main.py#L711)). 50
  parallel range reads collapse wall-clock toward a single seek's latency, so this wins on
  **both** bytes transferred *and* wall-clock vs. the full download.
- Timestamps are recorded per frame exactly as today (frames keep their true sampled time;
  see the existing `times.json` handling) — modulo keyframe rounding (§6).

### Expected result
2 GB video sampled at 50 points: transfer ≈ 100–200 MB (≈ 50 GOPs) instead of ~2 GB, with
wall-clock bounded by the slowest of 50 parallel seeks rather than a serial full download.

### Where it runs
Keep this **inside the ephemeral ingestion sandbox** (pass the presigned source URL in,
don't download). Running 50 parallel ffmpeg decodes on the API host would move heavy media
work onto the request process — rejected. The box already has ffmpeg.

---

## 6. Caveats & risks

| # | Risk | Mitigation |
|---|---|---|
| C1 | **decord can't range-read HTTP** | Force ffmpeg backend whenever the source is a URL. |
| C2 | **Input seeking is keyframe-approximate** (`-ss` lands on nearest prior keyframe) | Acceptable for the "highlevel approximate" overview and for clip windows. Document the ≤1-GOP drift. |
| C3 | **`moov` placement** — without `+faststart`, ffmpeg reads the file tail first (extra range GET per process) | Normalize at upload with `-movflags +faststart`, **or** accept the small extra tail read. |
| C4 | **Per-seek overhead** makes N seeks slower than one download for *small* videos | **Size/duration threshold** (e.g. < ~100 MB or < few min) → fall back to download + single pass. |
| C5 | **Presigned URL TTL** must exceed total processing time (overview can take ~tens of s; clip re-encode loops) | Set `SOURCE_URL_TTL` generously (e.g. ≥ clip/overview worst-case, with margin). The same signed URL serves all range requests for that object. |
| C6 | **HTTP flakiness** mid-decode | `-reconnect 1 -reconnect_streamed 1 -reconnect_delay_max 5`; on hard failure, fall back to download (G4). |
| C7 | **N moov re-reads** (one per seek process) | `moov` is small (KB–MB); N small reads are acceptable. Optionally cache/parse once if measured to matter. |

---

## 7. Configuration & rollout

New settings (sandbox env, mirroring `ambient/config.py` naming):

| Setting | Default | Meaning |
|---|---|---|
| `STREAM_SOURCE_VIDEO` | `true` | Master toggle: read source via presigned URL instead of download. |
| `SOURCE_URL_TTL` | `7200` | Presigned source-URL lifetime (s). Must exceed worst-case op time. |
| `STREAM_MIN_BYTES` | `104857600` (100 MB) | Below this, fall back to download (C4). |
| `OVERVIEW_SEEK_CONCURRENCY` | `16` | Parallel seeks for the overview sampler (§5). |

**Rollout**
1. Land behind `STREAM_SOURCE_VIDEO`; ship `false` first to validate plumbing.
2. Enable for the **clip path** first (Optimization 1) — smallest change, biggest live win.
3. Enable for the **overview path** (Optimization 2) once seek-sampling is validated.
4. Keep the download path as the fallback; never remove it (G4, C4, C6).

---

## 8. Validation

Measure, on a small (~50 MB) and a large (~2 GB) video, for both `fetch-clip` and overview:
- **Bytes transferred** into the sandbox (compare to file size).
- **Wall-clock** for the command.
- **Output parity**: frame count, approximate timestamp accuracy, clip correctness vs. the
  download path.

Acceptance: large-video clip and overview each transfer **< 20%** of the file with
wall-clock **≤** the current download path; small-video path falls back and is no slower.

---

## 9. Open questions

- **Q1.** Self-presign in the sandbox (§3) vs. host passes `--source-url`? Self-presign is
  simpler (no host change) and the box already holds S3 creds for output upload; the only
  reason to switch to host-passed URLs is to drop source-read creds from the box, which we
  don't currently need. **Proposed: self-presign.**
- **Q2.** Normalize all uploads to `+faststart` at ingestion (one remux) to make C3/C7
  cheap, or leave source files as-is and absorb the tail read? Remux adds an ingestion-time
  cost but makes every later seek cheaper.
- **Q3.** Threshold units for C4 — bytes (`STREAM_MIN_BYTES`) vs. duration? Bytes correlate
  better with transfer cost; duration is easier to read from ffprobe before sizing. Pick one.

---

## 10. Related

- Ingestion pipeline (precomputed description): `ambient/server/ingest.py`.
- Why **not** s3fs / managed media services (Cloudflare Stream, HLS segmentation): evaluated
  and deferred — FUSE infra risk and decord regression for s3fs; product adoption/cost for
  managed services. The presigned-URL approach captures the clip + overview wins with no new
  infra and is the recommended first step.

---

## 11. As-built — implementation log & decisions (in sequence)

> §3–§5 describe the *original* "stream everything" design. Reality (real yt-dlp
> videos) forced a refinement: **stream only faststart-progressive MP4; for
> everything else, download once and seek-sample the local file.** This section
> is the as-built record. Where it conflicts with §3–§5, this section wins.

### Step 1 — Implemented presigned-URL streaming + fragmentation fallback
- `resolve_video_path()` → `resolve_video_source(video_id) -> (source, is_url)`; the
  sandbox **self-presigns** the source object (it already holds S3 creds). Resolves Q1.
- `fetch-clip`: `ffmpeg -ss start -t dur -i <url>` with `-ss` before `-i` + reconnect flags.
- Overview: `_extract_frames_seek` — N parallel single-frame `-ss` seeks instead of the
  `-vf fps` full-timeline pass.
- Forced the **ffmpeg backend** for URL sources (decord can't range-read HTTP — C1).
- Settings added (§7): `STREAM_SOURCE_VIDEO`, `SOURCE_URL_TTL`, `STREAM_MIN_BYTES`,
  `OVERVIEW_SEEK_CONCURRENCY`, passed to the box via `media_sandbox._media_env`.

### Step 2 — Found: fragmented MP4 (fMP4) hangs over HTTP
- A fragmented MP4 (`moof`/`mfra` boxes, DASH/yt-dlp output) has **no sample table in
  `moov`**; ffmpeg must read the `mfra` index at the end or walk every `moof`, which over
  HTTP degrades to reading ~the whole file. Every op (`ffprobe`, `-ss 3`, `-ss 120`) hung.
- **Decision:** detect fragmentation and fall back to download. Added a cheap head-read
  probe; non-streamable inputs download instead of hanging. (R2 Range support and
  bandwidth were verified fine — fragmentation was the sole cause.)

### Step 3 — Found: moov-at-end kills the *overview* (not clips)
- Benchmarked stream vs download on a 2.6 GB / 54-min video (host **and** in-sandbox):
  | Operation | Download | Stream (faststart) |
  |---|---|---|
  | Overview (50 frames) | ~280s | **~14s** |
  | Clip (30s) | ~21s (download) + cut | ~10s |
- But production stayed slow (**200s** overview). Root cause: the benchmark had normalized
  the test file to **`+faststart`**. Real yt-dlp videos are **`moov`-at-end**. The overview
  opens the file **50×** (one ffmpeg process per seek), and with `moov` at the end **each
  open re-downloads the large tail `moov`** → 50× a big read → 200s. A *single* clip pays it
  once (~11s), so clips were fine; only the 50-seek overview was pathological. (This is C3/C7
  in §6, but far worse than the draft predicted — promoted from "small extra read" to a
  blocking issue.)

### Step 4 — Key insight: the slow part of "download" is the *extract*, not the transfer
- In-sandbox download leg = ~20s transfer + **~260s `-vf fps` full-timeline demux**. The
  transfer is cheap (E2B→R2 ~130 MB/s); the **full-demux extract** is the cost.
- Seek-sampling fixes extraction whether the source is a **URL** (range reads) **or a local
  file** (instant disk seeks). So a downloaded file can be extracted fast too — there was no
  need to stream to get fast extraction.

### Step 5 — Decision: stream only faststart; else download + **local** seek-sample
- Final routing in the sandbox:
  - **faststart progressive MP4** → stream (URL seeks) → overview ~14s, clips ~11s.
  - **moov-at-end / fragmented / non-mp4** → **download once → local seek-sample** →
    overview **~30s** (measured), clips: first downloads (~20s), rest reuse the cached file.
- **No host ffmpeg, no remux, no faststart normalization required** — all inside the
  existing sandbox.

### Step 6 — Implemented (final)
- `_looks_fragmented_mp4` → **`_probe_mp4_layout(s3, key) -> (fragmented, faststart)`** via a
  single 2 MB head-read (`faststart` = `moov` precedes `mdat`).
- `resolve_video_source`: stream **iff** `mp4/mov/m4v` **and not fragmented** **and
  faststart**; otherwise download (logging the reason). Download path is cached in the box.
- Overview routing: seek-sample whenever `max_frames is not None and self._backend ==
  "ffmpeg"` — covers streamed URLs **and** downloaded long files. decord still handles short
  local videos. The slow `-vf fps` overview pass is gone.
- **Verified in a real sandbox:** moov-at-end 2.6 GB overview **200s → 30.4s**.

### Step 7 — Clip decision: uniform download+cache for non-faststart
- `resolve_video_source` can't see whether the caller is an overview (50 opens) or a clip
  (1 open), so non-faststart clips **download once and cache** rather than stream.
- Trade-off (multi-clip session is the norm, since the description is precomputed):
  | Clips in session | Before (stream each) | After (download once + cache) |
  |---|---|---|
  | 1 | 11s | ~25s |
  | 3 | 33s | ~29s |
  | 5 | 55s | ~33s |
- **Decision: keep uniform download+cache** (simpler, wins for ≥2–3 clips). An
  operation-aware knob (clips stream moov-at-end for best first-clip latency, at the cost of
  no cache) is a possible future change, not done.

### Step 8 — Deployment notes
- The sandbox runs the **prebuilt template image** (`COPY . /app`), **not** local `main.py`.
  Changes go live only after **rebuilding the template** (`make e2b-build-prod`).
- ⚠️ Name mismatch observed: `build_prod.py` builds **`video-analysis-v1`**, but the server's
  `settings.e2b_template` resolved to **`video-analysis-v1-dev`**. Rebuild the template the
  server actually boots.

### Step 9 — Faststart normalization is now optional
- Download+local-seek solves the overview without it. Normalization (`-c copy -movflags
  +faststart`) would still shave overview ~30s → ~15s and let fragmented/moov-at-end **clips
  stream** — but it is **not required**.
- If adopted, it must **not** run on the API host (no sandbox there; the e2b setup expects
  ffmpeg dropped from the API image). Do it **in the ingest sandbox** or at the **yt-dlp
  download** (see the command below). Supersedes Q2.

### Net result
| Video layout (source) | Overview | First clip | Later clips (same session) |
|---|---|---|---|
| faststart progressive | ~14s (stream) | ~11s (stream) | ~11s (stream) |
| moov-at-end (typical yt-dlp) | ~30s (download + local seek) | ~20s (download) | near-instant (cached) |
| fragmented | ~30s (download + local seek) | ~20s (download) | near-instant (cached) |

---

Yt-dlp optimized command for non fragemented video download:

```
yt-dlp -f "bv*+ba/b" --merge-output-format mp4 \
  --postprocessor-args "Merger:-movflags +faststart" \
  --exec "ffmpeg -nostdin -y -i %(filepath)q -c copy -movflags +faststart %(filepath)q.fix.mp4 && mv -f %(filepath)q.fix.mp4 %(filepath)q" \
  "URL"
  ```

---

## 12. Implementation spec: sandbox-level YouTube URL ingestion

### Goal

When a user provides a YouTube URL, the API should create a normal `file`/`video_id`
record immediately, enqueue ingestion, and let an ephemeral E2B sandbox download the
video with `yt-dlp`, normalize it for streaming where possible, upload the canonical
source MP4 to R2/S3, and then run the existing background description pipeline.

After source preparation, overview and clip tools must continue to use only
`video_id`. They should not know whether the source began as multipart upload or a
YouTube URL.

### Current repo facts this builds on

- `POST /files` in `ambient/server/routes/managed_agents.py` accepts multipart bytes,
  calls `store_video()`, writes a `files` JSONB row keyed by `video_id`, and enqueues
  background description only when `r2_key` exists.
- `IngestWorker` in `ambient/server/ingest.py` owns durable background work through
  Redis Streams. It claims a `files` row, boots an ephemeral `E2BMediaSandbox`, runs
  `get_video_description`, persists `description`, and retries through row state plus
  re-enqueue.
- `E2BMediaSandbox.run(argv)` shells out to `/app/main.py <subcommand> ...` and parses
  the existing result envelope. New sandbox behavior should be another subcommand, not
  a separate host protocol.
- `main.py` already resolves `video_id` by checking `VIDEO_FOLDER/<video_id>/<video_id>.*`
  before S3/R2. If `yt-dlp` writes the prepared source into that cache path, existing
  `extract-frames` and `fetch-clip` can run unchanged in the same sandbox.
- The template Dockerfile currently installs `ffmpeg`, `Pillow`, `decord`, `pydantic`,
  and `boto3`. It does **not** install `yt-dlp` yet.

### Non-goals

- Do not run `yt-dlp`, remuxing, or faststart normalization on the API host.
- Do not change `runner.py`, `video_description.py`, `video_backend.py`, `search_clip`,
  or `focus_clip` to understand YouTube URLs.
- Do not stream YouTube directly into overview/clip ffmpeg commands. YouTube is a
  source-acquisition concern; media tools continue to operate on `video_id`.
- Do not add cookie/authenticated YouTube support in this pass. If needed later, it
  requires a separate credential-handling design.
- Do not add playlist support. Use single-video URLs only.

### API surface

Add a JSON endpoint next to the multipart upload route:

```http
POST /files/import
Content-Type: application/json

{
  "url": "https://www.youtube.com/watch?v=...",
  "source_type": "youtube",
  "filename": null,
  "metadata": {}
}
```

Recommended response shape is the existing file metadata shape:

```json
{
  "type": "file",
  "id": "vid_...",
  "filename": "vid_....mp4",
  "mime_type": "video/mp4",
  "size_bytes": null,
  "created_at": "...",
  "downloadable": false,
  "source_status": "pending",
  "description_status": "pending"
}
```

Why a separate endpoint instead of overloading multipart `POST /files`: it keeps the
current upload path stable and avoids form-vs-JSON ambiguity. The session attachment
path remains unchanged: clients still mount `{"type": "file", "file_id": "vid_..."}`.

### File record model

`files` stays schema-light JSONB. Uploaded byte files keep their current fields. YouTube
imports add source-acquisition fields:

```json
{
  "id": "vid_...",
  "video_id": "vid_...",
  "filename": "vid_....mp4",
  "mime_type": "video/mp4",
  "size_bytes": null,
  "local_path": null,
  "r2_key": null,

  "source_type": "youtube",
  "source_url": "https://www.youtube.com/watch?v=...",
  "source_status": "pending",
  "source_error": null,
  "source_attempts": 0,
  "source_updated_at": "...",

  "youtube": {
    "webpage_url": null,
    "title": null,
    "extractor": null,
    "duration": null,
    "width": null,
    "height": null,
    "format_id": null
  },

  "description": null,
  "description_status": "pending",
  "description_error": null,
  "description_attempts": 0
}
```

State rules:

- Multipart upload with `r2_key`: `source_type="upload"`, `source_status="ready"`,
  `description_status="pending"`.
- YouTube import: `source_type="youtube"`, `source_status="pending"`,
  `description_status="pending"`.
- The Redis stream still carries only `video_id`; the DB row is the source of truth.
- `description_status` remains the queue driver so the existing sweeper can continue to
  find rows stuck in `pending` or `processing`.
- `source_status` is a sub-stage inside the same ingest job, not a second queue.

### Host ingest flow

Extend `IngestWorker._run_job(video_id)` into two phases:

1. Claim the row exactly as today (`description_status -> processing`) under
   `Store.update_file()`.
2. Boot the ephemeral `E2BMediaSandbox`.
3. If `rec.source_type == "youtube"` and `rec.source_status != "ready"`:
   - mark `source_status="processing"`;
   - call `box.run(["prepare-youtube", "--video-id", video_id, "--url", source_url,
     "--upload-s3"])`;
   - persist returned `r2_key`, `size_bytes`, `mime_type`, `filename`, `youtube` metadata,
     and `source_status="ready"`.
4. Run the existing `_generate_description(video_id, box)`.
5. Persist `description_status="ready"` and `description`.
6. Kill the sandbox in `finally`, as today.

Failure handling:

- If `prepare-youtube` fails, increment `source_attempts`, set `source_error`, and set
  `source_status` to `pending` or terminal `failed` using the same max-attempt cap.
- Since source preparation is required before description, also set `description_error`
  to the source error and leave `description_status` as `pending` for retry, or `failed`
  when attempts are exhausted.
- A retry should re-run `prepare-youtube`. If the previous attempt already uploaded
  `r2_key`, the sandbox command may detect and reuse it, but correctness must not depend
  on partial output.

Read-side behavior:

- `GET /files/{id}` should include `source_status`, `source_error`, and the `youtube`
  metadata in addition to the existing description fields.
- Session startup continues to call `ensure_description(store, video_id, box=...)`.
  If a YouTube import is still pending, it waits through the same file-row state. No
  session or tool code should accept raw YouTube URLs.

### Sandbox command

Add a new `/app/main.py` subcommand:

```bash
python /app/main.py prepare-youtube \
  --video-id vid_... \
  --url "https://www.youtube.com/watch?v=..." \
  --upload-s3
```

It must emit the same fenced result envelope as `extract-frames` and `fetch-clip`.
Suggested data model:

```json
{
  "video_id": "vid_...",
  "source_file_path": "/tmp/videos/vid_.../vid_....mp4",
  "r2_key": "videos/vid_....mp4",
  "s3_uri": "s3://bucket/videos/vid_....mp4",
  "size_bytes": 123456789,
  "duration": 123.45,
  "width": 1920,
  "height": 1080,
  "source_kind": "youtube",
  "download_info": {
    "extractor": "youtube",
    "webpage_url": "https://www.youtube.com/watch?v=...",
    "title": "...",
    "format_id": "..."
  }
}
```

The prepared file must land at:

```text
VIDEO_FOLDER/<video_id>/<video_id>.mp4
```

That path is intentionally the same local cache path checked by `resolve_video_source()`.
In the same ingest sandbox, `get_video_description(video_id)` will then find the local
file and skip any S3 source download. In later session sandboxes, `resolve_video_source()`
will find the uploaded `S3_VIDEO_BASE_KEY/<video_id>.mp4`.

### yt-dlp command policy

Install `yt-dlp` into the E2B template image, preferably in the existing Python package
layer:

```dockerfile
RUN pip install --no-cache-dir \
    Pillow \
    decord \
    pydantic \
    boto3 \
    yt-dlp
```

Use an argv list, not shell string interpolation. Baseline command:

```bash
yt-dlp \
  --no-playlist \
  --restrict-filenames \
  -f "bv*[ext=mp4]+ba[ext=m4a]/b[ext=mp4]/bv*+ba/b" \
  --merge-output-format mp4 \
  --postprocessor-args "Merger:-movflags +faststart" \
  --paths "VIDEO_FOLDER/<video_id>" \
  --output "<video_id>.%(ext)s" \
  --print-json \
  "URL"
```

Notes:

- Prefer MP4 video + M4A audio so the merged result can be stream-copied into MP4.
- `--merge-output-format mp4` keeps the canonical output extension stable.
- `--postprocessor-args "Merger:-movflags +faststart"` asks the yt-dlp merger to put
  `moov` at the front.
- `--print-json` gives metadata for the host row, but stdout must not become the control
  plane; parse it inside `main.py` and emit only the standard Ambient result envelope.
- Log yt-dlp progress to stderr only.

After yt-dlp completes, locate the actual output file. If it is not exactly
`<video_id>.mp4`, remux or rename to the canonical path:

```text
VIDEO_FOLDER/<video_id>/<video_id>.mp4
```

Then verify layout using the same concepts as `_probe_mp4_layout`, but against local
bytes. If the file is not faststart progressive, attempt an explicit stream-copy
normalization:

```bash
ffmpeg -nostdin -y -i input.mp4 -c copy -movflags +faststart input.fix.mp4
mv -f input.fix.mp4 input.mp4
```

If stream-copy normalization fails, keep the downloaded file and continue. The existing
as-built fallback still handles non-faststart/fragmented sources by downloading once and
local seek-sampling. The optimization goal is "faststart when cheap", not "fail if not
faststart".

### Uploading canonical source to R2/S3

When `--upload-s3` is passed, upload:

```text
local: VIDEO_FOLDER/<video_id>/<video_id>.mp4
key:   S3_VIDEO_BASE_KEY/<video_id>.mp4
type:  video/mp4
```

This key shape matches the current extension probing and makes later sandboxes source the
same video through the existing resolver. Upload should happen after normalization so R2
stores the best canonical source available.

The command should return `r2_key` and `size_bytes`. The host should persist those fields
on the `files` row before running description.

### Guardrails and settings

Add settings in `ambient/config.py` and pass relevant sandbox values through
`_media_env()`:

```python
youtube_import_enabled: bool = True
youtube_max_duration_seconds: int = 3 * 60 * 60
youtube_max_size_bytes: int = 5 * 1024 * 1024 * 1024
youtube_download_timeout_seconds: int = 900
```

Sandbox behavior:

- Reject URLs unless `source_type == "youtube"` and the URL host is an allowed YouTube
  host (`youtube.com`, `www.youtube.com`, `m.youtube.com`, `youtu.be`).
- Always pass `--no-playlist`.
- Use `yt-dlp --dump-single-json` or metadata from `--print-json` before/after download
  to enforce duration where available.
- Enforce final file size after download and before upload.
- Use the existing command timeout style from `media_sandbox.py` and keep all failures in
  the result envelope.
- Do not pass cookies, browser profiles, or user credentials.

### End-to-end sequence

```text
POST /files/import {url}
  -> create files row:
       source_type=youtube, source_status=pending,
       description_status=pending
  -> XADD ingest stream {video_id}
  -> return video_id immediately

IngestWorker(video_id)
  -> claim description job
  -> boot ephemeral E2B sandbox
  -> prepare-youtube:
       yt-dlp download
       best-effort faststart normalization
       upload canonical MP4 to R2/S3
       return source metadata
  -> persist source_status=ready + r2_key + metadata
  -> existing get_video_description(video_id)
       resolves local prepared file in the same sandbox
       samples overview frames
       calls host LLM
  -> persist description_status=ready + description
  -> kill sandbox

Later session sandbox
  -> receives only video_id
  -> resolve_video_source(video_id)
       finds S3_VIDEO_BASE_KEY/<video_id>.mp4
       streams if faststart; otherwise downloads once + local seek
  -> clips/search/focus proceed unchanged
```

### Testing plan

Unit/local tests:

- URL validation accepts YouTube hosts and rejects non-YouTube hosts.
- File row creation for `/files/import` sets source and description states correctly.
- `IngestWorker` calls source preparation before `_generate_description` for
  `source_type="youtube"` rows.
- Failure in source preparation updates `source_status`, `source_attempts`,
  `source_error`, and retry/terminal `description_status` correctly.

Sandbox smoke tests:

- `python /app/main.py prepare-youtube --video-id test --url <short-youtube-url>
  --upload-s3` returns a valid envelope and uploads `videos/test.mp4`.
- Follow immediately in the same sandbox with `extract-frames --video-id test
  --max-frames 50 --upload-s3`; logs should show `found locally`.
- Start a fresh sandbox and run `extract-frames --video-id test`; logs should show S3
  resolution and then either faststart streaming or download+local-seek.

Integration test:

- `POST /files/import` returns quickly with `source_status=pending`.
- Poll `GET /files/{id}` until `source_status=ready` and `description_status=ready`.
- Create a session with the file resource and verify the first user turn receives the
  precomputed description without running YouTube download again.

### Deployment notes

- Rebuild the E2B template after adding `yt-dlp` and the new subcommand. Local edits to
  `main.py` are not used by already-built templates.
- Build the template name the server actually boots. In this repo,
  `build_prod.py` builds `video-analysis-v1`; `build_dev.py` builds
  `video-analysis-v1-dev`; `settings.e2b_template` decides which one is live.
- Roll out API and template together. The host can create YouTube rows only after the
  target template supports `prepare-youtube`.

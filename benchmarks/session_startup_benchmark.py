#!/usr/bin/env python3
"""Session-startup latency benchmark.

Measures the wall-clock cost of everything between "user submits a video" and
"agent produces its first output", against a *running* managed-agents server
(the same HTTP surface the demo / Anthropic SDK drive). It exists to quantify the
improvements in docs/session-startup-latency-plan.md: run it once to capture a
baseline, apply the changes, run it again, and diff.

It mirrors the demo flow exactly (import -> create agent/env/session -> wait for
the session to go idle -> send the first message -> stream the run) but stamps a
timestamp on every phase boundary and on each streamed event.

Phases captured (all seconds from t0 = just before the import/upload call):

  import_request        POST /files/import returned (video id in hand)
  source_ready          file.source_status == "ready"     (youtube download+upload done)
  description_ready     file.description_status == "ready" (overview + LLM description done)
  session_ready         session.status == "idle"          (sandbox booted AND — today —
                                                            description resolved; this is the
                                                            number Phase 1 of the plan attacks)
  first_run_event       first session.status_running after the first user message
                        (relative to send: "time to first processed input")
  first_agent_activity  first tool_use or agent.message   (relative to send)
  final_answer          session.status_idle for the run   (relative to send)
  total                 t0 -> final answer

Usage:
  # Cold end-to-end run against a fresh YouTube import:
  uv run python benchmarks/session_startup_benchmark.py run \
      --url "https://www.youtube.com/watch?v=4qYqPmIO0v0" \
      --label baseline --out benchmarks/results/baseline.json

  # Session-start only, reusing an already-ingested video (isolates sandbox/boot
  # from the ingest pipeline):
  uv run python benchmarks/session_startup_benchmark.py run \
      --video-id vid_abc123 --label baseline-warm --out benchmarks/results/warm.json

  # Compare two result files:
  uv run python benchmarks/session_startup_benchmark.py compare \
      benchmarks/results/baseline.json benchmarks/results/after.json

Env fallbacks (match demo/app.py): AMBIENT_BASE_URL, AMBIENT_API_KEY,
AMBIENT_AGENT_MODEL.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
from dataclasses import asdict, dataclass, field
from typing import Optional

import httpx

DEFAULT_BASE_URL = os.getenv("AMBIENT_BASE_URL", "http://127.0.0.1:8080")
DEFAULT_API_KEY = os.getenv("AMBIENT_API_KEY", "dev-token")
DEFAULT_MODEL = os.getenv("AMBIENT_AGENT_MODEL", "z-ai/glm-5.2")
DEFAULT_URL = "https://www.youtube.com/watch?v=4qYqPmIO0v0"
DEFAULT_QUESTION = "Give a one-sentence summary of what this video is about."


# ---------------------------------------------------------------------------
# Result model
# ---------------------------------------------------------------------------

@dataclass
class BenchmarkResult:
    label: str
    base_url: str
    model: str
    video_id: Optional[str] = None
    youtube_url: Optional[str] = None
    question: Optional[str] = None
    imported_fresh: bool = False
    started_at: str = ""

    # Phase timings in seconds from t0 (None = not reached / skipped).
    import_request_s: Optional[float] = None
    source_ready_s: Optional[float] = None
    description_ready_s: Optional[float] = None
    session_create_s: Optional[float] = None
    session_ready_s: Optional[float] = None

    # First-message timings, relative to the moment the message was sent.
    first_run_event_s: Optional[float] = None
    first_agent_activity_s: Optional[float] = None
    final_answer_s: Optional[float] = None
    total_s: Optional[float] = None

    # Extras for debugging / context.
    sandbox_boot_ms: Optional[int] = None
    answer_preview: Optional[str] = None
    event_log: list[dict] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# HTTP helpers
# ---------------------------------------------------------------------------

class ServerClient:
    def __init__(self, base_url: str, api_key: str) -> None:
        self.base = base_url.rstrip("/")
        self.headers = {"x-api-key": api_key}
        self._http = httpx.Client(timeout=60.0)

    def close(self) -> None:
        self._http.close()

    def _url(self, path: str) -> str:
        return f"{self.base}{path}"

    def import_youtube(self, url: str) -> dict:
        r = self._http.post(
            self._url("/v1/files/import"),
            headers=self.headers,
            json={"source_type": "youtube", "url": url},
        )
        r.raise_for_status()
        return r.json()

    def get_file(self, file_id: str) -> dict:
        r = self._http.get(self._url(f"/v1/files/{file_id}"), headers=self.headers)
        r.raise_for_status()
        return r.json()

    def create_agent(self, model: str) -> dict:
        r = self._http.post(
            self._url("/v1/agents"),
            headers=self.headers,
            json={
                "name": "Benchmark Agent",
                "model": model,
                "system": "Analyze videos and answer questions about them with citations.",
                "tools": [{"type": "agent_toolset_20260401"}],
            },
        )
        r.raise_for_status()
        return r.json()

    def create_environment(self) -> dict:
        r = self._http.post(
            self._url("/v1/environments"), headers=self.headers, json={"name": "benchmark"}
        )
        r.raise_for_status()
        return r.json()

    def create_session(self, agent_id: str, env_id: str, video_id: str, model: str) -> dict:
        r = self._http.post(
            self._url("/v1/sessions"),
            headers=self.headers,
            json={
                "agent": agent_id,
                "environment_id": env_id,
                "metadata": {"video_id": video_id},
                "model": model,
                "title": "benchmark",
            },
        )
        r.raise_for_status()
        return r.json()

    def get_session(self, session_id: str) -> dict:
        r = self._http.get(self._url(f"/v1/sessions/{session_id}"), headers=self.headers)
        r.raise_for_status()
        return r.json()

    def send_message(self, session_id: str, text: str) -> dict:
        r = self._http.post(
            self._url(f"/v1/sessions/{session_id}/events"),
            headers=self.headers,
            json={"events": [{"type": "user.message", "content": text}]},
        )
        r.raise_for_status()
        return r.json()

    def stream_events(self, session_id: str):
        """Yield (event_name, data_dict) tuples from the SSE run stream."""
        with self._http.stream(
            "GET",
            self._url(f"/v1/sessions/{session_id}/events/stream"),
            headers=self.headers,
            timeout=httpx.Timeout(300.0, read=300.0),
        ) as resp:
            resp.raise_for_status()
            event_name = None
            for line in resp.iter_lines():
                if line is None:
                    continue
                line = line.strip()
                if not line:
                    event_name = None
                    continue
                if line.startswith("event:"):
                    event_name = line[len("event:"):].strip()
                elif line.startswith("data:"):
                    raw = line[len("data:"):].strip()
                    try:
                        data = json.loads(raw)
                    except json.JSONDecodeError:
                        data = {"raw": raw}
                    yield event_name or data.get("type"), data


# ---------------------------------------------------------------------------
# Benchmark driver
# ---------------------------------------------------------------------------

class FileStatusPoller:
    """Background thread that stamps source/description readiness from t0.

    After Phase 1 the session goes ready long before ingestion finishes, so the
    main readiness loop no longer observes the file sub-phases. This polls the
    file row independently so source_ready_s / description_ready_s stay
    measurable (they're what Phases 2 & 3 improve).
    """

    def __init__(self, base_url: str, api_key: str, video_id: str,
                 result: BenchmarkResult, t0: float, timeout_s: float, poll_s: float) -> None:
        self._client = ServerClient(base_url, api_key)
        self._video_id = video_id
        self._result = result
        self._t0 = t0
        self._deadline = t0 + timeout_s
        self._poll_s = poll_s
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=5)
        self._client.close()

    def _run(self) -> None:
        while not self._stop.is_set() and time.monotonic() < self._deadline:
            try:
                f = self._client.get_file(self._video_id)
                if f.get("source_status") == "ready" and self._result.source_ready_s is None:
                    self._result.source_ready_s = round(time.monotonic() - self._t0, 2)
                if f.get("description_status") == "ready" and self._result.description_ready_s is None:
                    self._result.description_ready_s = round(time.monotonic() - self._t0, 2)
                if (self._result.source_ready_s is not None
                        and self._result.description_ready_s is not None):
                    return
            except Exception:  # noqa: BLE001 - best-effort observability
                pass
            self._stop.wait(self._poll_s)


def _poll_readiness(
    client: ServerClient,
    result: BenchmarkResult,
    session_id: str,
    video_id: Optional[str],
    t0: float,
    timeout_s: float,
    poll_s: float,
) -> bool:
    """Poll session + file until the session is idle (ready). Records phase times.

    Returns True once the session is ready, False on timeout/termination.
    """
    deadline = t0 + timeout_s
    while time.monotonic() < deadline:
        # File ingest sub-phases first, so the tick where the session flips to
        # idle still records source/description readiness before we early-return
        # (they often land on the same poll iteration).
        if video_id:
            try:
                f = client.get_file(video_id)
                if f.get("source_status") == "ready" and result.source_ready_s is None:
                    result.source_ready_s = round(time.monotonic() - t0, 2)
                if f.get("description_status") == "ready" and result.description_ready_s is None:
                    result.description_ready_s = round(time.monotonic() - t0, 2)
            except httpx.HTTPStatusError:
                pass

        # Session status (the user-perceived "ready" signal).
        sess = client.get_session(session_id)
        status = sess.get("status")
        if result.sandbox_boot_ms is None:
            boot = (sess.get("sandbox") or {}).get("boot_ms")
            if boot is not None:
                result.sandbox_boot_ms = boot
        if status == "idle" and result.session_ready_s is None:
            result.session_ready_s = round(time.monotonic() - t0, 2)
            return True
        if status == "terminated":
            result.errors.append("session terminated before becoming ready")
            return False

        time.sleep(poll_s)

    result.errors.append(f"session did not become ready within {timeout_s}s")
    return False


def _drive_first_message(
    client: ServerClient,
    result: BenchmarkResult,
    session_id: str,
    question: str,
) -> None:
    """Send the first user message and time the streamed run to completion."""
    send_t = time.monotonic()
    client.send_message(session_id, question)
    try:
        for name, data in client.stream_events(session_id):
            now = round(time.monotonic() - send_t, 2)
            result.event_log.append({"t": now, "event": name, "type": data.get("type")})
            etype = name or data.get("type")

            if etype == "session.status_running" and result.first_run_event_s is None:
                result.first_run_event_s = now
            elif etype in ("agent.tool_use", "agent.tool_result", "agent.message"):
                if result.first_agent_activity_s is None:
                    result.first_agent_activity_s = now
                if etype == "agent.message":
                    blocks = data.get("content") or []
                    text = "".join(b.get("text", "") for b in blocks if isinstance(b, dict))
                    if text:
                        result.answer_preview = text[:200]
            elif etype == "session.status_idle":
                result.final_answer_s = now
                break
    except Exception as exc:  # noqa: BLE001 - record and move on
        result.errors.append(f"stream error: {type(exc).__name__}: {exc}")


def run_benchmark(args: argparse.Namespace) -> BenchmarkResult:
    client = ServerClient(args.base_url, args.api_key)
    file_poller: Optional[FileStatusPoller] = None
    result = BenchmarkResult(
        label=args.label,
        base_url=args.base_url,
        model=args.model,
        youtube_url=None if args.video_id else args.url,
        question=None if args.no_question else args.question,
        imported_fresh=not args.video_id,
        started_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    )
    try:
        t0 = time.monotonic()

        # 1. Get a video id — either import a fresh YouTube URL or reuse one.
        if args.video_id:
            video_id = args.video_id
            result.video_id = video_id
            print(f"[bench] reusing video_id={video_id} (no import)", file=sys.stderr)
        else:
            print(f"[bench] importing {args.url}", file=sys.stderr)
            meta = client.import_youtube(args.url)
            video_id = meta["id"]
            result.video_id = video_id
            result.import_request_s = round(time.monotonic() - t0, 2)
            print(f"[bench] import returned video_id={video_id} "
                  f"in {result.import_request_s}s", file=sys.stderr)

        # Track ingest sub-phases independently of session readiness.
        file_poller = FileStatusPoller(
            args.base_url, args.api_key, video_id, result, t0,
            timeout_s=args.ready_timeout, poll_s=args.poll_interval,
        )
        file_poller.start()

        # 2. Create agent + environment + session immediately (as the demo does).
        agent = client.create_agent(args.model)
        env = client.create_environment()
        sess = client.create_session(agent["id"], env["id"], video_id, args.model)
        session_id = sess["id"]
        result.session_create_s = round(time.monotonic() - t0, 2)
        print(f"[bench] session {session_id} created in {result.session_create_s}s; "
              f"waiting for ready…", file=sys.stderr)

        # 3. Wait for the session to become ready (this is the key startup metric).
        ready = _poll_readiness(
            client, result, session_id, video_id, t0,
            timeout_s=args.ready_timeout, poll_s=args.poll_interval,
        )
        if result.session_ready_s is not None:
            print(f"[bench] session ready in {result.session_ready_s}s "
                  f"(sandbox boot {result.sandbox_boot_ms}ms)", file=sys.stderr)

        # 4. Send the first message and time the run.
        if ready and not args.no_question:
            print("[bench] sending first message…", file=sys.stderr)
            _drive_first_message(client, result, session_id, args.question)
            result.total_s = round(time.monotonic() - t0, 2)
            print(f"[bench] first run event {result.first_run_event_s}s after send; "
                  f"final answer {result.final_answer_s}s after send", file=sys.stderr)
    finally:
        if file_poller is not None:
            file_poller.stop()
        client.close()
    return result


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

_PHASES = [
    ("import_request_s", "Import request returned"),
    ("session_create_s", "Session created"),
    ("source_ready_s", "Source ready (download+upload)"),
    ("description_ready_s", "Description ready"),
    ("session_ready_s", "Session READY (time-to-input)"),
    ("first_run_event_s", "  ↳ first run event (after send)"),
    ("first_agent_activity_s", "  ↳ first agent activity (after send)"),
    ("final_answer_s", "  ↳ final answer (after send)"),
    ("total_s", "TOTAL (t0 → answer)"),
]


def _fmt(v: Optional[float]) -> str:
    return "—" if v is None else f"{v:>7.2f}s"


def print_report(result: BenchmarkResult) -> None:
    print()
    print("=" * 60)
    print(f"  Session-startup benchmark: {result.label}")
    print("=" * 60)
    print(f"  server : {result.base_url}")
    print(f"  model  : {result.model}")
    print(f"  video  : {result.video_id} "
          f"({'fresh import' if result.imported_fresh else 'reused'})")
    if result.sandbox_boot_ms is not None:
        print(f"  boot   : {result.sandbox_boot_ms} ms")
    print("-" * 60)
    for key, label in _PHASES:
        print(f"  {label:<42}{_fmt(getattr(result, key))}")
    print("-" * 60)
    if result.answer_preview:
        print(f"  answer : {result.answer_preview!r}")
    if result.errors:
        print("  errors :")
        for e in result.errors:
            print(f"    - {e}")
    print("=" * 60)


def print_comparison(a: dict, b: dict) -> None:
    print()
    print("=" * 74)
    print(f"  Comparison: {a.get('label', 'A')}  →  {b.get('label', 'B')}")
    print("=" * 74)
    print(f"  {'phase':<40}{'before':>10}{'after':>10}   {'Δ':<16}")
    print("-" * 74)
    for key, label in _PHASES:
        va, vb = a.get(key), b.get(key)
        if va is None and vb is None:
            continue
        delta = "—"
        if va is not None and vb is not None:
            d = vb - va
            pct = (d / va * 100) if va else 0.0
            delta = f"{d:+.2f}s ({pct:+.0f}%)"
        print(f"  {label:<40}{_fmt(va):>10}{_fmt(vb):>10}   {delta:<16}")
    print("=" * 74)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _add_run_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--base-url", default=DEFAULT_BASE_URL, help="server base URL")
    p.add_argument("--api-key", default=DEFAULT_API_KEY, help="x-api-key value")
    p.add_argument("--model", default=DEFAULT_MODEL, help="agent model id")
    p.add_argument("--url", default=DEFAULT_URL, help="YouTube URL to import")
    p.add_argument("--video-id", default=None,
                   help="reuse an existing (already-ingested) video id; skips import")
    p.add_argument("--question", default=DEFAULT_QUESTION, help="first user message")
    p.add_argument("--no-question", action="store_true",
                   help="stop after session-ready; don't send a message")
    p.add_argument("--label", default="run", help="label for this result")
    p.add_argument("--out", default=None, help="write the result JSON to this path")
    p.add_argument("--ready-timeout", type=float, default=900.0,
                   help="max seconds to wait for session ready (default 900)")
    p.add_argument("--poll-interval", type=float, default=1.0,
                   help="readiness poll cadence in seconds (default 1.0)")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command")

    p_run = sub.add_parser("run", help="run the benchmark against a live server")
    _add_run_args(p_run)

    p_cmp = sub.add_parser("compare", help="diff two result JSON files")
    p_cmp.add_argument("before")
    p_cmp.add_argument("after")

    # Default to `run` when no subcommand is given.
    argv = sys.argv[1:]
    if not argv or argv[0] not in {"run", "compare", "-h", "--help"}:
        argv = ["run", *argv]
    args = parser.parse_args(argv)

    if args.command == "compare":
        with open(args.before) as fh:
            a = json.load(fh)
        with open(args.after) as fh:
            b = json.load(fh)
        print_comparison(a, b)
        return 0

    result = run_benchmark(args)
    print_report(result)
    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w") as fh:
            json.dump(asdict(result), fh, indent=2)
        print(f"\n[bench] wrote {args.out}", file=sys.stderr)
    return 0 if not result.errors else 1


if __name__ == "__main__":
    raise SystemExit(main())

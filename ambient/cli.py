"""`ambient` — a zero-infra CLI for the video agent.

Ask a question about a local video (or a URL) and get a cited answer, running the
*same* agent loop the server uses (`server.runner.SessionRunner`) driven by the
in-memory adapters in `ambient.lite` — no Postgres, Redis, S3, or E2B. Media is
produced by host ffmpeg via the range-proxy backend, which reads clip windows,
frames, and overview samples straight off the local file (no sandbox, no S3).

    ambient analyze trip.mp4 "when do they reach the summit?"
    ambient analyze https://youtu.be/… "summarize the itinerary" --model …
    ambient doctor
    ambient config set OPENROUTER_API_KEY sk-or-…

IMPORTANT (import order): settings are a frozen singleton captured at first
import of `ambient.config`, and some modules capture values (e.g. VIDEO_FOLDER,
the tool set) at import time. So this module imports only stdlib + typer + rich
at top level; every `ambient.*` import happens inside a command, *after* the
environment has been populated from config files and CLI flags.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import stat
import subprocess
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console, Group
from rich.live import Live
from rich.markdown import Markdown
from rich.panel import Panel
from rich.spinner import Spinner
from rich.text import Text

app = typer.Typer(
    add_completion=False,
    help="Ask questions about long videos and get cited answers — no infra required.",
    no_args_is_help=True,
)
config_app = typer.Typer(help="View or set persisted config (~/.config/ambient/env).")
app.add_typer(config_app, name="config")

console = Console()
err = Console(stderr=True)

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
CONFIG_DIR = Path(os.path.expanduser("~")) / ".config" / "ambient"
CONFIG_FILE = CONFIG_DIR / "env"
WORKSPACE = Path(os.path.expanduser("~")) / ".cache" / "ambient" / "videos"
# Extensions the range-proxy backend recognizes as a local source; keep the real
# extension when we stage a file so `_local_source()` finds it.
VIDEO_EXTS = (".mp4", ".mov", ".mkv", ".webm", ".m4v", ".avi", ".mpeg", ".mpg", ".flv", ".wmv")
_URL_RE = re.compile(r"^https?://", re.IGNORECASE)


# --------------------------------------------------------------------------
# Config + environment (must run before any ambient import)
# --------------------------------------------------------------------------

def _load_config_files() -> None:
    """Populate os.environ from ~/.config/ambient/env then ./.env.

    Real environment variables always win (override=False). The persisted config
    file is loaded before the project-local .env so an explicit project setting
    takes precedence over the user default.
    """
    try:
        from dotenv import load_dotenv
    except Exception:  # noqa: BLE001 - dotenv is a transitive dep; degrade gracefully
        _load_env_file_manual(CONFIG_FILE)
        _load_env_file_manual(Path(".env"))
        return
    if CONFIG_FILE.exists():
        load_dotenv(CONFIG_FILE, override=False)
    load_dotenv(Path(".env"), override=False)


def _load_env_file_manual(path: Path) -> None:
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        os.environ.setdefault(key.strip(), val.strip().strip('"').strip("'"))


def _apply_env(model: Optional[str], vision_model: Optional[str], workspace: Path) -> None:
    """Translate CLI flags + OpenRouter convenience into the settings env vars.

    Runs before importing ambient so the frozen `settings` singleton reflects it.
    """
    # OpenRouter convenience: a single OPENROUTER_API_KEY populates both the agent
    # and vision credentials and points the base URL at OpenRouter (unless the user
    # has already set an explicit LLM_BASE_URL).
    or_key = os.environ.get("OPENROUTER_API_KEY")
    if or_key:
        os.environ.setdefault("LLM_API_KEY", or_key)
        os.environ.setdefault("AGENT_API_KEY", or_key)
        if not os.environ.get("LLM_BASE_URL"):
            os.environ["LLM_BASE_URL"] = OPENROUTER_BASE_URL
        os.environ.setdefault("AGENT_BASE_URL", os.environ["LLM_BASE_URL"])

    if model:
        # One --model drives both roles: AGENT_MODEL runs the tool loop, LLM_MODEL
        # runs the vision/tool calls. --vision-model overrides just the latter.
        os.environ["AGENT_MODEL"] = model
        os.environ["LLM_MODEL"] = model
    if vision_model:
        os.environ["LLM_MODEL"] = vision_model

    # CLI mode is always local, whatever a project .env might say:
    #  - host backend (range proxy over the local file — clips/frames/overview via
    #    host ffmpeg), never e2b. Any value != "e2b" selects host (make_video_tools).
    #  - no S3 bucket. This is what lets the workflow match the server without S3:
    #    the whole-video-context path (self_video_analysis) tries source_video_url,
    #    gets None for an empty bucket, and falls back to locally-sampled base64
    #    frames instead of a presigned URL for an object we never uploaded. Clips
    #    from the range proxy are already inline base64, so no upload is needed.
    # We intentionally do NOT override self-video-analysis: the CLI honors the same
    # derivation the server uses (on by default for a vision-capable agent model,
    # unless DISABLE_SELF_VIDEO_ANALYSIS_TOOL is set — see self_video_analysis_enabled),
    # so the agent workflow (tool loop vs whole-video self-analysis) is identical
    # across UI and CLI.
    os.environ["SANDBOX_BACKEND"] = "host"
    os.environ["S3_BUCKET"] = ""
    os.environ["VIDEO_FOLDER"] = str(workspace)


# --------------------------------------------------------------------------
# Preflight
# --------------------------------------------------------------------------

def _which(binary: str) -> bool:
    return shutil.which(binary) is not None


def _resolve_api_key() -> Optional[str]:
    return (
        os.environ.get("OPENROUTER_API_KEY")
        or os.environ.get("LLM_API_KEY")
        or os.environ.get("AGENT_API_KEY")
    )


def _preflight(require_key: bool = True) -> list[str]:
    """Return a list of human-readable problems (empty = good to go)."""
    problems: list[str] = []
    if not _which("ffmpeg"):
        problems.append("ffmpeg not found on PATH — install it (macOS: `brew install ffmpeg`, Debian/Ubuntu: `apt install ffmpeg`).")
    if not _which("ffprobe"):
        problems.append("ffprobe not found on PATH — it ships with ffmpeg; install ffmpeg.")
    if require_key and not _resolve_api_key():
        problems.append(
            "No API key found. Set OPENROUTER_API_KEY (get one at https://openrouter.ai/keys), "
            "then `ambient config set OPENROUTER_API_KEY sk-or-…` or export it."
        )
    return problems


# --------------------------------------------------------------------------
# Video resolution -> a local file at WORKSPACE/{video_id}{ext}
# --------------------------------------------------------------------------

def _sanitize_id(stem: str) -> str:
    """A filesystem/glob-safe video id. The backends resolve the source by
    `{video_id}{ext}` (range-proxy) and `glob({video_id}*)` (inprocess), so the id
    must contain no spaces or glob metacharacters."""
    cleaned = re.sub(r"[^A-Za-z0-9_-]+", "_", stem).strip("_")
    return cleaned or "video"


def _is_url(arg: str) -> bool:
    return bool(_URL_RE.match(arg))


def _stage_local(path: Path) -> str:
    """Symlink (or copy) a local video into the workspace as {video_id}{ext} and
    return the video_id. A content hash disambiguates collisions between different
    files that share a stem."""
    if not path.exists():
        raise typer.BadParameter(f"video file not found: {path}")
    ext = path.suffix.lower()
    if ext not in VIDEO_EXTS:
        raise typer.BadParameter(
            f"unsupported video extension {ext!r}; expected one of {', '.join(VIDEO_EXTS)}"
        )
    digest = hashlib.sha1(str(path.resolve()).encode()).hexdigest()[:8]
    video_id = f"{_sanitize_id(path.stem)}_{digest}"
    WORKSPACE.mkdir(parents=True, exist_ok=True)
    dest = WORKSPACE / f"{video_id}{ext}"
    if not dest.exists():
        try:
            dest.symlink_to(path.resolve())
        except OSError:
            shutil.copy2(path, dest)
    return video_id


def _download_url(url: str) -> str:
    """Download a URL (YouTube or direct) into the workspace via yt-dlp and return
    the video_id. Capped at 720p — nothing downstream uses more."""
    video_id = f"url_{hashlib.sha1(url.encode()).hexdigest()[:12]}"
    WORKSPACE.mkdir(parents=True, exist_ok=True)
    existing = [p for p in WORKSPACE.glob(f"{video_id}.*") if p.suffix.lower() in VIDEO_EXTS]
    if existing:
        return video_id
    if not _which("yt-dlp"):
        raise typer.BadParameter(
            "yt-dlp is required to fetch a URL. Install it (`pip install yt-dlp` or `brew install yt-dlp`), "
            "or pass a local video file instead."
        )
    out_tmpl = str(WORKSPACE / f"{video_id}.%(ext)s")
    err.print(f"[dim]Downloading {url} …[/dim]")
    proc = subprocess.run(
        ["yt-dlp", "-f", "bv*[height<=720]+ba/b[height<=720]/b",
         "--merge-output-format", "mp4", "-o", out_tmpl, url],
        capture_output=True, text=True,
    )
    if proc.returncode != 0:
        raise typer.BadParameter(f"yt-dlp failed:\n{proc.stderr[-500:]}")
    got = [p for p in WORKSPACE.glob(f"{video_id}.*") if p.suffix.lower() in VIDEO_EXTS]
    if not got:
        raise typer.BadParameter("download produced no recognizable video file")
    return video_id


def _resolve_video(arg: str) -> str:
    return _download_url(arg) if _is_url(arg) else _stage_local(Path(arg))


def _probe_duration(video_id: str) -> Optional[float]:
    src = None
    for ext in VIDEO_EXTS:
        p = WORKSPACE / f"{video_id}{ext}"
        if p.exists():
            src = str(p)
            break
    if not src:
        return None
    try:
        r = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "default=nw=1:nk=1", src],
            capture_output=True, text=True, timeout=60,
        )
        if r.returncode == 0 and r.stdout.strip():
            return float(r.stdout.strip())
    except Exception:  # noqa: BLE001
        pass
    return None


# --------------------------------------------------------------------------
# Streaming renderer — consumes runner events off the broker and renders a live,
# token-by-token transcript (tool calls inline, reasoning dim, answer streamed;
# the final answer re-rendered as markdown in place).
# --------------------------------------------------------------------------

def _tool_line(payload: dict) -> Text:
    name = payload.get("name")
    args = payload.get("input") or {}
    brief = ", ".join(f"{k}={v}" for k, v in args.items() if k != "video_id")
    return Text.from_markup(f"[cyan]→[/cyan] [bold]{name}[/bold]([dim]{brief}[/dim])")


def _accumulate_usage(usage: dict, payload: dict) -> None:
    u = payload.get("usage") or {}
    usage["input_tokens"] = usage.get("input_tokens", 0) + int(u.get("input_tokens") or 0)
    usage["output_tokens"] = usage.get("output_tokens", 0) + int(u.get("output_tokens") or 0)
    if payload.get("cost") is not None:
        usage["cost"] = round(usage.get("cost", 0.0) + float(payload["cost"]), 6)


async def _stream_run(broker, *, show_thinking: bool, live: bool, console: Console) -> tuple[str, dict]:
    """Drain one run's events, rendering live. Returns (answer, usage).

    `blocks` holds finished renderables (tool-call lines, past reasoning/answer
    segments); `reasoning_buf`/`answer_buf` hold the segment currently streaming.
    On run.completed the streamed answer is replaced in place by its markdown
    render. `live=False` (used by --json) consumes silently.
    """
    usage: dict = {}
    answer = ""
    blocks: list = []
    reasoning_buf: list[str] = []
    answer_buf: list[str] = []
    done = False

    def render():
        parts = list(blocks)
        # Reasoning is shown expanded only while it is actively streaming; once the
        # segment ends (answer begins / tool call / completion) it collapses to the
        # compact "💭 thought" line already pushed onto `blocks`.
        if show_thinking and reasoning_buf:
            parts.append(Text.from_markup("[dim]💭 [/dim]") + Text("".join(reasoning_buf), style="dim italic"))
        if done and answer:
            parts.append(Markdown(answer))
        elif answer_buf:
            parts.append(Text("".join(answer_buf)))
        if not parts:
            parts.append(Spinner("dots", text=Text(" thinking…", style="dim")))
        return Group(*parts)

    def _collapse_reasoning() -> None:
        """Collapse the current segment's live reasoning to a one-line summary."""
        if not reasoning_buf:
            return
        if show_thinking:
            words = len("".join(reasoning_buf).split())
            blocks.append(Text(f"💭 thought ({words} words)", style="dim"))
        reasoning_buf.clear()

    def _freeze_segment() -> None:
        _collapse_reasoning()
        if answer_buf:
            blocks.append(Text("".join(answer_buf)))
            answer_buf.clear()

    async def _pump(update) -> None:
        nonlocal answer, done
        async for ev in broker.events():
            etype = ev.get("type")
            p = ev.get("payload") or {}
            if etype == "chat.completion.chunk":
                for ch in p.get("choices") or []:
                    d = ch.get("delta") or {}
                    if isinstance(d.get("content"), str) and d["content"]:
                        # Answer text has begun -> this segment is done thinking;
                        # collapse the reasoning shown so far before streaming it.
                        _collapse_reasoning()
                        answer_buf.append(d["content"])
                    r = d.get("reasoning") or d.get("reasoning_content")
                    if isinstance(r, str):
                        reasoning_buf.append(r)
            elif etype == "tool.scheduled":
                _freeze_segment()
                blocks.append(_tool_line(p))
            elif etype == "tool.result":
                snippet = str(p.get("analysis") or "").strip().replace("\n", " ")
                if snippet:
                    blocks.append(Text(f"  {snippet[:160]}{'…' if len(snippet) > 160 else ''}", style="dim"))
            elif etype == "tool.failed":
                msg = (p.get("error") or {}).get("message", "tool failed")
                blocks.append(Text(f"  ✗ {msg}", style="red"))
            elif etype == "span.model_request_end":
                _accumulate_usage(usage, p)
            elif etype == "error":
                blocks.append(Text(f"error: {p.get('message') or p.get('code')}", style="red"))
            elif etype == "run.completed":
                _collapse_reasoning()
                answer = p.get("answer") or "".join(answer_buf)
                done = True
            update()

    if live:
        with Live(render(), console=console, refresh_per_second=12, vertical_overflow="visible") as _live:
            await _pump(lambda: _live.update(render()))
    else:
        await _pump(lambda: None)
    return answer, usage


async def _execute_turn(runner, broker, session_id: str, question: str, *,
                        output_structure: Optional[dict], show_thinking: bool,
                        live: bool, console: Console) -> tuple[str, dict]:
    """Submit one user message and stream the resulting run to completion."""
    import asyncio

    consumer = asyncio.create_task(_stream_run(broker, show_thinking=show_thinking, live=live, console=console))
    await runner.submit_user_message(question, output_structure=output_structure)
    if runner._run_task is not None:
        try:
            await runner._run_task
        except Exception:  # noqa: BLE001 - surfaced as an error event already
            pass
    # The run task has fully finished (including the runner's finally, which emits
    # a trailing status change). Publish exactly one close sentinel so the consumer
    # drains every event and leaves the queue empty for the next turn — deterministic
    # even if the run raised before emitting run.completed.
    await broker.publish(session_id, {"__close__": True, "seq": 0})
    return await consumer


# --------------------------------------------------------------------------
# The run driver (async)
# --------------------------------------------------------------------------

async def _build_session(*, video_id: str, model: str, mode: str, max_turns: int,
                          subtitles: Optional[str], duration: Optional[float]):
    """Create a runner backed by in-memory adapters and boot it. Returns
    (runner, broker, session_id)."""
    from ambient.lite import LiteBroker, LiteStore, _new_id
    from ambient.server.runner import SessionRunner
    from ambient.server.sandbox import SandboxLimits

    if subtitles:
        # Mirror run_agent: feed subtitles into the description tool's transcript.
        from ambient.tools import video_description as _vd
        _vd._TRANSCRIPT = Path(subtitles).read_text()

    store, broker = LiteStore(), LiteBroker()
    session_id = _new_id("ses")
    await store.create_session({
        "session_id": session_id, "status": "created", "video_id": video_id,
        "model": model, "mode": mode, "usage": {},
    })
    # Seed a file record so the source is treated as ready immediately and the
    # duration is reused (no re-probe). A local file is never a pending youtube source.
    await store.put_file({
        "video_id": video_id, "filename": f"{video_id}", "mime_type": "video/mp4",
        "source_type": "upload", "source_status": "ready",
        "description_status": None, "duration": duration or 0,
    })

    runner = SessionRunner(
        session_id=session_id, video_id=video_id, model=model,
        max_turns_per_run=max_turns, backend="inprocess",
        limits=SandboxLimits(), store=store, broker=broker, mode=mode, system=None,
    )
    # Agent mode boots a (no-op inprocess) sandbox + kicks off background
    # description resolution; fast mode samples frames inside its run.
    if mode != "fast":
        await runner.start_sandbox()
    return runner, broker, session_id


async def _run_analyze(*, video_id: str, question: str, model: str, mode: str,
                       max_turns: int, subtitles: Optional[str],
                       output_structure: Optional[dict], duration: Optional[float],
                       show_thinking: bool, live: bool, console: Console) -> tuple[str, dict, str]:
    runner, broker, session_id = await _build_session(
        video_id=video_id, model=model, mode=mode, max_turns=max_turns,
        subtitles=subtitles, duration=duration,
    )
    try:
        answer, usage = await _execute_turn(
            runner, broker, session_id, question, output_structure=output_structure,
            show_thinking=show_thinking, live=live, console=console,
        )
    finally:
        await runner.terminate()
    return answer, (runner.usage or usage), session_id


# --------------------------------------------------------------------------
# Commands
# --------------------------------------------------------------------------

@app.command()
def analyze(
    video: str = typer.Argument(..., help="Path to a local video file, or a YouTube/http(s) URL."),
    question: str = typer.Argument(..., help="What to ask about the video."),
    model: Optional[str] = typer.Option(None, "--model", "-m", help="Agent model id (default: settings.agent_model)."),
    vision_model: Optional[str] = typer.Option(None, "--vision-model", help="Override just the vision/tool model."),
    max_turns: int = typer.Option(20, "--max-turns", help="Max agent turns per run."),
    fast: bool = typer.Option(False, "--fast", help="Fast mode: one dense-frame vision pass, no tools."),
    subtitles: Optional[str] = typer.Option(None, "--subtitles", help="Path to a subtitle/transcript file to ground the answer."),
    schema: Optional[str] = typer.Option(None, "--schema", help="Path to a JSON Schema file; steers the final answer to that shape."),
    show_thinking: bool = typer.Option(False, "--show-thinking", help="Print the agent's reasoning as it works."),
    as_json: bool = typer.Option(False, "--json", help="Print a JSON result to stdout (all logs go to stderr)."),
) -> None:
    """Analyze a video and answer a question about it."""
    import asyncio

    _load_config_files()
    _apply_env(model, vision_model, WORKSPACE)

    problems = _preflight(require_key=True)
    if problems:
        for pr in problems:
            err.print(f"[red]✗[/red] {pr}")
        raise typer.Exit(code=1)

    output_structure = None
    if schema:
        try:
            output_structure = json.loads(Path(schema).read_text())
        except Exception as exc:  # noqa: BLE001
            raise typer.BadParameter(f"could not read --schema {schema!r}: {exc}")

    video_id = _resolve_video(video)
    duration = _probe_duration(video_id)

    # Import settings only now (env is populated) to resolve the effective model.
    from ambient.config import get_model_modalities, settings
    effective_model = model or settings.agent_model
    if not fast:
        mods = [m.value for m in get_model_modalities(effective_model)]
        if mods and "image" not in mods:
            err.print(f"[yellow]![/yellow] {effective_model} has no image input; using the text-only toolset (degraded).")

    if not as_json:
        err.print(Panel(
            f"[bold]model[/bold]   {effective_model}\n"
            f"[bold]video[/bold]   {video_id}"
            + (f"  ([dim]{duration:.0f}s[/dim])" if duration else "")
            + f"\n[bold]mode[/bold]    {'fast' if fast else 'agent'}\n"
            f"[bold]question[/bold]  {question}",
            title="ambient", border_style="cyan", expand=False,
        ))

    answer, usage, _sid = asyncio.run(_run_analyze(
        video_id=video_id, question=question, model=effective_model,
        mode="fast" if fast else "agent", max_turns=max_turns, subtitles=subtitles,
        output_structure=output_structure, duration=duration,
        show_thinking=show_thinking, live=not as_json, console=console,
    ))

    if as_json:
        console.print_json(data={
            "answer": answer,
            "video_id": video_id,
            "model": effective_model,
            "usage": {
                "input_tokens": usage.get("input_tokens", 0),
                "output_tokens": usage.get("output_tokens", 0),
                "cost": usage.get("cost", 0.0),
            },
        })
    else:
        # The live renderer already printed the markdown answer; just a usage footer.
        cost = usage.get("cost")
        toks = f"{usage.get('input_tokens', 0)} in / {usage.get('output_tokens', 0)} out tokens"
        err.print(f"[dim]{toks}" + (f" · ${cost:.4f}" if cost else "") + "[/dim]")


async def _chat_loop(*, video_id: str, model: str, mode: str, max_turns: int,
                     subtitles: Optional[str], duration: Optional[float],
                     show_thinking: bool) -> None:
    import asyncio

    runner, broker, session_id = await _build_session(
        video_id=video_id, model=model, mode=mode, max_turns=max_turns,
        subtitles=subtitles, duration=duration,
    )
    console.print(Panel(
        f"[bold]{video_id}[/bold]" + (f"  ([dim]{duration:.0f}s[/dim])" if duration else "")
        + f"\n[dim]model {model} · {'fast' if mode == 'fast' else 'agent'} mode · "
        f"Ctrl-D or /exit to quit[/dim]",
        title="ambient chat", border_style="cyan", expand=False,
    ))

    loop = asyncio.get_event_loop()
    try:
        while True:
            try:
                # Read input off-thread so background work (e.g. description
                # resolution) keeps progressing while we wait at the prompt.
                question = (await loop.run_in_executor(
                    None, lambda: console.input("[bold cyan]› [/bold cyan]"))).strip()
            except (EOFError, KeyboardInterrupt):
                console.print()
                break
            if not question:
                continue
            if question.lower() in ("/exit", "/quit", ":q"):
                break

            _, usage = await _execute_turn(
                runner, broker, session_id, question, output_structure=None,
                show_thinking=show_thinking, live=True, console=console,
            )
            cost = usage.get("cost")
            toks = f"{usage.get('input_tokens', 0)} in / {usage.get('output_tokens', 0)} out"
            err.print(f"[dim]{toks}" + (f" · ${cost:.4f}" if cost else "") + "[/dim]\n")
    finally:
        await runner.terminate()
    console.print("[dim]bye[/dim]")


@app.command()
def chat(
    video: str = typer.Argument(..., help="Path to a local video file, or a YouTube/http(s) URL."),
    model: Optional[str] = typer.Option(None, "--model", "-m", help="Agent model id (default: settings.agent_model)."),
    vision_model: Optional[str] = typer.Option(None, "--vision-model", help="Override just the vision/tool model."),
    max_turns: int = typer.Option(20, "--max-turns", help="Max agent turns per question."),
    fast: bool = typer.Option(False, "--fast", help="Fast mode: one dense-frame vision pass, no tools."),
    subtitles: Optional[str] = typer.Option(None, "--subtitles", help="Path to a subtitle/transcript file to ground answers."),
    show_thinking: bool = typer.Option(False, "--show-thinking", help="Stream the agent's reasoning as it works."),
) -> None:
    """Open an interactive chat over a video — ask many questions with context, streamed live."""
    import asyncio

    _load_config_files()
    _apply_env(model, vision_model, WORKSPACE)

    problems = _preflight(require_key=True)
    if problems:
        for pr in problems:
            err.print(f"[red]✗[/red] {pr}")
        raise typer.Exit(code=1)

    video_id = _resolve_video(video)
    duration = _probe_duration(video_id)

    from ambient.config import get_model_modalities, settings
    effective_model = model or settings.agent_model
    if not fast:
        mods = [m.value for m in get_model_modalities(effective_model)]
        if mods and "image" not in mods:
            err.print(f"[yellow]![/yellow] {effective_model} has no image input; using the text-only toolset (degraded).")

    asyncio.run(_chat_loop(
        video_id=video_id, model=effective_model, mode="fast" if fast else "agent",
        max_turns=max_turns, subtitles=subtitles, duration=duration, show_thinking=show_thinking,
    ))


@app.command()
def doctor() -> None:
    """Check the environment and print the resolved configuration."""
    _load_config_files()
    _apply_env(None, None, WORKSPACE)

    console.print("[bold]ambient doctor[/bold]\n")
    problems = _preflight(require_key=True)
    console.print(f"ffmpeg   : {'[green]ok[/green]' if _which('ffmpeg') else '[red]missing[/red]'}")
    console.print(f"ffprobe  : {'[green]ok[/green]' if _which('ffprobe') else '[red]missing[/red]'}")
    console.print(f"yt-dlp   : {'[green]ok[/green]' if _which('yt-dlp') else '[yellow]missing (URL input disabled)[/yellow]'}")

    key = _resolve_api_key()
    shown = f"{key[:6]}…redacted" if key else "[red]not set[/red]"
    console.print(f"api key  : {shown}")

    from ambient.config import settings
    console.print(f"base url : {settings.llm_base_url or '[red]not set[/red]'}")
    console.print(f"model    : {settings.agent_model}")
    console.print(f"workspace: {WORKSPACE}")

    if problems:
        console.print("\n[bold red]Problems:[/bold red]")
        for pr in problems:
            console.print(f"  [red]✗[/red] {pr}")
        raise typer.Exit(code=1)
    console.print("\n[green]All good — try:[/green] ambient analyze <video> \"your question\"")


@config_app.command("set")
def config_set(key: str, value: str) -> None:
    """Persist KEY=VALUE to ~/.config/ambient/env (chmod 600)."""
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    found = False
    if CONFIG_FILE.exists():
        for line in CONFIG_FILE.read_text().splitlines():
            if line.strip() and not line.strip().startswith("#") and line.split("=", 1)[0].strip() == key:
                lines.append(f"{key}={value}")
                found = True
            else:
                lines.append(line)
    if not found:
        lines.append(f"{key}={value}")
    CONFIG_FILE.write_text("\n".join(lines) + "\n")
    CONFIG_FILE.chmod(stat.S_IRUSR | stat.S_IWUSR)
    console.print(f"[green]set[/green] {key} in {CONFIG_FILE}")


@config_app.command("show")
def config_show() -> None:
    """Show persisted config (secrets redacted)."""
    if not CONFIG_FILE.exists():
        console.print(f"[dim]no config at {CONFIG_FILE}[/dim]")
        return
    for line in CONFIG_FILE.read_text().splitlines():
        if "=" in line and not line.strip().startswith("#"):
            k, _, v = line.partition("=")
            redacted = (v[:6] + "…") if ("KEY" in k.upper() or "SECRET" in k.upper()) and len(v) > 6 else v
            console.print(f"{k}={redacted}")
        elif line.strip():
            console.print(line)


if __name__ == "__main__":
    app()

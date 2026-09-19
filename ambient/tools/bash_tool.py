"""Bash tool — run a shell command, selecting the backend the same way media
tools do: the per-session E2B sandbox when one is active (`current_media_box`),
otherwise the HOST.

SECURITY: on the host backend this runs arbitrary shell via `/bin/sh -c` with the
server process's privileges and **no OS isolation** — like fx's local executor,
it trusts the operator's machine. It is opt-in (`settings.enable_bash_tool`,
default False). For untrusted use, run with the e2b backend, which confines the
command to the ephemeral sandbox. The host path applies only soft hygiene:
  * a wall-clock timeout, then SIGKILL of the whole process group,
  * cwd confinement (defaults to the video workspace),
  * an output byte cap per stream.
"""
from __future__ import annotations

import asyncio
import logging
import os
import signal
import subprocess
from typing import Dict, List, Optional, Tuple

from pydantic import BaseModel, Field

from ambient.config import settings
from ambient.tools.video_backend import current_media_box

log = logging.getLogger(__name__)


class BashTool(BaseModel):
    command: str = Field(
        description=(
            "The shell command to run, executed via `/bin/sh -c`. Must be "
            "non-interactive (no prompts/pagers). Useful for ffmpeg/ffprobe, "
            "inspecting files, or computing values."
        )
    )
    timeout: Optional[int] = Field(
        default=None,
        description="Max seconds before the command is killed. Defaults to the server setting.",
    )


def _cap(text: str, limit: int) -> str:
    text = text or ""
    if len(text) <= limit:
        return text
    return text[:limit] + f"\n… [truncated, {len(text) - limit} more bytes]"


def _format(result: dict, limit: int) -> str:
    stdout = result.get("stdout") or ""
    stderr = result.get("stderr") or ""
    exit_code = int(result.get("exit_code") or 0)
    parts = [f"exit_code: {exit_code}"]
    if stdout.strip():
        parts.append(f"stdout:\n{_cap(stdout, limit)}")
    if stderr.strip():
        parts.append(f"stderr:\n{_cap(stderr, limit)}")
    if not stdout.strip() and not stderr.strip():
        parts.append("(no output)")
    return "\n\n".join(parts)


def _run_host(command: str, timeout: int, cwd: str) -> dict:
    """Blocking host execution with soft hygiene. Runs in its own process group so
    a timeout kills the whole tree, not just the shell."""
    os.makedirs(cwd, exist_ok=True)
    try:
        proc = subprocess.Popen(
            command, shell=True, cwd=cwd, text=True,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            start_new_session=True,  # new process group -> killpg on timeout
        )
    except Exception as exc:  # noqa: BLE001
        return {"stdout": "", "stderr": f"failed to start command: {exc}", "exit_code": 127}
    try:
        stdout, stderr = proc.communicate(timeout=timeout)
        return {"stdout": stdout, "stderr": stderr, "exit_code": proc.returncode}
    except subprocess.TimeoutExpired:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            pass
        stdout, stderr = proc.communicate()
        return {
            "stdout": stdout or "",
            "stderr": (stderr or "") + f"\ncommand timed out after {timeout}s",
            "exit_code": 124,
        }


async def bash(command: str, timeout: Optional[int] = None) -> Tuple[str, List[Dict]]:
    """Run `command` in the active backend and return (analysis, user_message_contents).

    The e2b sandbox (when a `current_media_box` is set) runs it inside the box;
    otherwise it runs on the host. Returns the exit code plus captured
    stdout/stderr (each byte-capped). No media attachments.
    """
    if not settings.enable_bash_tool:
        return ("bash tool is disabled. Set ENABLE_BASH_TOOL=true to enable it.", [])

    timeout = int(timeout or settings.bash_timeout_seconds)
    limit = int(settings.bash_max_output_bytes)

    box = current_media_box.get()
    if box is not None:
        log.info("[bash] e2b: %s", command)
        result = await asyncio.to_thread(box.run_shell, command, timeout)
    else:
        cwd = settings.bash_cwd or settings.video_folder
        log.info("[bash] host (cwd=%s): %s", cwd, command)
        result = await asyncio.to_thread(_run_host, command, timeout, cwd)

    return (_format(result, limit), [])

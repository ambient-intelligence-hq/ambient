"""Shared types for tool execution.

Tools always run in the server process (see `dispatcher.ToolDispatcher`). The
only thing that varies by backend is where `VideoFrameTools`' media ops execute
(local ffmpeg/decord vs. an E2B sandbox), selected by `settings.sandbox_backend`.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Optional

from typing import Protocol
@dataclass
class SandboxLimits:
    cpu_seconds: Optional[int] = None
    memory_mb: Optional[int] = None
    wall_seconds: Optional[int] = None


class MediaSandbox(Protocol):
    backend: Literal["inprocess", "e2b"]

    def __init__(self, *args, **kwargs):
        # create the sandbox and initialize the sandbox id
        pass
    
    def create(self, *args, **kwargs):
        pass

    def connect(self, sandbox_id: str):
        # reattach to an already-running sandbox by id (cross-worker rehydration)
        pass

    @property
    def id(self) -> str:
        pass

    def run(self, argv: list[str]) -> dict:
        pass

    def run_shell(self, command: str, timeout: Optional[int] = None) -> dict:
        # Run a raw shell command inside the sandbox. Returns
        # {stdout, stderr, exit_code}. Used by the bash tool (e2b backend).
        pass

    def kill(self) -> None:
        pass

    def get_status(self):
        pass

@dataclass
class ToolEvent:
    """One step of tool execution.

    `type` is one of:
        started  -> dispatcher accepted the call
        progress -> optional progress signal
        result   -> terminal success {analysis, attachments, ...}
        failed   -> terminal failure {code, message, ...}
    """
    type: Literal["started", "progress", "result", "failed"]
    tool_use_id: str
    name: str
    data: dict[str, Any] = field(default_factory=dict)

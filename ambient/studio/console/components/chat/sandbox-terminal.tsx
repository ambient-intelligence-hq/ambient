"use client";

import { ChevronDown, TerminalIcon } from "lucide-react";
import { type ReactNode, useEffect, useMemo, useRef, useState } from "react";
import type { ChatMessage } from "@/lib/types";
import { cn } from "@/lib/utils";

// A read-only "sandbox" terminal view that mirrors the bash commands the agent
// runs (and their output) as if you were watching the sandbox shell. It reads
// the bash `dynamic-tool` parts straight off the conversation, so it works both
// live (streaming) and on reload (persisted parts).

const BANNER = String.raw`
 █████╗ ███╗   ███╗██████╗ ██╗███████╗███╗   ██╗████████╗
██╔══██╗████╗ ████║██╔══██╗██║██╔════╝████╗  ██║╚══██╔══╝
███████║██╔████╔██║██████╔╝██║█████╗  ██╔██╗ ██║   ██║
██╔══██║██║╚██╔╝██║██╔══██╗██║██╔══╝  ██║╚██╗██║   ██║
██║  ██║██║ ╚═╝ ██║██████╔╝██║███████╗██║ ╚████║   ██║
╚═╝  ╚═╝╚═╝     ╚═╝╚═════╝ ╚═╝╚══════╝╚═╝  ╚═══╝   ╚═╝`;

type BashExec = {
  id: string;
  command: string;
  state: string;
  output: string;
  errorText?: string;
};

type LoosePart = {
  type?: string;
  toolName?: string;
  toolCallId?: string;
  state?: string;
  input?: unknown;
  output?: unknown;
  errorText?: string;
};

function extractBashExecs(messages: ChatMessage[]): BashExec[] {
  const execs: BashExec[] = [];
  for (const message of messages) {
    for (const part of (message.parts ?? []) as LoosePart[]) {
      if (part.type !== "dynamic-tool" || part.toolName !== "bash") {
        continue;
      }
      const input = (part.input ?? {}) as Record<string, unknown>;
      const command =
        typeof input.command === "string" ? input.command : "";
      const output =
        typeof part.output === "string"
          ? part.output
          : part.output == null
            ? ""
            : JSON.stringify(part.output, null, 2);
      execs.push({
        command,
        errorText: part.errorText,
        id: part.toolCallId ?? `${execs.length}`,
        output,
        state: part.state ?? "",
      });
    }
  }
  return execs;
}

// The bash tool formats results as `exit_code: N\n\nstdout:\n...\n\nstderr:\n...`.
function parseBashOutput(raw: string) {
  const exitMatch = raw.match(/^exit_code:\s*(-?\d+)/);
  const exitCode = exitMatch ? Number(exitMatch[1]) : null;
  const soIdx = raw.indexOf("stdout:\n");
  const seIdx = raw.indexOf("stderr:\n");
  let stdout = "";
  let stderr = "";
  if (soIdx !== -1) {
    const end = seIdx !== -1 && seIdx > soIdx ? seIdx : raw.length;
    stdout = raw.slice(soIdx + "stdout:\n".length, end).replace(/\s+$/, "");
  }
  if (seIdx !== -1) {
    stderr = raw.slice(seIdx + "stderr:\n".length).replace(/\s+$/, "");
  }
  return { exitCode, stderr, stdout };
}

// Prompt and command flow as one run of text, like a real terminal: a long
// command wraps back to column 0 rather than hanging under the command.
function TerminalPrompt({ children }: { children: ReactNode }) {
  return (
    <div className="whitespace-pre-wrap break-all">
      <span className="select-none">
        <span className="text-emerald-700 dark:text-emerald-400">ambient@sandbox</span>
        <span className="text-neutral-500">:</span>
        <span className="text-sky-700 dark:text-sky-400">~</span>
        <span className="text-neutral-500">$ </span>
      </span>
      {children}
    </div>
  );
}

const Cursor = () => <span className="animate-pulse text-neutral-500 dark:text-neutral-400">▍</span>;

function ExecView({ exec }: { exec: BashExec }) {
  const running = exec.state === "input-available" || exec.state === "input-streaming";
  const { exitCode, stdout, stderr } = parseBashOutput(exec.output);
  return (
    <div>
      <TerminalPrompt>
        <span className="text-neutral-900 dark:text-neutral-100">{exec.command}</span>
      </TerminalPrompt>
      {running ? (
        <div>
          <Cursor />
        </div>
      ) : null}
      {exec.state === "output-error" && (
        <div className="whitespace-pre-wrap break-all text-rose-600 dark:text-rose-400">
          {exec.errorText || "command failed"}
        </div>
      )}
      {stdout && (
        <div className="whitespace-pre-wrap break-all text-neutral-700 dark:text-neutral-300">
          {stdout}
        </div>
      )}
      {stderr && (
        <div className="whitespace-pre-wrap break-all text-rose-600 dark:text-rose-400">
          {stderr}
        </div>
      )}
      {exitCode != null && exitCode !== 0 && (
        <div className="text-rose-600/80 dark:text-rose-400/80">exit {exitCode}</div>
      )}
    </div>
  );
}

export function SandboxTerminal({ messages }: { messages: ChatMessage[] }) {
  const execs = useMemo(() => extractBashExecs(messages), [messages]);
  const last = execs.at(-1);
  const idle =
    !last ||
    (last.state !== "input-available" && last.state !== "input-streaming");
  const [open, setOpen] = useState(true);
  const bodyRef = useRef<HTMLDivElement>(null);

  // Restore the collapsed/expanded preference (per viewer).
  useEffect(() => {
    try {
      const saved = localStorage.getItem("ambient.terminalOpen");
      if (saved != null) {
        setOpen(saved === "1");
      }
    } catch {
      /* private mode / blocked storage — keep the default */
    }
  }, []);

  const toggle = () => {
    setOpen((v) => {
      const next = !v;
      try {
        localStorage.setItem("ambient.terminalOpen", next ? "1" : "0");
      } catch {
        /* ignore */
      }
      return next;
    });
  };

  // Follow the tail as new commands/output stream in.
  // biome-ignore lint/correctness/useExhaustiveDependencies: scroll on exec growth
  useEffect(() => {
    if (open && bodyRef.current) {
      bodyRef.current.scrollTop = bodyRef.current.scrollHeight;
    }
  }, [execs.length, execs.at(-1)?.output, open]);

  return (
    // Open, the terminal is the same 16:9 frame as the video above it (same
    // width, same aspect), so the stage reads as two matching panes.
    <div
      className={cn(
        "mx-auto flex w-full max-w-3xl shrink-0 flex-col overflow-hidden rounded-2xl border bg-[#fcfcfb] dark:bg-[#0a0c10]",
        open && "aspect-video"
      )}
    >
      {/* Title bar — classic terminal chrome, doubles as the collapse toggle. */}
      <button
        aria-expanded={open}
        className="flex w-full shrink-0 items-center gap-2 px-3 py-2 text-left"
        onClick={toggle}
        type="button"
      >
        <span className="flex items-center gap-1.5">
          <span className="size-3 rounded-full bg-[#ff5f56]" />
          <span className="size-3 rounded-full bg-[#ffbd2e]" />
          <span className="size-3 rounded-full bg-[#27c93f]" />
        </span>
        <TerminalIcon className="ml-1 size-3.5 text-neutral-500 dark:text-neutral-400" />
        <span className="font-medium font-mono text-neutral-600 text-xs dark:text-neutral-300">
          AMBIENT — sandbox
        </span>
        <span className="rounded bg-neutral-200/70 px-1.5 py-0.5 font-mono text-[10px] text-neutral-500 dark:bg-neutral-800 dark:text-neutral-400">
          read-only
        </span>
        {execs.length > 0 && (
          <span className="font-mono text-[10px] text-neutral-500">
            {execs.length} cmd{execs.length === 1 ? "" : "s"}
          </span>
        )}
        <ChevronDown
          className={cn(
            "ml-auto size-4 text-neutral-500 transition-transform",
            !open && "-rotate-90"
          )}
        />
      </button>

      {open && (
        <div
          className="min-h-0 flex-1 overflow-auto px-3 pb-3 font-mono text-[11px] leading-relaxed"
          ref={bodyRef}
        >
          <pre className="mb-2 select-none overflow-x-auto text-[8px] text-emerald-600/90 leading-[1.1] sm:text-[10px] dark:text-emerald-400/90">
            {BANNER}
          </pre>
          <div className="mb-3 text-neutral-500">
            sandbox shell · commands the agent runs appear here
          </div>
          {execs.map((exec) => (
            <ExecView exec={exec} key={exec.id} />
          ))}
          {/* A fresh prompt waits at the bottom whenever nothing is running. */}
          {idle ? (
            <TerminalPrompt>
              <Cursor />
            </TerminalPrompt>
          ) : null}
        </div>
      )}
    </div>
  );
}

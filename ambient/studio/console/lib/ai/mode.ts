"use client";

// Global run MODE (track), chosen before a session via the top-center pill.
// Separate from agent definitions: an agent is model/endpoint/system; mode is
// how the engine analyzes video for this session.
//   agent -> multi-step tool loop (searches + focuses clips)
//   fast  -> single dense-frame pass, no tools
import { useSyncExternalStore } from "react";

export type AgentMode = "agent" | "fast";

const KEY = "ambient.mode";
const EVENT = "ambient:mode-change";

export function getMode(): AgentMode {
  if (typeof window === "undefined") return "agent";
  try {
    return window.localStorage.getItem(KEY) === "fast" ? "fast" : "agent";
  } catch {
    return "agent";
  }
}

export function setMode(mode: AgentMode) {
  try {
    window.localStorage.setItem(KEY, mode);
    window.dispatchEvent(new Event(EVENT));
  } catch {
    /* ignore */
  }
}

export function useMode(): [AgentMode, (m: AgentMode) => void] {
  const mode = useSyncExternalStore(
    (cb) => {
      if (typeof window === "undefined") return () => {};
      window.addEventListener(EVENT, cb);
      window.addEventListener("storage", cb);
      return () => {
        window.removeEventListener(EVENT, cb);
        window.removeEventListener("storage", cb);
      };
    },
    getMode,
    () => "agent" as AgentMode
  );
  return [mode, setMode];
}

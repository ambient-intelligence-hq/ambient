"use client";

// Studio-level "agent definitions" — a saved LLM setup: model + system prompt +
// (optional) its own LLM endpoint (base URL + key). Selecting one makes the chat
// route create/reuse an engine agent with that config and bind the session to
// it. Mode (agent/fast track) is NOT part of a definition — it's a separate
// global toggle (see lib/ai/mode.ts).
//
// There is always a built-in "Vanilla" definition with no overrides, which runs
// on the engine's configured defaults. The active definition is the last used.
//
// Stored client-side (localStorage) with a tiny pub/sub so the composer selector
// and the settings page stay in sync.

import { useSyncExternalStore } from "react";

export type AgentDef = {
  id: string;
  name: string;
  /** LLM model id. Empty = engine default. */
  model: string;
  /** System prompt. Empty = engine default. */
  system: string;
  /** LLM endpoint base URL. Empty = engine default. */
  baseUrl: string;
  /** LLM endpoint API key. Empty = engine default. */
  apiKey: string;
  /** The Vanilla built-in can't be deleted or renamed. */
  builtin?: boolean;
};

const DEFS_KEY = "ambient.agentDefs";
const ACTIVE_KEY = "ambient.activeAgentDef";
const EVENT = "ambient:agentdefs-change";

export const VANILLA_ID = "vanilla";

// The one built-in: no overrides -> pure engine defaults.
export const BUILTIN_DEFS: AgentDef[] = [
  {
    id: VANILLA_ID,
    name: "Vanilla",
    model: "",
    system: "",
    baseUrl: "",
    apiKey: "",
    builtin: true,
  },
];

/** True when a definition overrides nothing (runs on engine defaults). */
export function isVanilla(def: AgentDef): boolean {
  return !(def.model || def.system || def.baseUrl || def.apiKey);
}

function read<T>(key: string, fallback: T): T {
  if (typeof window === "undefined") return fallback;
  try {
    const raw = window.localStorage.getItem(key);
    return raw ? (JSON.parse(raw) as T) : fallback;
  } catch {
    return fallback;
  }
}

function emit() {
  if (typeof window !== "undefined") {
    window.dispatchEvent(new Event(EVENT));
  }
}

export function getAgentDefs(): AgentDef[] {
  const stored = read<AgentDef[] | null>(DEFS_KEY, null);
  if (!stored || stored.length === 0) return BUILTIN_DEFS;
  const customs = stored.filter((d) => !d.builtin && d.id !== VANILLA_ID);
  return [...BUILTIN_DEFS, ...customs];
}

export function saveAgentDefs(defs: AgentDef[]) {
  try {
    window.localStorage.setItem(DEFS_KEY, JSON.stringify(defs));
    emit();
  } catch {
    /* ignore */
  }
}

export function upsertAgentDef(def: AgentDef) {
  const defs = getAgentDefs();
  const idx = defs.findIndex((d) => d.id === def.id);
  if (idx >= 0) defs[idx] = def;
  else defs.push(def);
  saveAgentDefs(defs);
}

export function deleteAgentDef(id: string) {
  if (id === VANILLA_ID) return;
  saveAgentDefs(getAgentDefs().filter((d) => d.id !== id || d.builtin));
  if (getActiveAgentDefId() === id) setActiveAgentDefId(VANILLA_ID);
}

// The active definition doubles as "last used" — set on selection, defaulted on
// the next session.
// Stored as a bare string (setActiveAgentDefId), so it's read raw rather than
// through read()'s JSON.parse — which throws on a bare id and silently fell back
// to Vanilla, so no custom definition ever took effect. Tolerates a
// JSON-quoted value too.
export function getActiveAgentDefId(): string {
  if (typeof window === "undefined") return VANILLA_ID;
  try {
    const raw = window.localStorage.getItem(ACTIVE_KEY);
    if (!raw) return VANILLA_ID;
    return raw.startsWith('"') ? (JSON.parse(raw) as string) : raw;
  } catch {
    return VANILLA_ID;
  }
}

export function setActiveAgentDefId(id: string) {
  try {
    window.localStorage.setItem(ACTIVE_KEY, id);
    emit();
  } catch {
    /* ignore */
  }
}

export function getActiveAgentDef(): AgentDef {
  const defs = getAgentDefs();
  return defs.find((d) => d.id === getActiveAgentDefId()) ?? defs[0];
}

export function newAgentDefId(): string {
  return `def_${Math.random().toString(36).slice(2, 10)}`;
}

// ---- React binding --------------------------------------------------------
function subscribe(cb: () => void) {
  if (typeof window === "undefined") return () => {};
  window.addEventListener(EVENT, cb);
  window.addEventListener("storage", cb);
  return () => {
    window.removeEventListener(EVENT, cb);
    window.removeEventListener("storage", cb);
  };
}

let cachedDefs: AgentDef[] | null = null;
let cachedActive: string | null = null;

export function useAgentDefs(): {
  defs: AgentDef[];
  activeId: string;
  active: AgentDef;
} {
  const defs = useSyncExternalStore(
    (cb) =>
      subscribe(() => {
        cachedDefs = null;
        cb();
      }),
    () => {
      cachedDefs ??= getAgentDefs();
      return cachedDefs;
    },
    () => BUILTIN_DEFS
  );
  const activeId = useSyncExternalStore(
    (cb) =>
      subscribe(() => {
        cachedActive = null;
        cb();
      }),
    () => {
      cachedActive ??= getActiveAgentDefId();
      return cachedActive;
    },
    () => VANILLA_ID
  );
  const active = defs.find((d) => d.id === activeId) ?? defs[0];
  return { defs, activeId, active };
}

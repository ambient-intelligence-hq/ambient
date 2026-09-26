// Turn bookkeeping shared by the chat routes. Pure — no I/O — so it's unit-tested
// directly (lib/ambient/turn.test.ts).
//
// A "turn" is a user message plus the assistant message that answers it. The
// assistant message is streamed from an engine run that is independent of any
// HTTP request, so the copy the Studio persists can be cut mid-run (a reload, a
// navigation, a network drop) while the run keeps going. These helpers decide how
// to persist that message and when to rebuild it from the engine.

import type { AmbientFinishMetadata } from "@/lib/ai/ambient-provider";

type MessageLike = { id: string; role: string; metadata?: unknown };

/** True once the engine turn reached a terminal state (see AmbientFinishMetadata). */
export function isRunComplete(metadata: unknown): boolean {
  return (
    (metadata as { runComplete?: unknown } | null | undefined)?.runComplete ===
    true
  );
}

/** Index of the last user message, or -1. */
export function lastUserIndex(messages: readonly MessageLike[]): number {
  for (let i = messages.length - 1; i >= 0; i--) {
    if (messages[i].role === "user") {
      return i;
    }
  }
  return -1;
}

export type TurnSavePlan =
  | { skip: true }
  | { skip: false; deleteIds: string[]; exists: boolean };

/**
 * How to persist the assistant message for a chat's latest turn — the one after
 * its last user message. Several writers race to save the same turn: the original
 * request and any number of resumes (reload, reconnect, another tab), each of
 * which may itself be cut mid-run. So:
 *   - it replaces every other assistant row of the turn (a resume's message has a
 *     fresh id; without this the turn accumulates duplicates), and
 *   - it never overwrites a complete answer with a partial one (a cut stream that
 *     saves after the resume that finished).
 */
export function planTurnAssistantSave(
  rows: readonly MessageLike[],
  incoming: { id: string; metadata?: unknown }
): TurnSavePlan {
  const turn = rows
    .slice(lastUserIndex(rows) + 1)
    .filter((r) => r.role === "assistant");
  if (
    !isRunComplete(incoming.metadata) &&
    turn.some((r) => isRunComplete(r.metadata))
  ) {
    return { skip: true };
  }
  return {
    skip: false,
    deleteIds: turn.filter((r) => r.id !== incoming.id).map((r) => r.id),
    exists: rows.some((r) => r.id === incoming.id),
  };
}

export type TurnAnchorLike = { messageId: string; state: string };

/**
 * The messages to hand the client when it loads a chat, and whether its latest
 * turn should be resumed from the engine.
 *
 * A turn is resumable only when it's bound to an engine run (an anchor for its
 * user message that isn't `failed`) and its answer isn't known to be complete.
 * If a partial answer was persisted, it's dropped here: the client then sees a
 * trailing user message and resumes, and the replay rebuilds the answer from the
 * engine — complete, or still streaming if the run is ongoing. (Leaving the
 * partial in place is what made a reloaded running session look finished.)
 *
 * Without an anchor we don't know which engine run answers the turn, so nothing
 * is resumed — replaying "the engine's latest run" can show a different turn.
 */
export function prepareMessagesForLoad<T extends MessageLike>(
  messages: T[],
  anchor: TurnAnchorLike | null
): { messages: T[]; resumable: boolean } {
  const userIdx = lastUserIndex(messages);
  const anchored =
    userIdx >= 0 &&
    anchor !== null &&
    anchor.messageId === messages[userIdx].id &&
    anchor.state !== "failed";
  if (!anchored) {
    return { messages, resumable: false };
  }
  const tail = messages.slice(userIdx + 1);
  if (tail.some((m) => m.role !== "assistant" || isRunComplete(m.metadata))) {
    return { messages, resumable: false };
  }
  return { messages: messages.slice(0, userIdx + 1), resumable: true };
}

// Lifts the Ambient provider's finish metadata (AmbientFinishMetadata: run
// completion + the engine's cumulative usage) onto the assistant message. It
// rides the finish part's providerMetadata — per-step or overall depending on
// the AI-SDK version — so read it from any part that carries it. Shared with
// the resume route so both writers mark turns the same way.
export function ambientMessageMetadata({ part }: { part: unknown }) {
  const meta = (part as { providerMetadata?: { ambient?: AmbientFinishMetadata } })
    .providerMetadata?.ambient;
  if (!meta || typeof meta.runComplete !== "boolean") {
    return undefined;
  }
  return {
    runComplete: meta.runComplete,
    ...(meta.usage ? { usage: meta.usage } : {}),
  } as Record<string, unknown>;
}

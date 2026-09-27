import "server-only";

// Durable chat → engine pointers, stored on the chat row (Postgres).
//
// Resume needs to find the engine run for a chat from a *different* request than
// the one that started it (a page reload, a Fast-Refresh remount, another Next
// worker), and a follow-up after a restart needs the chat's engine session. The
// engine run + its events are already durable on the engine side; these are just
// the pointers to them:
//
//   Chat.engineSessionId  -> the engine session (one per chat)
//   Chat.turnAnchor       -> TurnAnchor for the chat's latest turn
//
// These used to live in Redis (with 24h/7d TTLs). Keeping them on the chat row
// makes them permanent and means the Studio needs no Redis at all. Writes are
// no-ops until the chat row exists (it's created before the engine is called).

import {
  getChatEnginePointers,
  setChatEngineSessionId,
  setChatTurnAnchor,
} from "@/lib/db/queries";

export async function setChatSession(
  chatId: string,
  sessionId: string
): Promise<void> {
  try {
    await setChatEngineSessionId({ id: chatId, engineSessionId: sessionId });
  } catch {
    /* best-effort */
  }
}

export async function getChatSessionId(chatId: string): Promise<string | null> {
  try {
    return (await getChatEnginePointers({ id: chatId }))?.engineSessionId ?? null;
  } catch {
    return null;
  }
}

// Binds a chat's latest user message to the exact engine run that answers it.
//
// Resume replays that run by `afterSeq` (just before the engine's user.message
// event) rather than asking the engine for its "latest run", which can be a
// different run — e.g. when this message never reached the engine. Replaying the
// wrong run is exactly how a turn ends up "repeating" an earlier one.
//
//   pending — the message is about to be sent; the run seq isn't known yet
//   sent    — the engine accepted it; `afterSeq` locates its run
//   failed  — the engine never accepted it; there is no run to resume
export type TurnAnchor = {
  messageId: string;
  sessionId: string;
  state: "pending" | "sent" | "failed";
  afterSeq?: number;
};

export async function setTurnAnchor(
  chatId: string,
  anchor: TurnAnchor
): Promise<void> {
  try {
    await setChatTurnAnchor({ id: chatId, turnAnchor: anchor });
  } catch {
    /* best-effort */
  }
}

export async function getTurnAnchor(chatId: string): Promise<TurnAnchor | null> {
  try {
    const anchor = (await getChatEnginePointers({ id: chatId }))?.turnAnchor;
    return anchor ? (anchor as TurnAnchor) : null;
  } catch {
    return null;
  }
}

// The anchor for `messageId`, waiting out a `pending` one (a resume that lands
// while the original request is mid-send) so it never guesses. Returns null when
// there's no anchor for this message.
export async function waitForTurnAnchor(
  chatId: string,
  messageId: string,
  timeoutMs = 15_000
): Promise<TurnAnchor | null> {
  const deadline = Date.now() + timeoutMs;
  for (;;) {
    const anchor = await getTurnAnchor(chatId);
    if (!anchor || anchor.messageId !== messageId) {
      return null;
    }
    if (anchor.state !== "pending" || Date.now() >= deadline) {
      return anchor;
    }
    await new Promise((r) => setTimeout(r, 250));
  }
}

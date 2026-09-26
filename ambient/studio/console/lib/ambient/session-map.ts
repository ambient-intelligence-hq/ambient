// Durable chat → engine pointers (Redis).
//
// Resume needs to find the engine run for a chat from a *different* request than
// the one that started it (a page reload, a Fast-Refresh remount, another Next
// worker). An in-memory Map doesn't survive those, so we persist the pointers in
// Redis. The engine run + its events are already durable on the engine side;
// these are just the pointers to them.
//
//   ambient:chat-session:<chatId>  -> engine session id (one per chat)
//   ambient:chat-turn:<chatId>     -> TurnAnchor for the chat's latest turn

import { createClient, type RedisClientType } from "redis";

const SESSION_TTL_SECONDS = 60 * 60 * 24; // 24h — matches the engine's run retention window
// Turn anchors outlive a browser session so a chat reopened days later can still
// be repaired from the engine's (durable) event log if its last turn was cut.
const TURN_TTL_SECONDS = 60 * 60 * 24 * 7;
const sessionKey = (chatId: string) => `ambient:chat-session:${chatId}`;
const turnKey = (chatId: string) => `ambient:chat-turn:${chatId}`;

let clientPromise: Promise<RedisClientType> | null = null;

function getClient(): Promise<RedisClientType> | null {
  const url = process.env.REDIS_URL;
  if (!url) {
    return null;
  }
  if (!clientPromise) {
    const client = createClient({ url }) as RedisClientType;
    client.on("error", () => {
      /* swallow — resume is best-effort */
    });
    clientPromise = client.connect().then(() => client);
  }
  return clientPromise;
}

export async function setChatSession(
  chatId: string,
  sessionId: string
): Promise<void> {
  try {
    const client = await getClient();
    await client?.set(sessionKey(chatId), sessionId, { EX: SESSION_TTL_SECONDS });
  } catch {
    /* best-effort */
  }
}

export async function getChatSessionId(chatId: string): Promise<string | null> {
  try {
    const client = await getClient();
    return (await client?.get(sessionKey(chatId))) ?? null;
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
    const client = await getClient();
    await client?.set(turnKey(chatId), JSON.stringify(anchor), {
      EX: TURN_TTL_SECONDS,
    });
  } catch {
    /* best-effort */
  }
}

export async function getTurnAnchor(chatId: string): Promise<TurnAnchor | null> {
  try {
    const client = await getClient();
    const raw = await client?.get(turnKey(chatId));
    return raw ? (JSON.parse(raw) as TurnAnchor) : null;
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

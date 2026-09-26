// Server-side client for the Ambient engine (Anthropic Managed Agents protocol).
// Used by the Ambient AI-SDK provider (lib/ai/ambient-provider.ts) to drive the
// engine as if it were a model, and by API routes for files/sessions.
//
// Config comes from env (server-only): ENGINE_URL + ENGINE_API_KEY.

const ENGINE_URL = (process.env.ENGINE_URL || "http://127.0.0.1:8080").replace(/\/$/, "");
const ENGINE_API_KEY = process.env.ENGINE_API_KEY || "dev-token";

function headers(extra: Record<string, string> = {}): Record<string, string> {
  return { "x-api-key": ENGINE_API_KEY, ...extra };
}

async function json(method: string, path: string, body?: unknown) {
  const res = await fetch(`${ENGINE_URL}/v1${path}`, {
    method,
    headers: body ? headers({ "content-type": "application/json" }) : headers(),
    body: body ? JSON.stringify(body) : undefined,
    cache: "no-store",
  });
  if (!res.ok) {
    throw new Error(`engine ${method} ${path} -> ${res.status}: ${(await res.text()).slice(0, 300)}`);
  }
  return res.status === 204 ? null : res.json();
}

export interface EngineFile {
  id: string;
  filename: string;
  mime_type: string;
  size_bytes: number;
  created_at: string;
  description_status?: string | null;
  description?: string;
  source_type?: string | null;
  source_status?: string | null;
  source_error?: string | null;
}
export interface EngineSession {
  id: string;
  status: string;
  metadata?: Record<string, unknown>;
  created_at: string;
  updated_at: string;
}

// ---- files -----------------------------------------------------------------
export const listFiles = (): Promise<{ data: EngineFile[] }> => json("GET", "/files");
export const getFile = (id: string): Promise<EngineFile> => json("GET", `/files/${id}`);
export const deleteFile = (id: string) => json("DELETE", `/files/${id}`);

// Ranged content URL, proxied through the Studio (keeps the engine key server-side).
export const contentPath = (id: string) => `/api/ambient/files/${id}/content`;

export async function uploadFile(file: Blob, filename: string): Promise<EngineFile> {
  const form = new FormData();
  form.append("file", file, filename);
  const res = await fetch(`${ENGINE_URL}/v1/files`, { method: "POST", headers: headers(), body: form });
  if (!res.ok) throw new Error(`upload -> ${res.status}: ${(await res.text()).slice(0, 200)}`);
  return res.json();
}

// Import a video from a YouTube URL (the engine downloads + prepares it). Returns
// the file metadata (content-addressed, so re-importing the same URL is a no-op).
export const importYouTube = (url: string): Promise<EngineFile> =>
  json("POST", "/files/import", { source_type: "youtube", url });

// ---- sessions --------------------------------------------------------------
export const listSessions = (): Promise<{ data: EngineSession[] }> => json("GET", "/sessions");
export const getSession = (id: string): Promise<EngineSession> => json("GET", `/sessions/${id}`);

// The video an engine session is bound to (set at session creation).
export async function getSessionVideoId(sessionId: string): Promise<string | null> {
  const s = await getSession(sessionId);
  const videoId = s.metadata?.video_id;
  return typeof videoId === "string" && videoId ? videoId : null;
}

// Create an engine agent from a Studio agent definition (model + system +
// toolset track). Lets the Studio drive the engine like the demo UI did instead
// of falling back to the default seeded agent. The LLM endpoint stays engine-
// global (runner reads settings.llm_base_url), so only model/system/tools vary.
const MODE_TOOLS: Record<"agent" | "fast", { type: string }[]> = {
  agent: [{ type: "video_agent_20260825" }],
  fast: [{ type: "video_fast_20260825" }],
};

// Stable ids of the engine's built-in default agents (one per mode). The Vanilla
// Studio definition binds to these instead of creating a custom agent.
export const DEFAULT_AGENT_IDS: Record<"agent" | "fast", string> = {
  agent: "ambient_v1",
  fast: "ambient_fast",
};

export function createAgent(cfg: {
  model: string;
  system?: string;
  baseUrl?: string;
  apiKey?: string;
  mode: "agent" | "fast";
  name?: string;
}): Promise<{ id: string }> {
  return json("POST", "/agents", {
    name: cfg.name || `Studio Agent (${cfg.mode})`,
    model: cfg.model,
    system: cfg.system || undefined,
    // Per-agent LLM endpoint override; engine falls back to global when unset.
    base_url: cfg.baseUrl || undefined,
    api_key: cfg.apiKey || undefined,
    tools: MODE_TOOLS[cfg.mode],
  });
}

export function createSession(
  videoId: string,
  mode: "agent" | "fast" = "agent",
  agentId?: string
): Promise<EngineSession> {
  const body: Record<string, unknown> = { metadata: { video_id: videoId, mode } };
  // Engine defaults the agent + environment when omitted; pass `agent` only for
  // a custom Studio definition.
  if (agentId) body.agent = agentId;
  return json("POST", "/sessions", body);
}

// `seq` is the persisted user.message event's position in the session log — the
// start of the run that answers it. Opening a stream at `after_seq = seq - 1`
// replays exactly that run.
export function sendUserMessage(
  sessionId: string,
  text: string
): Promise<{ event_id: string; seq: number; accepted_at: string }> {
  return json("POST", `/sessions/${sessionId}/events`, {
    events: [{ type: "user.message", content: text }],
  });
}

// Mapped Managed-Agents events for a session, paged by `after_seq`. Used to
// rehydrate past-session history (the live path streams via openEventStream).
// Each event carries its raw `seq`.
export function listEvents(
  sessionId: string,
  afterSeq = 0
): Promise<{ data: any[]; last_seq?: number; has_more?: boolean }> {
  return json("GET", `/sessions/${sessionId}/events?after_seq=${afterSeq}`);
}

// Raw SSE stream for one run. Returns the fetch Response so the caller can read
// `response.body` as an event stream.
//   afterSeq    — replay from just after this seq (pass the run's user.message
//                 seq - 1 to get exactly that run). Without it the engine picks
//                 its *latest* run, which can be a different one than intended.
//   lastEventId — resume after a dropped connection: the engine replays events
//                 after that seq and keeps tailing, so a broken engine↔server
//                 connection is stitched back together mid-run.
// The engine closes the stream at the run's run.completed.
export async function openEventStream(
  sessionId: string,
  signal?: AbortSignal,
  lastEventId?: string,
  afterSeq?: number
): Promise<Response> {
  const query = afterSeq && afterSeq > 0 ? `?after_seq=${afterSeq}` : "";
  const res = await fetch(`${ENGINE_URL}/v1/sessions/${sessionId}/events/stream${query}`, {
    headers: headers({
      accept: "text/event-stream",
      ...(lastEventId ? { "last-event-id": lastEventId } : {}),
    }),
    signal,
    cache: "no-store",
  });
  if (!res.ok || !res.body) {
    throw new Error(`engine stream -> ${res.status}`);
  }
  return res;
}

// Server-side proxy for video bytes with Range passthrough (used by the
// /api/ambient/files/[id]/content route so <video> never sees the engine key).
export async function proxyContent(id: string, range: string | null): Promise<Response> {
  return fetch(`${ENGINE_URL}/v1/files/${id}/content`, {
    headers: range ? headers({ range }) : headers(),
    cache: "no-store",
  });
}

// ---- SSE parsing -----------------------------------------------------------
// Yields {event, data, id} frames from a Managed-Agents SSE body. Each frame is
// a named event (`event: <type>`), a JSON `data:`, and an `id:` (the engine seq)
// used to resume via Last-Event-ID after a dropped connection.
export async function* parseSSE(
  body: ReadableStream<Uint8Array>
): AsyncGenerator<{ event: string; data: any; id?: string }> {
  const reader = body.getReader();
  const decoder = new TextDecoder();
  let buf = "";
  for (;;) {
    const { value, done } = await reader.read();
    if (done) break;
    buf += decoder.decode(value, { stream: true });
    // sse_starlette delimits frames with CRLF ("\r\n\r\n"); normalize to "\n"
    // so the "\n\n" split below works regardless of the server's separator.
    // (Without this the split never matches and the stream yields 0 events —
    // this was the Studio's "nothing streams" bug. demo/web does the same.)
    buf = buf.replace(/\r\n?/g, "\n");
    let idx: number;
    // SSE frames are separated by a blank line.
    while ((idx = buf.indexOf("\n\n")) !== -1) {
      const raw = buf.slice(0, idx);
      buf = buf.slice(idx + 2);
      let event = "message";
      let id: string | undefined;
      const dataLines: string[] = [];
      for (const line of raw.split("\n")) {
        if (line.startsWith("event:")) event = line.slice(6).trim();
        else if (line.startsWith("id:")) id = line.slice(3).trim();
        else if (line.startsWith("data:")) dataLines.push(line.slice(5).trim());
      }
      if (!dataLines.length) continue;
      const payload = dataLines.join("\n");
      if (payload === "[DONE]") return;
      try {
        yield { event, data: JSON.parse(payload), id };
      } catch {
        /* skip non-JSON keepalives */
      }
    }
  }
}

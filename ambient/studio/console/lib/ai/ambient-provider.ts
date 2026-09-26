// Ambient provider: a LanguageModelV2 (@ai-sdk/provider v4 / ai v7) whose
// doStream drives the Ambient engine (Managed Agents) and translates engine SSE
// events into AI-SDK stream parts. This is what lets vercel/ai-chatbot talk to
// our engine "as if it were a model" — see docs/studio-plan.md.
//
// The video/session context is threaded from the chat route via providerOptions:
//   streamText({ model: getLanguageModel(id), providerOptions: { ambient: {
//     videoId, mode, sessionId } }, ... })
// The route owns creating/reusing an engine session per chat and passing its id.
//
// Streaming: we read the engine's SSE stream server-side (undici) exactly the
// way demo/web reads it client-side. The engine anchors the stream's replay on
// the latest `user.message`, so the required order is send-message-THEN-open-
// stream (see stream_events() in server/routes/managed_agents.py). Opening the
// stream before the message is persisted makes the engine replay the *previous*
// run and close immediately — that was the "empty stream" bug, not undici.

import type {
  LanguageModelV2,
  LanguageModelV2CallOptions,
  LanguageModelV2StreamPart,
} from "@ai-sdk/provider";
import {
  createSession,
  openEventStream,
  parseSSE,
  sendUserMessage,
} from "@/lib/ambient/engine";
import { setTurnAnchor } from "@/lib/ambient/session-map";

type AmbientOpts = {
  videoId?: string;
  mode?: "agent" | "fast";
  sessionId?: string;
  // The Studio chat + user message this turn answers. When set, the provider
  // records a turn anchor (see lib/ambient/session-map.ts) so a later resume can
  // re-attach to exactly this run.
  chatId?: string;
  userMessageId?: string;
  // Resume mode: don't create a session or send a message — just (re)attach to
  // an existing run's engine stream and replay it from the start, then tail to
  // completion. Used by the /api/chat/[id]/stream resume endpoint.
  resumeOnly?: boolean;
  // Resume only: replay the run that starts after this engine seq (from the turn
  // anchor). Without it the engine falls back to its latest run.
  afterSeq?: number;
};

// Carried on the finish part's providerMetadata.ambient and copied onto the
// assistant message's metadata by the chat routes.
//   runComplete — the turn reached a terminal state that will never change: the
//     engine run finished, or the message was never accepted (so there is no
//     run). False means the stream was cut mid-run and the turn can be resumed.
export type AmbientFinishMetadata = {
  runComplete: boolean;
  usage?: Record<string, unknown>;
};

// Pull the latest user text out of the AI-SDK model prompt.
function latestUserText(prompt: LanguageModelV2CallOptions["prompt"]): string {
  for (let i = prompt.length - 1; i >= 0; i--) {
    const m = prompt[i];
    if (m.role !== "user") continue;
    if (typeof m.content === "string") return m.content;
    return m.content
      .filter((p: any) => p.type === "text")
      .map((p: any) => p.text)
      .join("\n");
  }
  return "";
}

// Managed-Agents events carry text as `[{type:"text", text}]` blocks.
function textOf(blocks: any): string {
  if (Array.isArray(blocks)) return blocks.filter((b) => b?.type === "text").map((b) => b.text).join("");
  return String(blocks ?? "");
}

export function createAmbientModel(modelId: string): LanguageModelV2 {
  return {
    specificationVersion: "v2",
    provider: "ambient",
    modelId,
    // The engine reads media itself; no URL fetching happens in the model.
    supportedUrls: {},

    // Non-streaming path. The engine agent loop is streaming-only, so we don't
    // route doGenerate to it — it's used by ai-chatbot only for chat-title
    // generation (a plain summarization of the first user message). Derive a
    // short title locally instead of calling the engine/LLM.
    async doGenerate(options: LanguageModelV2CallOptions) {
      const raw = latestUserText(options.prompt).replace(/\s+/g, " ").trim();
      const title = raw.length > 70 ? `${raw.slice(0, 70)}…` : raw || "New chat";
      return {
        content: [{ type: "text" as const, text: title }],
        finishReason: "stop" as const,
        usage: { inputTokens: 0, outputTokens: 0, totalTokens: 0 },
        warnings: [],
      };
    },

    async doStream(options: LanguageModelV2CallOptions) {
      const ambient = (options.providerOptions?.ambient ?? {}) as AmbientOpts;
      const text = latestUserText(options.prompt);

      const stream = new ReadableStream<LanguageModelV2StreamPart>({
        async start(controller) {
          const emit = (p: LanguageModelV2StreamPart) => {
            try {
              controller.enqueue(p);
            } catch {
              /* consumer already gone (client disconnected) */
            }
          };
          const close = () => {
            try {
              controller.close();
            } catch {
              /* already closed / cancelled */
            }
          };
          emit({ type: "stream-start", warnings: [] });

          const abort = options.abortSignal;
          const usage = { inputTokens: 0, outputTokens: 0 };
          // Cumulative run usage (tokens + cost + by-source) from the engine's
          // session.status_idle — surfaced to the UI as a usage card.
          let sessionUsage: Record<string, unknown> | null = null;

          // --- stream-part lifecycle -------------------------------------
          // The engine loop interleaves reasoning / tool calls / answer over
          // several steps. Each reasoning segment is its own block (fresh id
          // per open) and closes at every step boundary (a tool call or the
          // answer), so the UI renders one "thinking" chunk per step instead
          // of one giant merged block.
          let reasoningId: string | null = null;
          let reasoningSeq = 0;
          let sawReasoningDelta = false; // dedupe streamed deltas vs whole
          let textId: string | null = null;
          let textSeq = 0;
          let sawMessageDelta = false;
          // Engine tool_use_id -> tool name, so a later tool_result can name
          // its tool part. Tool calls surface as AI-SDK *dynamic* provider-
          // executed tools (the engine runs them) — real tool chips, not text.
          const toolNames = new Map<string, string>();

          // Reasoning, answer text and tool calls are mutually-exclusive "open"
          // blocks: opening one closes the others so every block is anchored at
          // its own chronological position. Each answer segment gets a FRESH id
          // (like reasoning) — a fixed id would anchor all answer text at its
          // first appearance, so later tool calls / thinking would render *after*
          // the final answer instead of before it.
          const openReasoning = () => {
            if (reasoningId === null) {
              closeText();
              reasoningId = `reasoning-${reasoningSeq++}`;
              emit({ type: "reasoning-start", id: reasoningId });
            }
            return reasoningId;
          };
          const closeReasoning = () => {
            if (reasoningId !== null) {
              emit({ type: "reasoning-end", id: reasoningId });
              reasoningId = null;
            }
            // NB: `sawReasoningDelta` is deliberately NOT reset here. The engine
            // emits the aggregated `agent.thinking` at the *end* of a step —
            // after the answer deltas have already closed the block. Keeping the
            // flag set lets us suppress that trailing duplicate instead of
            // rendering a second "thinking" block after the answer. It's reset
            // only at a real step boundary (a tool call).
          };
          const reasonDelta = (delta: string) => {
            if (!delta) return;
            emit({ type: "reasoning-delta", id: openReasoning(), delta });
          };
          const openText = () => {
            if (textId === null) {
              closeReasoning();
              textId = `answer-${textSeq++}`;
              emit({ type: "text-start", id: textId });
            }
            return textId;
          };
          const closeText = () => {
            if (textId !== null) {
              emit({ type: "text-end", id: textId });
              textId = null;
            }
          };

          // Did the engine accept this turn's message? Until it has, there is no
          // run — so a failure before that point is terminal, not resumable.
          let accepted = Boolean(ambient.resumeOnly);
          const anchorFor =
            ambient.chatId && ambient.userMessageId
              ? { chatId: ambient.chatId, messageId: ambient.userMessageId }
              : null;

          try {
            // 1) Resolve the engine session (route usually supplies one).
            let sessionId = ambient.sessionId;
            if (!sessionId) {
              if (ambient.resumeOnly) {
                throw new Error("resume: no engine session for this chat");
              }
              if (!ambient.videoId) {
                throw new Error("no video selected (ambient.videoId missing)");
              }
              const s = await createSession(ambient.videoId, ambient.mode ?? "agent");
              sessionId = s.id;
            }

            // 2) Send the user message FIRST (persisted synchronously) and bind
            //    this turn to the run it starts: the stream below — and any later
            //    resume, via the turn anchor — replays from just before that
            //    message's seq, so it can only ever show *this* run. Skipped on
            //    resume — the run is already in flight (or done).
            let runAfterSeq = ambient.afterSeq;
            if (!ambient.resumeOnly) {
              const turn = anchorFor
                ? { messageId: anchorFor.messageId, sessionId }
                : null;
              if (anchorFor && turn) {
                await setTurnAnchor(anchorFor.chatId, { ...turn, state: "pending" });
              }
              try {
                const sent = await sendUserMessage(sessionId, text);
                runAfterSeq = Math.max(0, Number(sent.seq) - 1);
              } catch (err) {
                if (anchorFor && turn) {
                  await setTurnAnchor(anchorFor.chatId, { ...turn, state: "failed" });
                }
                throw err;
              }
              accepted = true;
              if (anchorFor && turn) {
                await setTurnAnchor(anchorFor.chatId, {
                  ...turn,
                  state: "sent",
                  afterSeq: runAfterSeq,
                });
              }
            }

            // 3) Open the SSE stream and translate events live. Same wire the
            //    demo UI consumes; the engine closes it on run.completed. On
            //    resume this replays the whole run then tails it to completion.
            //
            //    Wrapped in a reconnect loop: if the engine↔server connection
            //    drops before the run finishes, reopen from the last seen event
            //    id (the engine replays after it and keeps tailing) so a transient
            //    network blip never ends the turn early. Each block's open/close
            //    state persists across a reconnect (they live outside this loop).
            let sawIdle = false;
            let lastEventId: string | undefined;
            let reconnects = 0;
            const MAX_RECONNECTS = 30;
            while (!abort?.aborted && !sawIdle) {
              const resp = await openEventStream(
                sessionId,
                abort ?? undefined,
                lastEventId,
                runAfterSeq
              );
              if (!resp.body) throw new Error("engine stream had no body");

              for await (const { event, data, id } of parseSSE(resp.body)) {
                if (abort?.aborted) break;
                if (id) lastEventId = id;
                switch (event) {
                case "agent.reasoning.delta": {
                  // Token-by-token reasoning (from chat.completion.chunk).
                  sawReasoningDelta = true;
                  reasonDelta(String(data.reasoning ?? ""));
                  break;
                }
                case "agent.thinking": {
                  // Whole reasoning for one step. If we already streamed this
                  // step's reasoning via deltas, just close the block; otherwise
                  // this text *is* the step's reasoning — its own block.
                  if (sawReasoningDelta) {
                    closeReasoning();
                  } else {
                    closeReasoning();
                    reasonDelta(textOf(data.content));
                    closeReasoning();
                  }
                  break;
                }
                case "agent.tool_use": {
                  // A step boundary: close the current thinking block, then emit
                  // a real (dynamic, provider-executed) tool call. `dynamic:true`
                  // + `providerExecuted:true` route it through the SDK's dynamic-
                  // tool path so it needs no registered tool and isn't executed
                  // client-side — the engine already ran it. (These fields aren't
                  // in the V2 stream-part type but the runtime reads them.)
                  closeReasoning();
                  closeText(); // a tool call ends any answer text before it
                  sawReasoningDelta = false; // next step's reasoning is fresh
                  const toolCallId = String(data.id ?? `tool-${toolNames.size}`);
                  const toolName = String(data.name ?? "tool");
                  toolNames.set(toolCallId, toolName);
                  const input = { ...(data.input ?? {}) };
                  emit({ type: "tool-input-start", id: toolCallId, toolName, dynamic: true } as any);
                  emit({
                    type: "tool-call",
                    toolCallId,
                    toolName,
                    input: JSON.stringify(input),
                    providerExecuted: true,
                    dynamic: true,
                  } as any);
                  break;
                }
                case "agent.tool_result": {
                  const toolCallId = String(data.tool_use_id ?? "");
                  const toolName = toolNames.get(toolCallId) ?? "tool";
                  // Emit the analysis as a plain STRING (not an object) so the UI
                  // renders it as readable preformatted text — an object would be
                  // JSON.stringify'd, escaping every newline into a literal "\n".
                  const result = textOf(data.content);
                  emit({
                    type: "tool-result",
                    toolCallId,
                    toolName,
                    result,
                    isError: Boolean(data.is_error),
                    providerExecuted: true,
                    dynamic: true,
                  } as any);
                  break;
                }
                case "agent.message.delta": {
                  // Live answer tokens — close reasoning and stream into answer.
                  sawMessageDelta = true;
                  closeReasoning();
                  const d = textOf(data.content);
                  if (d) { emit({ type: "text-delta", id: openText(), delta: d }); }
                  break;
                }
                case "agent.message": {
                  // Whole answer (run.completed). Only emit if we didn't already
                  // stream it via deltas.
                  closeReasoning();
                  if (!sawMessageDelta) {
                    const answer = textOf(data.content);
                    if (answer) { emit({ type: "text-delta", id: openText(), delta: answer }); }
                  }
                  break;
                }
                case "span.model_request_end": {
                  const u = data.model_usage ?? {};
                  usage.inputTokens += Number(u.input_tokens ?? 0);
                  usage.outputTokens += Number(u.output_tokens ?? 0);
                  break;
                }
                case "session.error": {
                  // Surface as visible text + a normal finish (an AI-SDK error
                  // part leaves useChat spinning with no finish).
                  closeReasoning();
                  emit({
                    type: "text-delta",
                    id: openText(),
                    delta: `\n\n⚠️ Engine error: ${String(data.error?.message ?? "unknown error")}`,
                  });
                  break;
                }
                case "session.status_idle":
                  // Terminal for the turn; the engine closes the stream next.
                  if (data.usage && typeof data.usage === "object") {
                    sessionUsage = data.usage as Record<string, unknown>;
                  }
                  sawIdle = true;
                  break;
                default:
                  break;
              }
            }

            // The stream ended. If the run finished (or we were aborted), stop.
            // Otherwise the engine connection dropped mid-run — reconnect from the
            // last event id and keep going.
            if (sawIdle || abort?.aborted) {
              break;
            }
            reconnects += 1;
            if (reconnects > MAX_RECONNECTS) {
              break;
            }
            await new Promise((r) => setTimeout(r, 500));
            }

            closeReasoning();
            closeText();
            emit({
              type: "finish",
              finishReason: "stop",
              usage: {
                inputTokens: usage.inputTokens,
                outputTokens: usage.outputTokens,
                totalTokens: usage.inputTokens + usage.outputTokens,
              },
              // Whether the run finished (vs. the stream being cut mid-run), plus
              // the engine's cumulative usage for the usage card.
              providerMetadata: {
                ambient: {
                  runComplete: sawIdle,
                  ...(sessionUsage ? { usage: sessionUsage } : {}),
                } satisfies AmbientFinishMetadata,
              },
            } as LanguageModelV2StreamPart);
            close();
          } catch (err) {
            closeReasoning();
            if (abort?.aborted) {
              // The client went away (reload / navigation). Nothing to show; the
              // engine run carries on independently and a resume picks it up.
              closeText();
            } else {
              // Same rationale as session.error: show it, then finish cleanly so
              // the UI doesn't hang on a missing finish.
              const msg = err instanceof Error ? err.message : String(err);
              emit({ type: "text-delta", id: openText(), delta: `⚠️ ${msg}` });
              closeText();
            }
            emit({
              type: "finish",
              finishReason: "error",
              usage: { inputTokens: 0, outputTokens: 0, totalTokens: 0 },
              // A failure before the engine accepted the message is final (there
              // is no run to resume); after that, the run may still complete.
              providerMetadata: {
                ambient: { runComplete: !accepted } satisfies AmbientFinishMetadata,
              },
            } as LanguageModelV2StreamPart);
            close();
          }
        },
      });

      return { stream };
    },
  };
}

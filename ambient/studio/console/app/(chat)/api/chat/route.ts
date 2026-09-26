import { geolocation, ipAddress } from "@vercel/functions";
import {
  convertToModelMessages,
  createUIMessageStream,
  createUIMessageStreamResponse,
  generateId,
  isStepCount,
  streamText,
  toUIMessageStream,
} from "ai";
import { checkBotId } from "botid/server";
import { after } from "next/server";
import { createResumableStreamContext } from "resumable-stream";
import { auth, type UserType } from "@/app/(auth)/auth";
import { entitlementsByUserType } from "@/lib/ai/entitlements";
import {
  allowedModelIds,
  chatModels,
  DEFAULT_CHAT_MODEL,
  getCapabilities,
  getModelAvailability,
} from "@/lib/ai/models";
import { type RequestHints, systemPrompt } from "@/lib/ai/prompts";
import { getLanguageModel } from "@/lib/ai/providers";
import { createDocument } from "@/lib/ai/tools/create-document";
import { editDocument } from "@/lib/ai/tools/edit-document";
import { getWeather } from "@/lib/ai/tools/get-weather";
import { requestSuggestions } from "@/lib/ai/tools/request-suggestions";
import { updateDocument } from "@/lib/ai/tools/update-document";
import { isProductionEnvironment } from "@/lib/constants";
import {
  createStreamId,
  deleteChatById,
  getChatById,
  getMessageCountByUserId,
  getMessagesByChatId,
  saveChat,
  saveMessages,
  saveTurnAssistant,
  updateChatTitleById,
  updateMessage,
} from "@/lib/db/queries";
import type { DBMessage } from "@/lib/db/schema";
import { ChatbotError } from "@/lib/errors";
import { checkIpRateLimit } from "@/lib/ratelimit";
import type { ChatMessage, WaitingStatusData } from "@/lib/types";
import { convertToUIMessages, generateUUID } from "@/lib/utils";
import { generateTitleFromUserMessage } from "../../actions";
import {
  createAgent as createEngineAgent,
  createSession as createEngineSession,
  DEFAULT_AGENT_IDS,
} from "@/lib/ambient/engine";
import { getChatSessionId, setChatSession } from "@/lib/ambient/session-map";
import { resolveChatVideoId } from "@/lib/ambient/chat-video";
import { ambientMessageMetadata } from "@/lib/ambient/turn";
import { type PostRequestBody, postRequestBodySchema } from "./schema";

// chatId -> engine sessionId. In-memory (dev); survives the server process.
const engineSessionForChat = new Map<string, string>();
// Agent-definition signature (model|system|mode) -> engine agent id, so a custom
// Studio agent definition is created on the engine once and reused across chats.
const engineAgentForDef = new Map<string, string>();

type AgentDefInput = {
  model?: string;
  system?: string;
  baseUrl?: string;
  apiKey?: string;
};

async function resolveEngineAgentId(
  def: AgentDefInput | undefined,
  mode: "agent" | "fast"
): Promise<string> {
  const overrides = !!(
    def?.model ||
    def?.system ||
    def?.baseUrl ||
    def?.apiKey
  );
  // Vanilla (no overrides) => the engine's built-in default agent for the mode.
  if (!overrides) {
    return DEFAULT_AGENT_IDS[mode];
  }
  // A custom definition => create (and reuse) an engine agent for this exact
  // config + mode. Tools come from the mode, endpoint/model/system from the def.
  const sig = [
    def?.model ?? "",
    def?.system ?? "",
    def?.baseUrl ?? "",
    def?.apiKey ?? "",
    mode,
  ].join("||");
  const cached = engineAgentForDef.get(sig);
  if (cached) {
    return cached;
  }
  const agent = await createEngineAgent({
    model: def?.model || "",
    system: def?.system,
    baseUrl: def?.baseUrl,
    apiKey: def?.apiKey,
    mode,
  });
  engineAgentForDef.set(sig, agent.id);
  return agent.id;
}

// Ambient agent runs (multi-round tool loops + reasoning) routinely exceed a
// minute — clip analysis alone is tens of seconds each. Cap at 1 hour so the
// HTTP stream isn't cut mid-run (the engine run is independent and completes,
// but a cut stream freezes the UI on the last event it received).
export const maxDuration = 3600;

const HEALTH_CHECK_DELAY_MS = 9000;

function isModelStreamActivity(chunk: { type: string }) {
  return !["start", "start-step", "finish-step", "finish", "raw"].includes(
    chunk.type
  );
}

function getStreamContext() {
  try {
    return createResumableStreamContext({ waitUntil: after });
  } catch {
    return null;
  }
}

export { getStreamContext };

export async function POST(request: Request) {
  let requestBody: PostRequestBody;

  try {
    const json = await request.json();
    requestBody = postRequestBodySchema.parse(json);
  } catch {
    return new ChatbotError("bad_request:api").toResponse();
  }

  try {
    const {
      id,
      message,
      messages,
      selectedChatModel,
      selectedVisibilityType,
      videoId,
      ambientMode: ambientModeInput,
      agentDef,
    } = requestBody;

    // Mode is the global top-center toggle now (not part of the agent def).
    const ambientMode: "agent" | "fast" =
      ambientModeInput ?? (selectedChatModel === "fast" ? "fast" : "agent");

    // The video this turn is about. An existing chat is bound to its own video
    // (its engine session was created for it) — use that, not whatever the
    // client currently has selected, so a follow-up can never be answered about
    // a different video than the chat's. A new chat takes the selected video.
    const existingChat = await getChatById({ id });
    const turnVideoId =
      (existingChat ? await resolveChatVideoId(existingChat) : null) ??
      videoId ??
      undefined;

    // Resolve (once per chat) the engine session this chat maps to, so follow-up
    // questions keep the agent's context. The in-memory map is a cache; the
    // Redis pointer is the durable copy — without it, a follow-up after a server
    // restart / HMR module reload / another worker silently started a fresh
    // engine session and lost the conversation.
    let ambientSessionId = turnVideoId
      ? (engineSessionForChat.get(id) ?? (await getChatSessionId(id)) ?? undefined)
      : undefined;
    if (ambientSessionId) {
      engineSessionForChat.set(id, ambientSessionId);
    }
    if (turnVideoId && !ambientSessionId) {
      // Bind the session to a custom engine agent when the active definition
      // sets a model; otherwise the engine uses its default seeded agent.
      const engineAgentId = await resolveEngineAgentId(agentDef, ambientMode);
      const s = await createEngineSession(turnVideoId, ambientMode, engineAgentId);
      ambientSessionId = s.id;
      engineSessionForChat.set(id, ambientSessionId);
    }
    // Persist the chat → engine-session pointer so the resume endpoint can find
    // this run from a different request (reload / Fast-Refresh remount / worker).
    if (ambientSessionId) {
      await setChatSession(id, ambientSessionId);
    }

    const [botIdResult, session] = await Promise.all([
      checkBotId().catch(() => null),
      auth(),
    ]);

    if (botIdResult?.isBot) {
      return new ChatbotError("forbidden:api").toResponse();
    }

    if (!session?.user) {
      return new ChatbotError("unauthorized:chat").toResponse();
    }

    const chatModel = allowedModelIds.has(selectedChatModel)
      ? selectedChatModel
      : DEFAULT_CHAT_MODEL;

    await checkIpRateLimit(ipAddress(request));

    const userType: UserType = session.user.type;

    const messageCount = await getMessageCountByUserId({
      differenceInHours: 1,
      id: session.user.id,
    });

    if (messageCount > entitlementsByUserType[userType].maxMessagesPerHour) {
      return new ChatbotError("rate_limit:chat").toResponse();
    }

    const isToolApprovalFlow = Boolean(messages);

    const chat = existingChat;
    let messagesFromDb: DBMessage[] = [];
    let titlePromise: Promise<string> | null = null;

    if (chat) {
      if (chat.userId !== session.user.id) {
        return new ChatbotError("forbidden:chat").toResponse();
      }
      messagesFromDb = await getMessagesByChatId({ id });
    } else if (message?.role === "user") {
      await saveChat({
        id,
        title: "New chat",
        userId: session.user.id,
        videoId: turnVideoId ?? null,
        visibility: selectedVisibilityType,
      });
      titlePromise = generateTitleFromUserMessage({ message });
    }

    let uiMessages: ChatMessage[];

    if (isToolApprovalFlow && messages) {
      const dbMessages = convertToUIMessages(messagesFromDb);
      const approvalStates = new Map(
        messages.flatMap(
          (m) =>
            m.parts
              ?.filter(
                (p: Record<string, unknown>) =>
                  p.state === "approval-responded" ||
                  p.state === "output-denied"
              )
              .map((p: Record<string, unknown>) => [
                String(p.toolCallId ?? ""),
                p,
              ]) ?? []
        )
      );
      uiMessages = dbMessages.map((msg) => ({
        ...msg,
        parts: msg.parts.map((part) => {
          if (
            "toolCallId" in part &&
            approvalStates.has(String(part.toolCallId))
          ) {
            return { ...part, ...approvalStates.get(String(part.toolCallId)) };
          }
          return part;
        }),
      })) as ChatMessage[];
    } else {
      uiMessages = [
        ...convertToUIMessages(messagesFromDb),
        message as ChatMessage,
      ];
    }

    const { longitude, latitude, city, country } = geolocation(request);

    const requestHints: RequestHints = {
      city,
      country,
      latitude,
      longitude,
    };

    if (message?.role === "user") {
      await saveMessages({
        messages: [
          {
            attachments: [],
            chatId: id,
            createdAt: new Date(),
            id: message.id,
            metadata: null,
            parts: message.parts,
            role: "user",
          },
        ],
      });
    }

    const modelConfig = chatModels.find((m) => m.id === chatModel);
    const modelCapabilities = await getCapabilities();
    const capabilities = modelCapabilities[chatModel];
    const isReasoningModel = capabilities?.reasoning === true;
    const supportsTools = capabilities?.tools === true;

    const modelMessages = await convertToModelMessages(uiMessages);

    // Heartbeat that keeps the browser's SSE connection alive during idle gaps
    // (a running clip tool or a model prefill emits no message chunks for tens of
    // seconds; without bytes the connection is dropped and the run "freezes").
    // Cleared when the stream ends (onEnd/onError below).
    let keepAliveTimer: ReturnType<typeof setInterval> | undefined;
    const clearKeepAlive = () => {
      if (keepAliveTimer) {
        clearInterval(keepAliveTimer);
        keepAliveTimer = undefined;
      }
    };

    const stream = createUIMessageStream({
      execute: async ({ writer: dataStream }) => {
        const modelName = modelConfig?.name ?? chatModel;
        let hasModelActivity = false;
        let healthCheckTimer: ReturnType<typeof setTimeout> | undefined;

        // Emit a transient heartbeat every 10s for the whole turn.
        keepAliveTimer = setInterval(() => {
          try {
            dataStream.write({
              data: Date.now(),
              transient: true,
              type: "data-keepalive",
            });
          } catch {
            clearKeepAlive();
          }
        }, 10_000);

        const clearHealthCheckTimer = () => {
          if (healthCheckTimer) {
            clearTimeout(healthCheckTimer);
          }
        };

        const writeWaitingStatus = (
          phase: WaitingStatusData["phase"],
          messageText: string
        ) => {
          if (hasModelActivity && phase !== "thinking") {
            return;
          }
          dataStream.write({
            data: {
              message: messageText,
              modelId: chatModel,
              modelName,
              phase,
            },
            transient: true,
            type: "data-waiting-status",
          });
        };

        writeWaitingStatus("waiting", "Waiting...");

        healthCheckTimer = setTimeout(() => {
          getModelAvailability(chatModel)
            .then((availability) => {
              if (availability === "impacted") {
                writeWaitingStatus(
                  "health",
                  `${modelName} may be slow or unavailable right now...`
                );
              } else {
                writeWaitingStatus("still-waiting", "Still waiting...");
              }
            })
            .catch(() => {
              writeWaitingStatus("still-waiting", "Still waiting...");
            });
        }, HEALTH_CHECK_DELAY_MS);

        const markModelActive = () => {
          if (hasModelActivity) {
            return;
          }
          hasModelActivity = true;
          clearHealthCheckTimer();
          writeWaitingStatus("thinking", "Thinking...");
        };

        const stopWaitingStatus = () => {
          hasModelActivity = true;
          clearHealthCheckTimer();
        };

        const result = streamText({
          // Stop reading the engine when the client goes away (reload, navigate).
          // The engine run is independent and keeps going; the next page load
          // resumes it. Without this every reload left an orphaned reader tailing
          // the engine until the run ended.
          abortSignal: request.signal,
          activeTools:
            isReasoningModel && !supportsTools
              ? []
              : [
                  "getWeather",
                  "createDocument",
                  "editDocument",
                  "updateDocument",
                  "requestSuggestions",
                ],
          instructions: systemPrompt({ requestHints, supportsTools }),
          messages: modelMessages,
          model: getLanguageModel(chatModel),
          onAbort() {
            stopWaitingStatus();
          },
          onChunk({ chunk }) {
            if (isModelStreamActivity(chunk)) {
              markModelActive();
            }
          },
          onEnd() {
            stopWaitingStatus();
          },
          onError() {
            stopWaitingStatus();
          },
          providerOptions: {
            ...(modelConfig?.gatewayOrder && {
              gateway: { order: modelConfig.gatewayOrder },
            }),
            ...(modelConfig?.reasoningEffort && {
              openai: { reasoningEffort: modelConfig.reasoningEffort },
            }),
            // Ambient provider: video + engine session context, plus this turn's
            // identity so the provider can bind it to its engine run (turn anchor).
            ambient: {
              videoId: turnVideoId ?? null,
              sessionId: ambientSessionId ?? null,
              mode: ambientMode ?? "agent",
              chatId: id,
              userMessageId: message?.role === "user" ? message.id : undefined,
            },
          },
          // Our engine runs its own multi-step tool loop; one provider round-trip.
          stopWhen: isStepCount(1),
          telemetry: {
            functionId: "stream-text",
            isEnabled: isProductionEnvironment,
          },
          tools: {
            createDocument: createDocument({
              dataStream,
              modelId: chatModel,
              session,
            }),
            editDocument: editDocument({ dataStream, session }),
            getWeather,
            requestSuggestions: requestSuggestions({
              dataStream,
              modelId: chatModel,
              session,
            }),
            updateDocument: updateDocument({
              dataStream,
              modelId: chatModel,
              session,
            }),
          },
        });

        dataStream.merge(
          toUIMessageStream({
            sendReasoning: isReasoningModel,
            stream: result.stream,
            // Copy the ambient provider's finish metadata (run completion +
            // cumulative usage for the usage card) onto the message.
            messageMetadata: ambientMessageMetadata,
          })
        );

        if (titlePromise) {
          try {
            const title = await titlePromise;
            dataStream.write({ data: title, type: "data-chat-title" });
            updateChatTitleById({ chatId: id, title });
          } catch {
            /* non-fatal */
          }
        }
      },
      generateId: generateUUID,
      onEnd: async ({ messages: finishedMessages }) => {
        clearKeepAlive();
        if (isToolApprovalFlow) {
          await Promise.all(
            finishedMessages.map(async (finishedMsg) => {
              const existingMsg = uiMessages.find(
                (m) => m.id === finishedMsg.id
              );
              if (existingMsg) {
                await updateMessage({
                  id: finishedMsg.id,
                  parts: finishedMsg.parts,
                });
                return;
              }

              await saveMessages({
                messages: [
                  {
                    attachments: [],
                    chatId: id,
                    createdAt: new Date(),
                    id: finishedMsg.id,
                    metadata: finishedMsg.metadata ?? null,
                    parts: finishedMsg.parts,
                    role: finishedMsg.role,
                  },
                ],
              });
            })
          );
        } else {
          // The answer may be partial (the client left mid-run); it's persisted
          // without `runComplete`, so the next load resumes it from the engine.
          for (const currentMessage of finishedMessages) {
            if (currentMessage.role === "assistant") {
              await saveTurnAssistant({
                chatId: id,
                message: {
                  id: currentMessage.id,
                  metadata: currentMessage.metadata ?? null,
                  parts: currentMessage.parts,
                },
              });
            } else {
              await saveMessages({
                messages: [
                  {
                    attachments: [],
                    chatId: id,
                    createdAt: new Date(),
                    id: currentMessage.id,
                    metadata: currentMessage.metadata ?? null,
                    parts: currentMessage.parts,
                    role: currentMessage.role,
                  },
                ],
              });
            }
          }
        }
      },
      onError: (error) => {
        clearKeepAlive();
        if (
          error instanceof Error &&
          error.message?.includes(
            "AI Gateway requires a valid credit card on file to service requests"
          )
        ) {
          return "AI Gateway requires a valid credit card on file to service requests. Please visit https://vercel.com/d?to=%2F%5Bteam%5D%2F%7E%2Fai%3Fmodal%3Dadd-credit-card to add a card and unlock your free credits.";
        }
        return "Oops, an error occurred!";
      },
      originalMessages: isToolApprovalFlow ? uiMessages : undefined,
    });

    return createUIMessageStreamResponse({
      async consumeSseStream({ stream: sseStream }) {
        if (!process.env.REDIS_URL) {
          return;
        }
        try {
          const streamContext = getStreamContext();
          if (streamContext) {
            const streamId = generateId();
            await createStreamId({ chatId: id, streamId });
            await streamContext.createNewResumableStream(
              streamId,
              () => sseStream
            );
          }
        } catch {
          /* non-critical */
        }
      },
      stream,
    });
  } catch (error) {
    const vercelId = request.headers.get("x-vercel-id");

    if (error instanceof ChatbotError) {
      return error.toResponse();
    }

    if (
      error instanceof Error &&
      error.message?.includes(
        "AI Gateway requires a valid credit card on file to service requests"
      )
    ) {
      return new ChatbotError("bad_request:activate_gateway").toResponse();
    }

    console.error("Unhandled error in chat API:", error, { vercelId });
    return new ChatbotError("offline:chat").toResponse();
  }
}

export async function DELETE(request: Request) {
  const { searchParams } = new URL(request.url);
  const id = searchParams.get("id");

  if (!id) {
    return new ChatbotError("bad_request:api").toResponse();
  }

  const session = await auth();

  if (!session?.user) {
    return new ChatbotError("unauthorized:chat").toResponse();
  }

  const chat = await getChatById({ id });

  if (chat?.userId !== session.user.id) {
    return new ChatbotError("forbidden:chat").toResponse();
  }

  const deletedChat = await deleteChatById({ id });

  return Response.json(deletedChat, { status: 200 });
}

import {
  createUIMessageStream,
  createUIMessageStreamResponse,
  isStepCount,
  streamText,
  toUIMessageStream,
} from "ai";
import { auth } from "@/app/(auth)/auth";
import { getLanguageModel } from "@/lib/ai/providers";
import { waitForTurnAnchor } from "@/lib/ambient/session-map";
import { ambientMessageMetadata, lastUserIndex } from "@/lib/ambient/turn";
import {
  getChatById,
  getMessagesByChatId,
  saveTurnAssistant,
} from "@/lib/db/queries";
import { ChatbotError } from "@/lib/errors";
import type { ChatMessage } from "@/lib/types";
import { generateUUID } from "@/lib/utils";

// Resume endpoint for an in-progress (or just-finished) assistant turn.
//
// The AI SDK's DefaultChatTransport.reconnectToStream GETs this URL
// (`/api/chat/{chatId}/stream`) whenever the client needs to (re)attach to a
// run — on page load with a pending turn, and after a dev Fast-Refresh remount
// or a network drop cuts the live stream.
//
// It resumes by RE-READING THE ENGINE, which is the durable source of truth: the
// engine run is independent of any HTTP request and persists every event with
// replay support. The run to replay comes from the chat's turn anchor — the
// exact engine run its latest user message started — never the engine's
// "latest run", which can belong to a different turn (e.g. when the latest
// message was rejected). The replay covers the whole run and then tails it to
// completion, so the client catches up on everything it missed.
export async function GET(
  request: Request,
  { params }: { params: Promise<{ id: string }> }
) {
  const { id: chatId } = await params;
  if (!chatId) {
    return new ChatbotError("bad_request:api").toResponse();
  }

  const session = await auth();
  if (!session?.user) {
    return new ChatbotError("unauthorized:chat").toResponse();
  }

  const chat = await getChatById({ id: chatId });
  if (!chat) {
    // Chat row not created yet (very early in the run) — nothing to resume.
    return new Response(null, { status: 204 });
  }
  if (chat.visibility === "private" && chat.userId !== session.user.id) {
    return new ChatbotError("forbidden:chat").toResponse();
  }

  // Only the latest turn is ever resumed, and only when we know which engine run
  // answers it. A `pending` anchor (the original request is mid-send) is waited
  // out rather than guessed at.
  const history = await getMessagesByChatId({ id: chatId });
  const userIdx = lastUserIndex(history);
  if (userIdx < 0) {
    return new Response(null, { status: 204 });
  }
  const anchor = await waitForTurnAnchor(chatId, history[userIdx].id);
  if (anchor?.state !== "sent" || anchor.afterSeq === undefined) {
    return new Response(null, { status: 204 });
  }

  const stream = createUIMessageStream({
    execute: ({ writer }) => {
      const result = streamText({
        // Stop reading the engine if this client goes away too; the run carries
        // on and the next load resumes it again.
        abortSignal: request.signal,
        model: getLanguageModel("agent"),
        // The prompt is ignored in resumeOnly mode (the provider reads the
        // engine, not the messages), but streamText requires one.
        messages: [{ role: "user", content: "resume" }],
        providerOptions: {
          ambient: {
            sessionId: anchor.sessionId,
            afterSeq: anchor.afterSeq,
            resumeOnly: true,
          },
        },
        stopWhen: isStepCount(1),
      });
      writer.merge(
        toUIMessageStream({
          messageMetadata: ambientMessageMetadata,
          sendReasoning: true,
          stream: result.stream,
        })
      );
    },
    generateId: generateUUID,
    onError: () => "The run could not be resumed.",
    // Persist the resumed answer as this turn's answer. It may itself be cut
    // (another reload); then it's saved without `runComplete` and the next load
    // resumes again — and a partial never replaces a complete answer.
    onEnd: async ({ messages }) => {
      const assistant = (messages as ChatMessage[])
        .filter((m) => m.role === "assistant")
        .at(-1);
      if (!assistant) {
        return;
      }
      try {
        await saveTurnAssistant({
          chatId,
          message: {
            id: assistant.id,
            metadata: assistant.metadata ?? null,
            parts: assistant.parts,
          },
        });
      } catch {
        /* best-effort persistence */
      }
    },
  });

  return createUIMessageStreamResponse({ stream });
}

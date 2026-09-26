import { auth } from "@/app/(auth)/auth";
import { resolveChatVideoId } from "@/lib/ambient/chat-video";
import { getTurnAnchor } from "@/lib/ambient/session-map";
import { prepareMessagesForLoad } from "@/lib/ambient/turn";
import { getChatById, getMessagesByChatId } from "@/lib/db/queries";
import { convertToUIMessages } from "@/lib/utils";

export async function GET(request: Request) {
  const { searchParams } = new URL(request.url);
  const chatId = searchParams.get("chatId");

  if (!chatId) {
    return Response.json({ error: "chatId required" }, { status: 400 });
  }

  const [session, chat, messages] = await Promise.all([
    auth(),
    getChatById({ id: chatId }),
    getMessagesByChatId({ id: chatId }),
  ]);

  if (!chat) {
    return Response.json({
      isReadonly: false,
      messages: [],
      userId: null,
      visibility: "private",
    });
  }

  if (
    chat.visibility === "private" &&
    (!session?.user || session.user.id !== chat.userId)
  ) {
    return Response.json({ error: "forbidden" }, { status: 403 });
  }

  const isReadonly = !session?.user || session.user.id !== chat.userId;
  const uiMessages = convertToUIMessages(messages);

  // If the latest turn's answer was cut mid-run (reload / navigation / network
  // drop while the engine kept going), hand the owner the history *without* that
  // partial answer: the client sees a trailing user message and resumes the turn
  // from the engine — complete if the run finished, still streaming if it's
  // running. Otherwise a running session loads looking finished and never
  // completes. (See prepareMessagesForLoad.)
  const forClient = isReadonly
    ? uiMessages
    : prepareMessagesForLoad(uiMessages, await getTurnAnchor(chatId)).messages;

  return Response.json({
    isReadonly,
    messages: forClient,
    userId: chat.userId,
    // The video this chat is about — the client switches the player to it so a
    // session page always shows its own video, not the last one selected.
    videoId: await resolveChatVideoId(chat),
    visibility: chat.visibility,
  });
}

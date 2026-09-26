import "server-only";

import { getSessionVideoId } from "@/lib/ambient/engine";
import { getChatSessionId } from "@/lib/ambient/session-map";
import { setChatVideoIdIfMissing } from "@/lib/db/queries";

// The video a chat is about. A chat maps to one engine session and a session is
// bound to one video, so this is fixed for the chat's lifetime.
//
// Chats record it at creation (Chat.videoId). Chats created before that column
// existed fall back to their engine session's video — found through the Redis
// chat→session pointer — and get it backfilled so later loads skip the engine.
// Returns null when neither is known (e.g. an old chat whose pointer expired).
export async function resolveChatVideoId(chat: {
  id: string;
  videoId: string | null;
}): Promise<string | null> {
  if (chat.videoId) {
    return chat.videoId;
  }
  try {
    const sessionId = await getChatSessionId(chat.id);
    if (!sessionId) {
      return null;
    }
    const videoId = await getSessionVideoId(sessionId);
    if (videoId) {
      await setChatVideoIdIfMissing({ id: chat.id, videoId });
    }
    return videoId;
  } catch {
    return null; // best-effort: the page still loads, just without switching video
  }
}

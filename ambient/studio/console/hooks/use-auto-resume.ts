"use client";

import type { UseChatHelpers } from "@ai-sdk/react";
import { useEffect, useRef } from "react";
import { useDataStream } from "@/components/chat/data-stream-provider";
import type { ChatMessage } from "@/lib/types";

export type UseAutoResumeParams = {
  autoResume: boolean;
  chatId: string;
  initialMessages: ChatMessage[];
  resumeStream: UseChatHelpers<ChatMessage>["resumeStream"];
  setMessages: UseChatHelpers<ChatMessage>["setMessages"];
  status: UseChatHelpers<ChatMessage>["status"];
};

// Re-attach to the chat's latest turn on load when it hasn't finished. The
// server hands back history ending in a user message for such a turn — either
// its answer was never saved, or it was cut mid-run and dropped (see
// prepareMessagesForLoad) — and the resume endpoint replays it from the engine.
export function useAutoResume({
  autoResume,
  chatId,
  initialMessages,
  resumeStream,
  setMessages,
  status,
}: UseAutoResumeParams) {
  const { dataStream } = useDataStream();
  const statusRef = useRef(status);
  statusRef.current = status;
  // One attempt per chat load. The effect re-runs whenever the fresh message
  // list arrives; without this it could resume the same turn twice.
  const attemptedForRef = useRef<string | null>(null);

  useEffect(() => {
    if (!autoResume || attemptedForRef.current === chatId) {
      return;
    }
    attemptedForRef.current = chatId;

    // A request is already streaming this chat's turn — e.g. the page just
    // started it and the /chat/:id URL swap loaded the chat's history mid-run.
    // A second, concurrent resume would replay the same run into the same
    // message.
    if (statusRef.current !== "ready") {
      return;
    }

    if (initialMessages.at(-1)?.role === "user") {
      resumeStream();
    }
  }, [autoResume, chatId, initialMessages, resumeStream]);

  useEffect(() => {
    if (!dataStream) {
      return;
    }
    if (dataStream.length === 0) {
      return;
    }

    const [dataPart] = dataStream;

    if (dataPart.type === "data-appendMessage") {
      const message = JSON.parse(dataPart.data);
      setMessages([...initialMessages, message]);
    }
  }, [dataStream, initialMessages, setMessages]);
}

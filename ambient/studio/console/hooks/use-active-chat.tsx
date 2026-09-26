"use client";

import type { UseChatHelpers } from "@ai-sdk/react";
import { useChat } from "@ai-sdk/react";
import { DefaultChatTransport } from "ai";
import { usePathname } from "next/navigation";
import {
  createContext,
  type Dispatch,
  type ReactNode,
  type SetStateAction,
  useContext,
  useEffect,
  useMemo,
  useRef,
  useState,
} from "react";
import useSWR, { useSWRConfig } from "swr";
import { unstable_serialize } from "swr/infinite";
import { useDataStream } from "@/components/chat/data-stream-provider";
import { getActiveAgentDef } from "@/lib/ai/agent-defs";
import { getMode } from "@/lib/ai/mode";
import { getChatHistoryPaginationKey } from "@/components/chat/sidebar-history";
import { toast } from "@/components/chat/toast";
import type { VisibilityType } from "@/components/chat/visibility-selector";
import { useAutoResume } from "@/hooks/use-auto-resume";
import { setSelectedVideo } from "@/hooks/use-selected-video";
import { DEFAULT_CHAT_MODEL } from "@/lib/ai/models";
import type { Vote } from "@/lib/db/schema";
import { ChatbotError } from "@/lib/errors";
import type { ChatMessage } from "@/lib/types";
import { fetcher, fetchWithErrorHandlers, generateUUID } from "@/lib/utils";

type ActiveChatContextValue = {
  chatId: string;
  messages: ChatMessage[];
  setMessages: UseChatHelpers<ChatMessage>["setMessages"];
  sendMessage: UseChatHelpers<ChatMessage>["sendMessage"];
  status: UseChatHelpers<ChatMessage>["status"];
  stop: UseChatHelpers<ChatMessage>["stop"];
  regenerate: UseChatHelpers<ChatMessage>["regenerate"];
  addToolApprovalResponse: UseChatHelpers<ChatMessage>["addToolApprovalResponse"];
  input: string;
  setInput: Dispatch<SetStateAction<string>>;
  visibilityType: VisibilityType;
  isReadonly: boolean;
  isLoading: boolean;
  votes: Vote[] | undefined;
  currentModelId: string;
  setCurrentModelId: (id: string) => void;
  // The video the open chat is about (null for a new chat, or an old chat whose
  // video can't be resolved).
  chatVideoId: string | null;
  showCreditCardAlert: boolean;
  setShowCreditCardAlert: Dispatch<SetStateAction<boolean>>;
};

const ActiveChatContext = createContext<ActiveChatContextValue | null>(null);

function extractChatId(pathname: string): string | null {
  const match = pathname.match(/\/chat\/([^/]+)/);
  return match ? match[1] : null;
}

export function ActiveChatProvider({ children }: { children: ReactNode }) {
  const pathname = usePathname();
  const { setDataStream, setWaitingStatus } = useDataStream();
  const { mutate } = useSWRConfig();

  const chatIdFromUrl = extractChatId(pathname);
  const isNewChat = !chatIdFromUrl;
  const newChatIdRef = useRef(generateUUID());
  const prevPathnameRef = useRef(pathname);

  if (isNewChat && prevPathnameRef.current !== pathname) {
    newChatIdRef.current = generateUUID();
  }
  prevPathnameRef.current = pathname;

  const chatId = chatIdFromUrl ?? newChatIdRef.current;

  const [currentModelId, setCurrentModelId] = useState(DEFAULT_CHAT_MODEL);
  const currentModelIdRef = useRef(currentModelId);
  useEffect(() => {
    currentModelIdRef.current = currentModelId;
  }, [currentModelId]);

  const [input, setInput] = useState("");
  const [showCreditCardAlert, setShowCreditCardAlert] = useState(false);

  const { data: chatData, isLoading, isValidating } = useSWR(
    isNewChat
      ? null
      : `${process.env.NEXT_PUBLIC_BASE_PATH ?? ""}/api/messages?chatId=${chatId}`,
    fetcher,
    { revalidateOnFocus: false }
  );

  const initialMessages: ChatMessage[] = isNewChat
    ? []
    : (chatData?.messages ?? []);
  const visibility: VisibilityType = isNewChat
    ? "private"
    : (chatData?.visibility ?? "private");
  const chatVideoId: string | null = isNewChat
    ? null
    : (chatData?.videoId ?? null);

  // A session page shows its own video. The selection is global (the player,
  // the selector and the new-chat transport all read it), so switch it to this
  // chat's video whenever a chat is opened — otherwise the page kept showing
  // whichever video was selected last. A chat's video never changes, so even a
  // stale cached copy of the chat is safe to act on.
  useEffect(() => {
    if (chatVideoId) {
      setSelectedVideo(chatVideoId);
    }
  }, [chatId, chatVideoId]);

  const {
    messages,
    setMessages,
    sendMessage,
    status,
    stop,
    regenerate,
    resumeStream,
    addToolApprovalResponse,
  } = useChat<ChatMessage>({
    // Batch message re-renders. A resume replays a whole engine run at once — a
    // long turn is tens of thousands of reasoning deltas — and rendering each one
    // separately pins the main thread. 50ms is imperceptible while live-streaming.
    experimental_throttle: 50,
    generateId: generateUUID,
    id: chatId,
    messages: initialMessages,
    onData: (dataPart) => {
      // Heartbeat that keeps the connection alive during idle gaps — ignore it.
      if (dataPart.type === "data-keepalive") {
        return;
      }
      if (dataPart.type === "data-waiting-status") {
        setWaitingStatus(dataPart.data);
        return;
      }
      setDataStream((ds) => (ds ? [...ds, dataPart] : []));
    },
    onError: (error) => {
      if (error.message?.includes("AI Gateway requires a valid credit card")) {
        setShowCreditCardAlert(true);
        return;
      }
      if (error instanceof ChatbotError) {
        toast({ description: error.message, type: "error" });
        return;
      }
      // A generic stream/network drop mid-run. The engine run is independent and
      // durable, so don't fail the turn or toast — the reconnect effect below
      // resumes from the engine and finishes. It only toasts if recovery gives up.
    },
    onFinish: () => {
      mutate(unstable_serialize(getChatHistoryPaginationKey));
    },
    sendAutomaticallyWhen: ({ messages: currentMessages }) => {
      const lastMessage = currentMessages.at(-1);
      return (
        lastMessage?.parts?.some(
          (part) =>
            "state" in part &&
            part.state === "approval-responded" &&
            "approval" in part &&
            (part.approval as { approved?: boolean })?.approved === true
        ) ?? false
      );
    },
    transport: new DefaultChatTransport({
      api: `${process.env.NEXT_PUBLIC_BASE_PATH ?? ""}/api/chat`,
      fetch: fetchWithErrorHandlers,
      prepareSendMessagesRequest(request) {
        const lastMessage = request.messages.at(-1);
        const isToolApprovalContinuation =
          lastMessage?.role !== "user" ||
          request.messages.some((msg) =>
            msg.parts?.some((part) => {
              const { state } = part as { state?: string };
              return (
                state === "approval-responded" || state === "output-denied"
              );
            })
          );

        // Ambient: the currently-selected video (set by the video selector).
        const videoId =
          typeof window !== "undefined"
            ? localStorage.getItem("ambient.videoId") || undefined
            : undefined;
        // Ambient: the active Studio agent definition (LLM setup) + the global
        // run mode (top-center toggle) — separate concerns now.
        const isClient = typeof window !== "undefined";
        const activeDef = isClient ? getActiveAgentDef() : undefined;
        const agentDef = activeDef
          ? {
              id: activeDef.id,
              model: activeDef.model,
              system: activeDef.system,
              baseUrl: activeDef.baseUrl,
              apiKey: activeDef.apiKey,
            }
          : undefined;

        return {
          body: {
            id: request.id,
            ...(isToolApprovalContinuation
              ? { messages: request.messages }
              : { message: lastMessage }),
            selectedChatModel: currentModelIdRef.current,
            selectedVisibilityType: visibility,
            videoId,
            ambientMode: isClient ? getMode() : undefined,
            agentDef,
            ...request.body,
          },
        };
      },
    }),
  });

  // Read inside effects without re-running them on every status change.
  const statusRef = useRef(status);
  statusRef.current = status;

  useEffect(() => {
    if (status === "submitted" || status === "ready" || status === "error") {
      setWaitingStatus(undefined);
    }
  }, [status, setWaitingStatus]);

  // Auto-reconnect on a mid-run stream drop. The engine run is independent and
  // durable (it keeps going and persists every event), so when the network/stream
  // fails we resume from it — replaying what we missed and tailing to completion —
  // with exponential backoff. This is what makes a long session survive transient
  // network errors instead of freezing with a "Running" tool. useAutoResume only
  // covers page-load/remount; this covers errors during an active turn.
  const reconnectRef = useRef({ attempts: 0 });
  const MAX_RECONNECT = 6;
  useEffect(() => {
    const r = reconnectRef.current;
    if (status === "streaming" || status === "ready") {
      r.attempts = 0;
      return;
    }
    if (status !== "error") {
      return;
    }
    if (r.attempts >= MAX_RECONNECT) {
      toast({
        description: "Connection lost. Reload the page to pick the run back up.",
        type: "error",
      });
      return;
    }
    const delay = Math.min(800 * 2 ** r.attempts, 8000);
    const timer = setTimeout(async () => {
      r.attempts += 1;
      // Drop the partial assistant message so the resumed engine replay rebuilds
      // it cleanly from the start instead of appending a second copy to it.
      let dropped: ChatMessage | undefined;
      setMessages((msgs) => {
        const last = msgs.at(-1);
        if (last?.role !== "assistant") {
          return msgs;
        }
        dropped = last;
        return msgs.slice(0, -1);
      });
      await resumeStream();
      // Nothing to resume (the endpoint had no run for this turn): put the
      // partial back rather than leave the turn blank.
      if (dropped) {
        const restored = dropped;
        setMessages((msgs) =>
          msgs.at(-1)?.role === "user" ? [...msgs, restored] : msgs
        );
      }
    }, delay);
    return () => clearTimeout(timer);
  }, [status, resumeStream, setMessages]);

  // Chats started on this page: their live stream is the source of truth, so the
  // server copy (fetched mid-run once the URL switches to /chat/:id) never
  // overwrites them.
  const createdChatIds = useRef(new Set<string>());

  if (isNewChat && !createdChatIds.current.has(newChatIdRef.current)) {
    createdChatIds.current.add(newChatIdRef.current);
  }

  // Chats opened from history: apply the server copy once it's fresh. Applying
  // only the *first* copy SWR produced (as before) meant revisiting a chat within
  // the app showed the stale cached copy — e.g. a turn that was still running on
  // an earlier visit loaded partially and never caught up. Skipped while this
  // chat is streaming so a live turn is never clobbered.
  useEffect(() => {
    if (createdChatIds.current.has(chatId)) {
      return;
    }
    if (!chatData?.messages || isValidating) {
      return;
    }
    if (statusRef.current !== "ready") {
      return;
    }
    setMessages(chatData.messages);
  }, [chatId, chatData?.messages, isValidating, setMessages]);

  const prevChatIdRef = useRef(chatId);
  useEffect(() => {
    if (prevChatIdRef.current !== chatId) {
      prevChatIdRef.current = chatId;
      if (isNewChat) {
        setMessages([]);
      }
    }
  }, [chatId, isNewChat, setMessages]);

  useEffect(() => {
    if (chatData && !isNewChat) {
      const cookieModel = document.cookie
        .split("; ")
        .find((row) => row.startsWith("chat-model="))
        ?.split("=")[1];
      if (cookieModel) {
        setCurrentModelId(decodeURIComponent(cookieModel));
      }
    }
  }, [chatData, isNewChat]);

  const hasAppendedQueryRef = useRef(false);
  useEffect(() => {
    const params = new URLSearchParams(window.location.search);
    const query = params.get("query");
    if (query && !hasAppendedQueryRef.current) {
      hasAppendedQueryRef.current = true;
      window.history.replaceState(
        {},
        "",
        `${process.env.NEXT_PUBLIC_BASE_PATH ?? ""}/chat/${chatId}`
      );
      sendMessage({
        parts: [{ text: query, type: "text" }],
        role: "user" as const,
      });
    }
  }, [sendMessage, chatId]);

  useAutoResume({
    // Wait for the fresh server copy (not a stale SWR cache entry) — it decides
    // whether the latest turn needs resuming.
    autoResume: !isNewChat && !!chatData && !isValidating,
    chatId,
    initialMessages,
    resumeStream,
    setMessages,
    status,
  });

  const isReadonly = isNewChat ? false : (chatData?.isReadonly ?? false);

  const { data: votes } = useSWR<Vote[]>(
    !isReadonly && messages.length >= 2
      ? `${process.env.NEXT_PUBLIC_BASE_PATH ?? ""}/api/vote?chatId=${chatId}`
      : null,
    fetcher,
    { revalidateOnFocus: false }
  );

  const value = useMemo<ActiveChatContextValue>(
    () => ({
      addToolApprovalResponse,
      chatId,
      chatVideoId,
      currentModelId,
      input,
      isLoading: !isNewChat && isLoading,
      isReadonly,
      messages,
      regenerate,
      sendMessage,
      setCurrentModelId,
      setInput,
      setMessages,
      setShowCreditCardAlert,
      showCreditCardAlert,
      status,
      stop,
      visibilityType: visibility,
      votes,
    }),
    [
      chatId,
      chatVideoId,
      messages,
      setMessages,
      sendMessage,
      status,
      stop,
      regenerate,
      addToolApprovalResponse,
      input,
      visibility,
      isReadonly,
      isNewChat,
      isLoading,
      votes,
      currentModelId,
      showCreditCardAlert,
    ]
  );

  return (
    <ActiveChatContext.Provider value={value}>
      {children}
    </ActiveChatContext.Provider>
  );
}

export function useActiveChat() {
  const context = useContext(ActiveChatContext);
  if (!context) {
    throw new Error("useActiveChat must be used within ActiveChatProvider");
  }
  return context;
}

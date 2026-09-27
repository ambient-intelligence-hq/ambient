"use client";

import type { UseChatHelpers } from "@ai-sdk/react";
import { useChat } from "@ai-sdk/react";
import { DefaultChatTransport } from "ai";
import { usePathname, useRouter } from "next/navigation";
import {
  createContext,
  type Dispatch,
  type ReactNode,
  type SetStateAction,
  useCallback,
  useContext,
  useEffect,
  useLayoutEffect,
  useMemo,
  useRef,
  useState,
} from "react";
import useSWR, { useSWRConfig } from "swr";
import { unstable_serialize } from "swr/infinite";
import { useDataStream } from "@/components/chat/data-stream-provider";
import { getChatHistoryPaginationKey } from "@/components/chat/sidebar-history";
import { toast } from "@/components/chat/toast";
import type { VisibilityType } from "@/components/chat/visibility-selector";
import { useAutoResume } from "@/hooks/use-auto-resume";
import { getActiveAgentDef } from "@/lib/ai/agent-defs";
import { getMode } from "@/lib/ai/mode";
import { DEFAULT_CHAT_MODEL } from "@/lib/ai/models";
import type { Vote } from "@/lib/db/schema";
import { ChatbotError } from "@/lib/errors";
import { NEW_CHAT_EVENT } from "@/lib/new-chat";
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
  showCreditCardAlert: boolean;
  setShowCreditCardAlert: Dispatch<SetStateAction<boolean>>;
};

const ActiveChatContext = createContext<ActiveChatContextValue | null>(null);

// The page's video, kept in its own context so the stage, the video selector and
// the drop zone don't re-render on every streamed token (the chat context
// changes with each message update).
type ActiveVideoContextValue = {
  // What this page is about: the open chat's own video, or — on a new chat — the
  // video picked for it. Null until one is picked: every new chat starts empty.
  videoId: string | null;
  // Pick a video for this page. A new chat takes it as its video; a chat that
  // has already started is bound to its own video (its engine session reads only
  // that one), so picking a different one opens a new chat about it instead.
  selectVideo: (id: string) => void;
  // True once the chat has started (so its video is fixed and picking another
  // opens a new chat).
  boundToVideo: boolean;
};

const ActiveVideoContext = createContext<ActiveVideoContextValue | null>(null);

function extractChatId(pathname: string): string | null {
  const match = pathname.match(/\/chat\/([^/]+)/);
  return match ? match[1] : null;
}

export function ActiveChatProvider({ children }: { children: ReactNode }) {
  const pathname = usePathname();
  const router = useRouter();
  const { setDataStream, setWaitingStatus } = useDataStream();
  const { mutate } = useSWRConfig();

  const chatIdFromUrl = extractChatId(pathname);
  const isNewChat = !chatIdFromUrl;
  const newChatIdRef = useRef(generateUUID());
  const prevPathnameRef = useRef(pathname);

  // Bumped by "New chat" (see lib/new-chat.ts). Forces a fresh chat id even when
  // already on "/", where navigating changes nothing — so a new chat always
  // starts clean, with no video picked.
  const [newChatNonce, setNewChatNonce] = useState(0);
  const prevNewChatNonceRef = useRef(newChatNonce);
  useEffect(() => {
    const onNewChat = () => setNewChatNonce((n) => n + 1);
    window.addEventListener(NEW_CHAT_EVENT, onNewChat);
    return () => window.removeEventListener(NEW_CHAT_EVENT, onNewChat);
  }, []);

  if (
    isNewChat &&
    (prevPathnameRef.current !== pathname ||
      prevNewChatNonceRef.current !== newChatNonce)
  ) {
    newChatIdRef.current = generateUUID();
  }
  prevPathnameRef.current = pathname;
  prevNewChatNonceRef.current = newChatNonce;

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
  // The video recorded for this chat (see Chat.videoId); fixed once set.
  const chatVideoId: string | null = isNewChat
    ? null
    : (chatData?.videoId ?? null);

  // The video picked for a chat that has none recorded yet — i.e. a new chat.
  // Keyed by chat id, so every new chat starts empty. Once the first message
  // creates the chat, the server's `chatVideoId` takes over (the same value).
  const [draftVideo, setDraftVideo] = useState<{
    chatId: string;
    videoId: string;
  } | null>(null);
  const videoId =
    chatVideoId ?? (draftVideo?.chatId === chatId ? draftVideo.videoId : null);
  // Read by the chat transport at send time and by selectVideo.
  const videoIdRef = useRef(videoId);
  videoIdRef.current = videoId;
  const chatIdRef = useRef(chatId);
  chatIdRef.current = chatId;
  const chatVideoIdRef = useRef(chatVideoId);
  chatVideoIdRef.current = chatVideoId;

  // A video picked while viewing a chat bound to a different one: handed to the
  // new chat that picking it opens. Applied before paint so that chat never
  // flashes the empty screen.
  const pendingNewChatVideoRef = useRef<string | null>(null);
  useLayoutEffect(() => {
    if (isNewChat && pendingNewChatVideoRef.current) {
      setDraftVideo({ chatId, videoId: pendingNewChatVideoRef.current });
      pendingNewChatVideoRef.current = null;
    }
  }, [chatId, isNewChat]);

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

        // Ambient: this page's video (the chat's own, or the one picked for a
        // new chat). The server uses an existing chat's recorded video anyway.
        const pageVideoId = videoIdRef.current ?? undefined;
        // Ambient: the active Studio agent definition (LLM setup) + the global
        // run mode (top-center toggle) — separate concerns now.
        const isClient = typeof window !== "undefined";
        const activeDef = isClient ? getActiveAgentDef() : undefined;
        const agentDef = activeDef
          ? {
              apiKey: activeDef.apiKey,
              baseUrl: activeDef.baseUrl,
              id: activeDef.id,
              model: activeDef.model,
              system: activeDef.system,
            }
          : undefined;

        return {
          body: {
            id: request.id,
            ...(isToolApprovalContinuation
              ? { messages: request.messages }
              : { message: lastMessage }),
            agentDef,
            ambientMode: isClient ? getMode() : undefined,
            selectedChatModel: currentModelIdRef.current,
            selectedVisibilityType: visibility,
            videoId: pageVideoId,
            ...request.body,
          },
        };
      },
    }),
  });

  // Read inside effects without re-running them on every status change.
  const statusRef = useRef(status);
  statusRef.current = status;
  const hasMessagesRef = useRef(false);
  hasMessagesRef.current = messages.length > 0;

  const selectVideo = useCallback(
    (id: string) => {
      if (id === videoIdRef.current) {
        return;
      }
      // A chat that has started is bound to its video. Picking another would
      // show one video while follow-ups are answered about the other, so open a
      // new chat about the picked video instead.
      if (hasMessagesRef.current || chatVideoIdRef.current !== null) {
        pendingNewChatVideoRef.current = id;
        router.push("/");
        return;
      }
      setDraftVideo({ chatId: chatIdRef.current, videoId: id });
    },
    [router]
  );

  const boundToVideo = messages.length > 0 || chatVideoId !== null;
  const videoValue = useMemo<ActiveVideoContextValue>(
    () => ({ boundToVideo, selectVideo, videoId }),
    [boundToVideo, selectVideo, videoId]
  );

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
      <ActiveVideoContext.Provider value={videoValue}>
        {children}
      </ActiveVideoContext.Provider>
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

export function useActiveVideo() {
  const context = useContext(ActiveVideoContext);
  if (!context) {
    throw new Error("useActiveVideo must be used within ActiveChatProvider");
  }
  return context;
}

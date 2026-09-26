"use client";

import {
  type MouseEvent as ReactMouseEvent,
  useCallback,
  useEffect,
  useRef,
  useState,
} from "react";
import {
  AlertDialog,
  AlertDialogAction,
  AlertDialogCancel,
  AlertDialogContent,
  AlertDialogDescription,
  AlertDialogFooter,
  AlertDialogHeader,
  AlertDialogTitle,
} from "@/components/ui/alert-dialog";
import { useActiveChat } from "@/hooks/use-active-chat";
import {
  initialArtifactData,
  useArtifact,
  useArtifactSelector,
} from "@/hooks/use-artifact";
import type { Attachment, ChatMessage } from "@/lib/types";
import { cn } from "@/lib/utils";
import { Artifact } from "./artifact";
import { AssistantPanelHeader } from "./assistant-panel-header";
import { ChatHeader } from "./chat-header";
import { DataStreamHandler } from "./data-stream-handler";
import { submitEditedMessage } from "./message-editor";
import { Messages } from "./messages";
import { MultimodalInput } from "./multimodal-input";
import { SandboxTerminal } from "./sandbox-terminal";
import { VideoStage } from "./video-stage";

const CHAT_MIN_WIDTH = 360;
const CHAT_MAX_WIDTH = 860;
const clampChatWidth = (w: number) =>
  Math.max(CHAT_MIN_WIDTH, Math.min(CHAT_MAX_WIDTH, w));

export function ChatShell() {
  const {
    chatId,
    messages,
    setMessages,
    sendMessage,
    status,
    stop,
    regenerate,
    addToolApprovalResponse,
    input,
    setInput,
    visibilityType,
    isReadonly,
    isLoading,
    votes,
    currentModelId,
    setCurrentModelId,
    showCreditCardAlert,
    setShowCreditCardAlert,
  } = useActiveChat();

  const [editingMessage, setEditingMessage] = useState<ChatMessage | null>(
    null
  );
  const [attachments, setAttachments] = useState<Attachment[]>([]);
  const isArtifactVisible = useArtifactSelector((state) => state.isVisible);
  const { setArtifact } = useArtifact();

  // Resizable chat pane: a draggable grip on its left edge sets the panel width
  // (clamped + persisted per viewer). Until measured we fall back to the default
  // responsive width class to avoid a hydration/layout jump.
  const [chatWidth, setChatWidth] = useState<number | null>(null);
  const chatWidthRef = useRef<number | null>(null);
  chatWidthRef.current = chatWidth;

  useEffect(() => {
    try {
      const saved = Number(localStorage.getItem("ambient.chatWidth"));
      if (saved > 0) {
        setChatWidth(clampChatWidth(saved));
      }
    } catch {
      /* blocked storage — keep the default */
    }
  }, []);

  const startChatResize = useCallback((e: ReactMouseEvent) => {
    e.preventDefault();
    document.body.style.cursor = "col-resize";
    document.body.style.userSelect = "none";
    const onMove = (ev: MouseEvent) => {
      // Chat pane is right-aligned; its width is the gap from the cursor to the
      // right edge (minus the card's right gutter).
      setChatWidth(clampChatWidth(window.innerWidth - ev.clientX - 14));
    };
    const onUp = () => {
      document.body.style.cursor = "";
      document.body.style.userSelect = "";
      window.removeEventListener("mousemove", onMove);
      window.removeEventListener("mouseup", onUp);
      try {
        if (chatWidthRef.current) {
          localStorage.setItem(
            "ambient.chatWidth",
            String(Math.round(chatWidthRef.current))
          );
        }
      } catch {
        /* ignore */
      }
    };
    window.addEventListener("mousemove", onMove);
    window.addEventListener("mouseup", onUp);
  }, []);

  const stopRef = useRef(stop);
  stopRef.current = stop;

  const prevChatIdRef = useRef(chatId);
  useEffect(() => {
    if (prevChatIdRef.current !== chatId) {
      prevChatIdRef.current = chatId;
      stopRef.current();
      setArtifact(initialArtifactData);
      setEditingMessage(null);
      setAttachments([]);
    }
  }, [chatId, setArtifact]);

  const handleEditMessage = useCallback(
    (msg: ChatMessage) => {
      const text = msg.parts
        ?.filter((p) => p.type === "text")
        .map((p) => p.text)
        .join("");
      setInput(text ?? "");
      setEditingMessage(msg);
    },
    [setInput]
  );

  const handleCancelEdit = useCallback(() => {
    setEditingMessage(null);
    setInput("");
  }, [setInput]);

  const handleSendEditedMessage = useCallback(async () => {
    if (!editingMessage) {
      return;
    }

    const msg = editingMessage;
    setEditingMessage(null);
    await submitEditedMessage({
      message: msg,
      regenerate,
      setMessages,
      text: input,
    });
    setInput("");
  }, [editingMessage, input, regenerate, setInput, setMessages]);

  const handleActivateGateway = useCallback(() => {
    window.open(
      "https://vercel.com/d?to=%2F%5Bteam%5D%2F%7E%2Fai%3Fmodal%3Dadd-credit-card",
      "_blank"
    );
    window.location.href = `${process.env.NEXT_PUBLIC_BASE_PATH ?? ""}/`;
  }, []);

  return (
    <>
      <div className="flex h-dvh w-full flex-row overflow-hidden bg-sidebar">
        {/* Center stage: the video is the hero, with the sandbox terminal docked
            right beneath it. Both are centered in one scroll column so they read
            as a single stack instead of hugging opposite corners. */}
        <div className="flex min-w-0 flex-1 flex-col overflow-hidden">
          <ChatHeader
            chatId={chatId}
            isReadonly={isReadonly}
            selectedVisibilityType={visibilityType}
          />
          {/* Top-aligned stack: the video starts at the top of the stage (level
              with the chat panel), the terminal docked right beneath it. */}
          <div className="flex min-h-0 flex-1 flex-col gap-4 overflow-y-auto px-4 pt-3 pb-6 md:px-6">
            <VideoStage />
            <SandboxTerminal messages={messages} />
          </div>
        </div>

        {/* Draggable grip to resize the conversation panel. */}
        <button
          aria-label="Resize chat panel"
          className="group hidden shrink-0 cursor-col-resize items-center justify-center px-1 md:flex"
          onMouseDown={startChatResize}
          tabIndex={-1}
          type="button"
        >
          <span className="h-16 w-1 rounded-full bg-border transition-colors group-hover:bg-primary/60" />
        </button>

        {/* Right: the conversation panel — a floating rounded card. */}
        <div
          className={cn(
            "flex shrink-0 flex-col overflow-hidden border-l bg-background md:my-3 md:mr-3 md:rounded-3xl md:border md:shadow-[var(--shadow-float)]",
            chatWidth == null && "w-[42%] min-w-[400px] max-w-[600px]"
          )}
          style={chatWidth == null ? undefined : { width: chatWidth }}
        >
          <AssistantPanelHeader />
          <div className="relative flex min-h-0 flex-1 flex-col overflow-hidden">
            <Messages
              addToolApprovalResponse={addToolApprovalResponse}
              chatId={chatId}
              isArtifactVisible={isArtifactVisible}
              isLoading={isLoading}
              isReadonly={isReadonly}
              messages={messages}
              onEditMessage={handleEditMessage}
              regenerate={regenerate}
              selectedModelId={currentModelId}
              setMessages={setMessages}
              status={status}
              votes={votes}
            />

            <div className="sticky bottom-0 z-1 flex w-full gap-2 bg-background px-2 pb-3 md:px-3 md:pb-3">
              {!isReadonly && (
                <MultimodalInput
                  attachments={attachments}
                  chatId={chatId}
                  editingMessage={editingMessage}
                  input={input}
                  isLoading={isLoading}
                  messages={messages}
                  onCancelEdit={handleCancelEdit}
                  onModelChange={setCurrentModelId}
                  selectedModelId={currentModelId}
                  selectedVisibilityType={visibilityType}
                  sendMessage={
                    editingMessage ? handleSendEditedMessage : sendMessage
                  }
                  setAttachments={setAttachments}
                  setInput={setInput}
                  setMessages={setMessages}
                  status={status}
                  stop={stop}
                />
              )}
            </div>
          </div>
        </div>

        <Artifact
          addToolApprovalResponse={addToolApprovalResponse}
          attachments={attachments}
          chatId={chatId}
          input={input}
          isReadonly={isReadonly}
          messages={messages}
          regenerate={regenerate}
          selectedModelId={currentModelId}
          selectedVisibilityType={visibilityType}
          sendMessage={sendMessage}
          setAttachments={setAttachments}
          setInput={setInput}
          setMessages={setMessages}
          status={status}
          stop={stop}
          votes={votes}
        />
      </div>

      <DataStreamHandler />

      <AlertDialog
        onOpenChange={setShowCreditCardAlert}
        open={showCreditCardAlert}
      >
        <AlertDialogContent>
          <AlertDialogHeader>
            <AlertDialogTitle>Activate AI Gateway</AlertDialogTitle>
            <AlertDialogDescription>
              This application requires{" "}
              {process.env.NODE_ENV === "production" ? "the owner" : "you"} to
              activate Vercel AI Gateway.
            </AlertDialogDescription>
          </AlertDialogHeader>
          <AlertDialogFooter>
            <AlertDialogCancel>Cancel</AlertDialogCancel>
            <AlertDialogAction onClick={handleActivateGateway}>
              Activate
            </AlertDialogAction>
          </AlertDialogFooter>
        </AlertDialogContent>
      </AlertDialog>
    </>
  );
}

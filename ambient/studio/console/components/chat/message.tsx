"use client";
import type { UseChatHelpers } from "@ai-sdk/react";
import { useCallback, useEffect, useMemo, useState } from "react";
import type { Vote } from "@/lib/db/schema";
import type { ChatMessage } from "@/lib/types";
import { cn, sanitizeText } from "@/lib/utils";
import { MessageContent, MessageResponse } from "../ai-elements/message";
import { linkifyTimestamps, seekComponents } from "./seek";
import { Shimmer } from "../ai-elements/shimmer";
import {
  Tool,
  ToolContent,
  ToolHeader,
  ToolInput,
  ToolOutput,
} from "../ai-elements/tool";
import { useDataStream } from "./data-stream-provider";
import { DocumentToolResult } from "./document";
import { DocumentPreview } from "./document-preview";
import { AmbientAvatar } from "./ambient-mark";
import { MessageActions } from "./message-actions";
import { MessageReasoning } from "./message-reasoning";
import { PreviewAttachment } from "./preview-attachment";
import { UsageCard } from "./usage-card";
import { Weather } from "./weather";

// Playful, video-themed filler shown while the engine is between steps (first
// turn, or after a tool call while the model prefills its next move). A real
// engine status (e.g. "attaching video") always wins; these only fill the gaps.
const VIDEO_WAITING_MESSAGES = [
  "Scrubbing the timeline...",
  "Reading between the frames...",
  "Decoding pixels into prose...",
  "Chasing keyframes...",
  "Enhance... enhance... ENHANCE...",
  "Rewinding for a closer look...",
  "Interrogating the frames...",
  "Teaching pixels to talk...",
  "Untangling the timecodes...",
  "Pondering the scene...",
  "Squinting at frame 404...",
  "Asking the footage politely...",
  "Consulting the director's cut...",
  "Sifting through the b-roll...",
  "Cueing up the next scene...",
];

function WaitingText() {
  const { waitingStatus } = useDataStream();
  const status = waitingStatus?.message;

  // Shuffle once per mount so the lines vary between runs without repeating
  // back-to-back, then cycle one per second while there's no real status.
  const messages = useMemo(() => {
    const arr = [...VIDEO_WAITING_MESSAGES];
    for (let i = arr.length - 1; i > 0; i--) {
      const j = Math.floor(Math.random() * (i + 1));
      [arr[i], arr[j]] = [arr[j], arr[i]];
    }
    return arr;
  }, []);
  const [idx, setIdx] = useState(0);

  useEffect(() => {
    if (status) {
      return; // a real engine status is showing; don't rotate over it
    }
    const id = setInterval(
      () => setIdx((i) => (i + 1) % messages.length),
      8000
    );
    return () => clearInterval(id);
  }, [status, messages.length]);

  const waitingText = status ?? messages[idx];

  return (
    <div className="flex min-h-[calc(13px*1.65)] min-w-0 items-center text-[13px] leading-[1.65]">
      <Shimmer
        as="span"
        className="font-medium whitespace-normal break-words"
        duration={1}
      >
        {waitingText}
      </Shimmer>
    </div>
  );
}

function ToolApprovalActions({
  addToolApprovalResponse,
  approvalId,
}: {
  addToolApprovalResponse: UseChatHelpers<ChatMessage>["addToolApprovalResponse"];
  approvalId: string;
}) {
  const handleDeny = useCallback(() => {
    addToolApprovalResponse({
      approved: false,
      id: approvalId,
      reason: "User denied weather lookup",
    });
  }, [addToolApprovalResponse, approvalId]);

  const handleAllow = useCallback(() => {
    addToolApprovalResponse({
      approved: true,
      id: approvalId,
    });
  }, [addToolApprovalResponse, approvalId]);

  return (
    <div className="flex items-center justify-end gap-2 border-t px-4 py-3">
      <button
        className="rounded-md px-3 py-1.5 text-muted-foreground text-sm transition-colors hover:bg-muted hover:text-foreground"
        onClick={handleDeny}
        type="button"
      >
        Deny
      </button>
      <button
        className="rounded-md bg-primary px-3 py-1.5 text-primary-foreground text-sm transition-colors hover:bg-primary/90"
        onClick={handleAllow}
        type="button"
      >
        Allow
      </button>
    </div>
  );
}

const PurePreviewMessage = ({
  addToolApprovalResponse,
  chatId,
  message,
  vote,
  isLoading,
  setMessages: _setMessages,
  regenerate: _regenerate,
  isReadonly,
  requiresScrollPadding: _requiresScrollPadding,
  onEdit,
}: {
  addToolApprovalResponse: UseChatHelpers<ChatMessage>["addToolApprovalResponse"];
  chatId: string;
  message: ChatMessage;
  vote: Vote | undefined;
  isLoading: boolean;
  setMessages: UseChatHelpers<ChatMessage>["setMessages"];
  regenerate: UseChatHelpers<ChatMessage>["regenerate"];
  isReadonly: boolean;
  requiresScrollPadding: boolean;
  onEdit?: (message: ChatMessage) => void;
}) => {
  const attachmentsFromMessage = message.parts.filter(
    (part) => part.type === "file"
  );

  useDataStream();

  const isUser = message.role === "user";
  const isAssistant = message.role === "assistant";

  const hasAnyContent = message.parts?.some(
    (part) =>
      (part.type === "text" && part.text?.trim().length > 0) ||
      (part.type === "reasoning" &&
        "text" in part &&
        part.text?.trim().length > 0) ||
      part.type.startsWith("tool-") ||
      part.type === "dynamic-tool"
  );
  const isThinking = isAssistant && isLoading && !hasAnyContent;

  // After a tool call finishes, the engine goes quiet while the model prefills
  // its next turn (often many seconds on big multimodal prompts). Nothing streams
  // in that window, so without this the UI shows finished tool chips and no sign
  // of progress. Show a trailing loader whenever the live assistant message's
  // last part is a tool in a terminal state — i.e. tool done, next response pending.
  const lastPart = message.parts?.[message.parts.length - 1];
  const isAwaitingNextStep =
    isAssistant &&
    isLoading &&
    lastPart?.type === "dynamic-tool" &&
    (lastPart.state === "output-available" || lastPart.state === "output-error");

  const attachments = attachmentsFromMessage.length > 0 && (
    <div
      className="flex flex-row justify-end gap-2"
      data-testid={"message-attachments"}
    >
      {attachmentsFromMessage.map((attachment) => (
        <PreviewAttachment
          attachment={{
            contentType: attachment.mediaType,
            name: attachment.filename ?? "file",
            url: attachment.url,
          }}
          key={attachment.url}
        />
      ))}
    </div>
  );

  const parts = message.parts?.map((part, index) => {
    const { type } = part;
    const key = `message-${message.id}-part-${index}`;

    // Each reasoning part is rendered as its own "thinking" block (the engine
    // emits one per agent step, interleaved with tool calls) — deliberately not
    // merged, so multi-step runs read as distinct thinking chunks.
    if (type === "reasoning") {
      if (!part.text?.trim().length) {
        return null;
      }
      const isStreaming = "state" in part ? part.state === "streaming" : false;
      return (
        <MessageReasoning
          isLoading={isLoading && isStreaming}
          key={key}
          reasoning={part.text}
        />
      );
    }

    // Engine tools surface as AI-SDK dynamic (provider-executed) tools. We show
    // the arguments as key/value rows and the result as readable preformatted
    // text (the engine's tool output is plain text, not JSON).
    if (type === "dynamic-tool") {
      const { toolCallId, toolName, state } = part;
      const args =
        part.input && typeof part.input === "object"
          ? Object.entries(part.input as Record<string, unknown>).filter(
              ([k]) => k !== "video_id" && k !== "video_path"
            )
          : [];
      const output = part.output;
      const outputText =
        typeof output === "string"
          ? output
          : output == null
            ? ""
            : JSON.stringify(output, null, 2);
      return (
        <div className="w-[min(100%,520px)]" key={toolCallId}>
          <Tool className="w-full" defaultOpen={false}>
            <ToolHeader state={state} toolName={toolName} type="dynamic-tool" />
            <ToolContent>
              {args.length > 0 && (
                <div className="space-y-1.5">
                  <h4 className="font-medium text-muted-foreground text-xs uppercase tracking-wide">
                    Arguments
                  </h4>
                  <div className="space-y-1 rounded-md bg-muted/50 p-3">
                    {args.map(([k, v]) => (
                      <div className="flex gap-2 text-xs" key={k}>
                        <span className="shrink-0 font-medium text-muted-foreground">
                          {k}
                        </span>
                        <span className="break-all font-mono">
                          {typeof v === "string" ? v : JSON.stringify(v)}
                        </span>
                      </div>
                    ))}
                  </div>
                </div>
              )}
              {state === "output-available" && outputText && (
                <div className="space-y-1.5">
                  <h4 className="font-medium text-muted-foreground text-xs uppercase tracking-wide">
                    Result
                  </h4>
                  <pre className="max-h-72 overflow-auto whitespace-pre-wrap break-words rounded-md bg-muted/50 p-3 text-xs leading-relaxed">
                    {outputText}
                  </pre>
                </div>
              )}
              {state === "output-error" && (
                <ToolOutput errorText={part.errorText} output={undefined} />
              )}
            </ToolContent>
          </Tool>
        </div>
      );
    }

    if (type === "text") {
      return (
        <MessageContent
          className={cn("text-[13px] leading-[1.65]", {
            "w-fit max-w-[min(80%,56ch)] overflow-hidden break-words rounded-2xl rounded-br-lg border border-border/30 bg-gradient-to-br from-secondary to-muted px-3.5 py-2 shadow-[var(--shadow-card)]":
              message.role === "user",
          })}
          data-testid="message-content"
          key={key}
        >
          <MessageResponse components={seekComponents}>
            {message.role === "assistant"
              ? linkifyTimestamps(sanitizeText(part.text))
              : sanitizeText(part.text)}
          </MessageResponse>
        </MessageContent>
      );
    }

    if (type === "tool-getWeather") {
      const { toolCallId, state } = part;
      const approvalId = (part as { approval?: { id: string } }).approval?.id;
      const isDenied =
        state === "output-denied" ||
        (state === "approval-responded" &&
          (part as { approval?: { approved?: boolean } }).approval?.approved ===
            false);
      const widthClass = "w-[min(100%,450px)]";

      if (state === "output-available") {
        return (
          <div className={widthClass} key={toolCallId}>
            <Weather weatherAtLocation={part.output} />
          </div>
        );
      }

      if (isDenied) {
        return (
          <div className={widthClass} key={toolCallId}>
            <Tool className="w-full" defaultOpen={true}>
              <ToolHeader state="output-denied" type="tool-getWeather" />
              <ToolContent>
                <div className="px-4 py-3 text-muted-foreground text-sm">
                  Weather lookup was denied.
                </div>
              </ToolContent>
            </Tool>
          </div>
        );
      }

      if (state === "approval-responded") {
        return (
          <div className={widthClass} key={toolCallId}>
            <Tool className="w-full" defaultOpen={true}>
              <ToolHeader state={state} type="tool-getWeather" />
              <ToolContent>
                <ToolInput input={part.input} />
              </ToolContent>
            </Tool>
          </div>
        );
      }

      return (
        <div className={widthClass} key={toolCallId}>
          <Tool className="w-full" defaultOpen={true}>
            <ToolHeader state={state} type="tool-getWeather" />
            <ToolContent>
              {(state === "input-available" ||
                state === "approval-requested") && (
                <ToolInput input={part.input} />
              )}
              {state === "approval-requested" && approvalId && (
                <ToolApprovalActions
                  addToolApprovalResponse={addToolApprovalResponse}
                  approvalId={approvalId}
                />
              )}
            </ToolContent>
          </Tool>
        </div>
      );
    }

    if (type === "tool-createDocument") {
      const { toolCallId } = part;

      if (part.output && "error" in part.output) {
        return (
          <div
            className="rounded-lg border border-red-200 bg-red-50 p-4 text-red-500 dark:bg-red-950/50"
            key={toolCallId}
          >
            Error creating document: {String(part.output.error)}
          </div>
        );
      }

      return (
        <DocumentPreview
          isReadonly={isReadonly}
          key={toolCallId}
          result={part.output}
        />
      );
    }

    if (type === "tool-updateDocument") {
      const { toolCallId } = part;

      if (part.output && "error" in part.output) {
        return (
          <div
            className="rounded-lg border border-red-200 bg-red-50 p-4 text-red-500 dark:bg-red-950/50"
            key={toolCallId}
          >
            Error updating document: {String(part.output.error)}
          </div>
        );
      }

      return (
        <div className="relative" key={toolCallId}>
          <DocumentPreview
            args={{ ...part.output, isUpdate: true }}
            isReadonly={isReadonly}
            result={part.output}
          />
        </div>
      );
    }

    if (type === "tool-requestSuggestions") {
      const { toolCallId, state } = part;

      return (
        <Tool
          className="w-[min(100%,450px)]"
          defaultOpen={true}
          key={toolCallId}
        >
          <ToolHeader state={state} type="tool-requestSuggestions" />
          <ToolContent>
            {state === "input-available" && <ToolInput input={part.input} />}
            {state === "output-available" && (
              <ToolOutput
                errorText={undefined}
                output={
                  "error" in part.output ? (
                    <div className="rounded border p-2 text-red-500">
                      Error: {String(part.output.error)}
                    </div>
                  ) : (
                    <DocumentToolResult
                      isReadonly={isReadonly}
                      result={part.output}
                      type="request-suggestions"
                    />
                  )
                }
              />
            )}
          </ToolContent>
        </Tool>
      );
    }

    return null;
  });

  const actions = !isReadonly && (
    <MessageActions
      chatId={chatId}
      isLoading={isLoading}
      key={`action-${message.id}`}
      message={message}
      onEdit={onEdit ? () => onEdit(message) : undefined}
      vote={vote}
    />
  );

  const content = isThinking ? (
    <WaitingText />
  ) : (
    <>
      {attachments}
      {parts}
      {isAwaitingNextStep && <WaitingText />}
      {isAssistant && message.metadata?.usage && (
        <UsageCard usage={message.metadata.usage} />
      )}
      {actions}
    </>
  );

  return (
    <div
      className={cn(
        "group/message w-full",
        !isAssistant && "animate-[fade-up_0.25s_cubic-bezier(0.22,1,0.36,1)]"
      )}
      data-role={message.role}
      data-testid={`message-${message.role}`}
    >
      <div
        className={cn(
          isUser ? "flex flex-col items-end gap-2" : "flex items-start gap-3"
        )}
      >
        {isAssistant && (
          <div className="flex h-[calc(13px*1.65)] shrink-0 items-center">
            <AmbientAvatar />
          </div>
        )}
        {isAssistant ? (
          <div className="flex min-w-0 flex-1 flex-col gap-2">{content}</div>
        ) : (
          content
        )}
      </div>
    </div>
  );
};

export const PreviewMessage = PurePreviewMessage;

export const ThinkingMessage = () => (
  <div
    className="group/message w-full"
    data-role="assistant"
    data-testid="message-assistant-loading"
  >
    <div className="flex items-start gap-3">
      <div className="flex h-[calc(13px*1.65)] shrink-0 items-center">
        <AmbientAvatar />
      </div>

      <WaitingText />
    </div>
  </div>
);

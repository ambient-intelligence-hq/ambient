"use client";

import { UploadIcon } from "lucide-react";
import {
  type DragEvent,
  type ReactNode,
  useCallback,
  useRef,
  useState,
} from "react";
import { useActiveVideo } from "@/hooks/use-active-chat";
import { isVideoFile, useVideoIntake } from "@/hooks/use-video-intake";
import { cn } from "@/lib/utils";
import { toast } from "./toast";

const carriesFiles = (e: DragEvent) =>
  Array.from(e.dataTransfer.types).includes("Files");

// Drop a video file anywhere on the center stage to upload it. On a new chat it
// becomes the chat's video; on a chat that already has one, it opens a new chat
// about the dropped video (see selectVideo).
export function StageDropZone({
  children,
  className,
}: {
  children: ReactNode;
  className?: string;
}) {
  const { boundToVideo } = useActiveVideo();
  const { uploadVideo, busy } = useVideoIntake();
  const [dragging, setDragging] = useState(false);
  // dragenter/dragleave fire for every child crossed; count them so the overlay
  // doesn't flicker as the cursor moves over the stage's contents.
  const depth = useRef(0);

  const onDragEnter = useCallback((e: DragEvent) => {
    if (!carriesFiles(e)) {
      return;
    }
    e.preventDefault();
    depth.current += 1;
    setDragging(true);
  }, []);

  const onDragLeave = useCallback((e: DragEvent) => {
    if (!carriesFiles(e)) {
      return;
    }
    depth.current = Math.max(0, depth.current - 1);
    if (depth.current === 0) {
      setDragging(false);
    }
  }, []);

  const onDragOver = useCallback(
    (e: DragEvent) => {
      if (carriesFiles(e)) {
        e.preventDefault();
        e.dataTransfer.dropEffect = busy ? "none" : "copy";
      }
    },
    [busy]
  );

  const onDrop = useCallback(
    (e: DragEvent) => {
      if (!carriesFiles(e)) {
        return;
      }
      e.preventDefault();
      depth.current = 0;
      setDragging(false);
      if (busy) {
        return;
      }
      const video = Array.from(e.dataTransfer.files).find(isVideoFile);
      if (!video) {
        toast({ description: "Drop a video file to add it.", type: "error" });
        return;
      }
      uploadVideo(video);
    },
    [busy, uploadVideo]
  );

  return (
    // biome-ignore lint/a11y/noStaticElementInteractions: a drop target, not a control — keyboard users add videos via the stage's buttons
    // biome-ignore lint/a11y/noNoninteractiveElementInteractions: same drop target (drag events only)
    <div
      className={cn("relative", className)}
      onDragEnter={onDragEnter}
      onDragLeave={onDragLeave}
      onDragOver={onDragOver}
      onDrop={onDrop}
    >
      {children}
      {dragging ? (
        <div className="pointer-events-none absolute inset-2 z-20 flex flex-col items-center justify-center gap-3 rounded-3xl border-2 border-primary/40 border-dashed bg-background/85 backdrop-blur-sm">
          <div className="flex size-12 items-center justify-center rounded-2xl border bg-card shadow-[var(--shadow-card)]">
            <UploadIcon className="size-5" />
          </div>
          <p className="font-medium text-sm">
            {busy
              ? "Wait for the current upload to finish"
              : "Drop to add this video"}
          </p>
          {boundToVideo && !busy ? (
            <p className="text-muted-foreground text-xs">
              It will open in a new chat.
            </p>
          ) : null}
        </div>
      ) : null}
    </div>
  );
}

"use client";

import {
  FileVideoIcon,
  Loader2Icon,
  SparklesIcon,
  TriangleAlertIcon,
} from "lucide-react";
import { useEffect, useRef, useState } from "react";
import useSWR from "swr";
import { useActiveChat } from "@/hooks/use-active-chat";
import { useSelectedVideo } from "@/hooks/use-selected-video";
import { cn, fetcher } from "@/lib/utils";
import { onSeek } from "./seek";

type EngineFile = {
  id: string;
  filename: string;
  description?: string;
  description_status?: string | null;
  size_bytes?: number;
  source_type?: string | null;
  source_status?: string | null;
  source_error?: string | null;
};
const contentPath = (id: string) => `/api/ambient/files/${id}/content`;

// The center "stage": the video is the hero (reference layout). Citation clicks
// in the chat seek it via the `ambient:seek` event.
export function VideoStage() {
  // A chat page shows its own video. The global selection (shared across tabs
  // via localStorage) only decides the video for a new chat — reading it here
  // made a session page play whatever was selected last, in any tab.
  const { chatVideoId } = useActiveChat();
  const selectedVideo = useSelectedVideo();
  const videoId = chatVideoId ?? selectedVideo;
  const vref = useRef<HTMLVideoElement>(null);
  const [flash, setFlash] = useState(false);
  const { data } = useSWR<{ data: EngineFile[] }>("/api/ambient/files", fetcher, {
    // While the selected source is still being prepared (e.g. a YouTube import
    // downloading), poll so the player swaps in the moment it's ready.
    refreshInterval: (latest) => {
      const f = latest?.data?.find((x) => x.id === videoId);
      return f?.source_status && f.source_status !== "ready" ? 2500 : 0;
    },
  });
  const file = data?.data.find((f) => f.id === videoId);
  const preparing =
    file?.source_status != null &&
    file.source_status !== "ready" &&
    file.source_status !== "failed";
  const sourceFailed = file?.source_status === "failed";

  useEffect(
    () =>
      onSeek((seconds) => {
        const v = vref.current;
        if (!v) return;
        v.currentTime = Math.max(0, seconds);
        v.play().catch(() => {});
        setFlash(true);
        setTimeout(() => setFlash(false), 700);
      }),
    []
  );

  if (!videoId) {
    return (
      <div className="flex flex-1 flex-col items-center justify-center gap-4 p-8 text-center">
        <div className="flex size-16 items-center justify-center rounded-3xl bg-muted">
          <FileVideoIcon className="size-7 text-muted-foreground" />
        </div>
        <div>
          <h2 className="font-semibold text-lg">Pick a video to begin</h2>
          <p className="mt-1 max-w-sm text-muted-foreground text-sm">
            Choose one from the selector above or upload a new video, then ask
            anything about it on the right.
          </p>
        </div>
      </div>
    );
  }

  return (
    <div className="mx-auto flex w-full max-w-3xl shrink-0 flex-col gap-4">
      {/* Hero video card — a spinner while the source is still being prepared
          (e.g. a YouTube download), an error if preparation failed, else the
          player. */}
      <div
        className={cn(
          "relative overflow-hidden rounded-3xl border bg-black shadow-[var(--shadow-float)] transition-shadow",
          flash && "ring-2 ring-primary ring-offset-2 ring-offset-background"
        )}
      >
        {preparing ? (
          <div className="flex aspect-video w-full flex-col items-center justify-center gap-3 p-6 text-center">
            <Loader2Icon className="size-7 animate-spin text-muted-foreground" />
            <div>
              <p className="font-medium text-sm text-white/90">
                {file?.source_type === "youtube"
                  ? "Importing from YouTube…"
                  : "Preparing video…"}
              </p>
              <p className="mt-1 text-white/50 text-xs">
                Downloading and preparing the source. This can take a moment.
              </p>
            </div>
          </div>
        ) : sourceFailed ? (
          <div className="flex aspect-video w-full flex-col items-center justify-center gap-3 p-6 text-center">
            <TriangleAlertIcon className="size-7 text-amber-400" />
            <div>
              <p className="font-medium text-sm text-white/90">
                Couldn&apos;t prepare this video
              </p>
              <p className="mt-1 max-w-md text-white/50 text-xs">
                {file?.source_error ||
                  "The import failed. Re-paste the URL to try again."}
              </p>
            </div>
          </div>
        ) : (
          <video
            className="aspect-video w-full"
            controls
            key={videoId}
            preload="metadata"
            ref={vref}
            src={contentPath(videoId)}
          >
            <track kind="captions" />
          </video>
        )}
      </div>

      {/* Title row */}
      <div className="flex items-center gap-2 px-1">
        <FileVideoIcon className="size-4 shrink-0 text-muted-foreground" />
        <h1 className="truncate font-semibold text-base" title={file?.filename}>
          {file?.filename ?? videoId}
        </h1>
        {file?.description_status === "ready" && (
          <span className="ml-auto flex items-center gap-1 rounded-full bg-primary/10 px-2 py-0.5 text-primary text-xs">
            <SparklesIcon className="size-3" /> described
          </span>
        )}
      </div>

      {/* Description card */}
      {file?.description && (
        <div className="rounded-2xl border bg-card/50 p-4 text-muted-foreground text-sm leading-relaxed">
          <div className="mb-1.5 font-medium text-foreground/70 text-xs uppercase tracking-wide">
            Overview
          </div>
          {file.description}
        </div>
      )}
    </div>
  );
}

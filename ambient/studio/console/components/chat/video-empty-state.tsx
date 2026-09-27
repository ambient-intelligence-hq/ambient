"use client";

import {
  ArrowRightIcon,
  ClockIcon,
  FolderOpenIcon,
  Loader2Icon,
  MessageSquareIcon,
  PlayIcon,
  SparklesIcon,
  UploadIcon,
  YoutubeIcon,
} from "lucide-react";
import {
  type ChangeEvent,
  type FormEvent,
  useCallback,
  useRef,
  useState,
} from "react";
import { Button } from "@/components/ui/button";
import {
  InputGroup,
  InputGroupAddon,
  InputGroupButton,
  InputGroupInput,
} from "@/components/ui/input-group";
import { useActiveVideo } from "@/hooks/use-active-chat";
import { useVideoIntake } from "@/hooks/use-video-intake";
import { VideoLibraryDialog } from "./video-library-dialog";

const FEATURES = [
  { icon: MessageSquareIcon, label: "Understand what’s happening" },
  { icon: ClockIcon, label: "Find key moments with timestamps" },
  { icon: SparklesIcon, label: "Take action with agent tools" },
];

// The stage before a video is picked — every new chat starts here. Three ways
// in: upload from disk (or drop a file anywhere on the stage), reuse a video
// from the library, or import a YouTube URL.
export function VideoEmptyState() {
  const { videoId, selectVideo } = useActiveVideo();
  const { uploadVideo, importYouTube, upload, importing, busy } =
    useVideoIntake();
  const fileRef = useRef<HTMLInputElement>(null);
  const [libraryOpen, setLibraryOpen] = useState(false);
  const [url, setUrl] = useState("");

  const onImport = useCallback(
    async (e: FormEvent) => {
      e.preventDefault();
      if (await importYouTube(url)) {
        setUrl("");
      }
    },
    [importYouTube, url]
  );
  const openFilePicker = useCallback(() => fileRef.current?.click(), []);
  const openLibrary = useCallback(() => setLibraryOpen(true), []);
  const onUrlChange = useCallback(
    (e: ChangeEvent<HTMLInputElement>) => setUrl(e.target.value),
    []
  );
  const onFilePicked = useCallback(
    (e: ChangeEvent<HTMLInputElement>) => {
      const file = e.target.files?.[0];
      e.target.value = "";
      if (file) {
        uploadVideo(file);
      }
    },
    [uploadVideo]
  );

  const uploadPercent = upload ? Math.round(upload.progress * 100) : 0;

  return (
    <div className="flex flex-1 flex-col items-center justify-center px-4 py-12">
      <div className="flex w-full max-w-lg flex-col items-center text-center">
        <div className="flex size-12 items-center justify-center rounded-2xl border bg-card shadow-[var(--shadow-card)]">
          <PlayIcon className="ml-0.5 size-5 text-foreground/80" />
        </div>

        <h2 className="mt-6 font-semibold text-2xl tracking-tight">
          Add a video to get started
        </h2>
        <p className="mt-2 text-muted-foreground text-sm">
          Analyze, ask questions, or take action with Ambient.
        </p>

        <div className="mt-8 flex flex-wrap items-center justify-center gap-3">
          <Button
            className="h-11 min-w-36 gap-2 rounded-full px-6 font-medium"
            disabled={busy}
            onClick={openFilePicker}
          >
            {upload ? (
              <Loader2Icon className="size-4 animate-spin" />
            ) : (
              <UploadIcon className="size-4" />
            )}
            {upload
              ? upload.progress < 1
                ? `Uploading ${uploadPercent}%`
                : "Preparing…"
              : "Add video"}
          </Button>
          <Button
            className="h-11 min-w-36 gap-2 rounded-full border border-border/60 px-6 font-medium"
            disabled={busy}
            onClick={openLibrary}
            variant="secondary"
          >
            <FolderOpenIcon className="size-4" />
            Browse files
          </Button>
        </div>

        {upload ? (
          <div className="mt-4 w-full max-w-xs">
            <div className="h-1 overflow-hidden rounded-full bg-muted">
              <div
                className="h-full rounded-full bg-primary transition-[width] duration-200"
                style={{ width: `${Math.max(uploadPercent, 2)}%` }}
              />
            </div>
            <p
              className="mt-2 truncate text-muted-foreground text-xs"
              title={upload.name}
            >
              {upload.name}
            </p>
          </div>
        ) : null}

        <div className="mt-8 flex w-full items-center gap-3 text-muted-foreground text-xs">
          <span className="h-px flex-1 bg-border" />
          or paste a YouTube URL
          <span className="h-px flex-1 bg-border" />
        </div>

        <form className="mt-4 w-full" onSubmit={onImport}>
          <InputGroup className="h-12 rounded-2xl bg-card shadow-[var(--shadow-card)]">
            <InputGroupAddon className="pl-4">
              <YoutubeIcon className="text-muted-foreground" />
            </InputGroupAddon>
            <InputGroupInput
              aria-label="YouTube URL"
              disabled={busy}
              inputMode="url"
              onChange={onUrlChange}
              placeholder="https://www.youtube.com/watch?v=..."
              value={url}
            />
            <InputGroupAddon align="inline-end" className="pr-2">
              <InputGroupButton
                aria-label="Import from YouTube"
                className="rounded-xl"
                disabled={busy || !url.trim()}
                size="icon-sm"
                type="submit"
                variant="secondary"
              >
                {importing ? (
                  <Loader2Icon className="size-4 animate-spin" />
                ) : (
                  <ArrowRightIcon className="size-4" />
                )}
              </InputGroupButton>
            </InputGroupAddon>
          </InputGroup>
        </form>

        <div className="mt-14 grid w-full grid-cols-3 gap-4">
          {FEATURES.map(({ icon: Icon, label }) => (
            <div
              className="flex flex-col items-center gap-2.5 px-2"
              key={label}
            >
              <Icon
                className="size-5 text-muted-foreground"
                strokeWidth={1.75}
              />
              <span className="max-w-32 text-muted-foreground text-xs leading-relaxed">
                {label}
              </span>
            </div>
          ))}
        </div>
      </div>

      <input
        accept="video/*"
        className="hidden"
        onChange={onFilePicked}
        ref={fileRef}
        type="file"
      />
      <VideoLibraryDialog
        currentVideoId={videoId}
        onOpenChange={setLibraryOpen}
        onPick={selectVideo}
        open={libraryOpen}
      />
    </div>
  );
}

"use client";

import { formatDistanceToNow } from "date-fns";
import {
  ArrowRightIcon,
  FileVideoIcon,
  Loader2Icon,
  SearchIcon,
  TriangleAlertIcon,
  UploadIcon,
  YoutubeIcon,
} from "lucide-react";
import {
  type ChangeEvent,
  type KeyboardEvent,
  type SyntheticEvent,
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
} from "react";
import useSWR from "swr";
import { Button } from "@/components/ui/button";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import {
  InputGroup,
  InputGroupAddon,
  InputGroupInput,
} from "@/components/ui/input-group";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Skeleton } from "@/components/ui/skeleton";
import { useVideoIntake, VIDEO_FILES_KEY } from "@/hooks/use-video-intake";
import { cn, fetcher } from "@/lib/utils";

type EngineFile = {
  id: string;
  filename: string;
  created_at: string;
  source_type?: string | null;
  source_status?: string | null;
};

const contentPath = (id: string) => `/api/ambient/files/${id}/content`;
const SKELETON_KEYS = ["s1", "s2", "s3", "s4", "s5", "s6"];
const YOUTUBE_URL = /^(https?:\/\/)?([\w-]+\.)*(youtube\.com|youtu\.be)\//i;

function formatDuration(seconds: number): string {
  const s = Math.max(0, Math.round(seconds));
  const h = Math.floor(s / 3600);
  const m = Math.floor((s % 3600) / 60);
  const sec = String(s % 60).padStart(2, "0");
  return h > 0 ? `${h}:${String(m).padStart(2, "0")}:${sec}` : `${m}:${sec}`;
}

// A frame from the video itself, loaded only once the card scrolls into view
// (the library can hold dozens of videos; each preview is a ranged request).
// Seeks a little way in so the frame isn't a black intro, and plays muted while
// the card is hovered.
function VideoThumb({
  id,
  active,
  onDuration,
}: {
  id: string;
  active: boolean;
  onDuration: (seconds: number) => void;
}) {
  const boxRef = useRef<HTMLDivElement>(null);
  const videoRef = useRef<HTMLVideoElement>(null);
  const [visible, setVisible] = useState(false);
  const [ready, setReady] = useState(false);
  const posterTime = useRef(0);

  useEffect(() => {
    const el = boxRef.current;
    if (!el) {
      return;
    }
    const io = new IntersectionObserver(
      ([entry]) => {
        if (entry?.isIntersecting) {
          setVisible(true);
          io.disconnect();
        }
      },
      { rootMargin: "200px" }
    );
    io.observe(el);
    return () => io.disconnect();
  }, []);

  useEffect(() => {
    const v = videoRef.current;
    if (!v || !ready) {
      return;
    }
    if (active) {
      v.play().catch(() => {
        /* autoplay can be refused; the still frame stays */
      });
    } else {
      v.pause();
      v.currentTime = posterTime.current;
    }
  }, [active, ready]);

  const onLoadedMetadata = useCallback(
    (e: SyntheticEvent<HTMLVideoElement>) => {
      const v = e.currentTarget;
      if (Number.isFinite(v.duration)) {
        onDuration(v.duration);
        posterTime.current = Math.min(1, v.duration * 0.1);
      }
      v.currentTime = posterTime.current;
    },
    [onDuration]
  );
  const onSeeked = useCallback(() => setReady(true), []);

  return (
    <div
      className="relative aspect-video w-full overflow-hidden bg-muted"
      ref={boxRef}
    >
      {ready ? null : <Skeleton className="absolute inset-0 rounded-none" />}
      {visible ? (
        <video
          className={cn(
            "size-full object-cover transition-opacity duration-300",
            ready ? "opacity-100" : "opacity-0"
          )}
          loop
          muted
          onLoadedMetadata={onLoadedMetadata}
          onSeeked={onSeeked}
          playsInline
          preload="metadata"
          ref={videoRef}
          src={contentPath(id)}
        />
      ) : null}
    </div>
  );
}

function LibraryCard({
  file,
  isCurrent,
  onPick,
}: {
  file: EngineFile;
  isCurrent: boolean;
  onPick: (id: string) => void;
}) {
  const [hovered, setHovered] = useState(false);
  const [duration, setDuration] = useState<number | null>(null);
  const failed = file.source_status === "failed";
  const preparing =
    file.source_status !== null &&
    file.source_status !== undefined &&
    file.source_status !== "ready" &&
    !failed;
  const activate = useCallback(() => setHovered(true), []);
  const deactivate = useCallback(() => setHovered(false), []);
  const pick = useCallback(() => onPick(file.id), [onPick, file.id]);
  const isYouTube = file.source_type === "youtube";

  return (
    <button
      className={cn(
        "group flex flex-col overflow-hidden rounded-2xl border bg-card text-left outline-none transition-all duration-200",
        "hover:-translate-y-0.5 hover:border-foreground/15 hover:shadow-[var(--shadow-card)]",
        "focus-visible:ring-[3px] focus-visible:ring-ring/50",
        "disabled:pointer-events-none disabled:opacity-55",
        isCurrent && "border-primary/40 ring-2 ring-primary/30"
      )}
      disabled={failed}
      onBlur={deactivate}
      onClick={pick}
      onFocus={activate}
      onMouseEnter={activate}
      onMouseLeave={deactivate}
      type="button"
    >
      <div className="relative">
        {preparing || failed ? (
          <div className="flex aspect-video w-full flex-col items-center justify-center gap-2 bg-muted text-muted-foreground">
            {failed ? (
              <TriangleAlertIcon className="size-5 text-amber-500" />
            ) : (
              <Loader2Icon className="size-5 animate-spin" />
            )}
            <span className="text-xs">
              {failed ? "Import failed" : "Importing…"}
            </span>
          </div>
        ) : (
          <VideoThumb active={hovered} id={file.id} onDuration={setDuration} />
        )}
        {duration === null ? null : (
          <span className="absolute right-2 bottom-2 rounded-md bg-black/70 px-1.5 py-0.5 font-medium text-[11px] text-white tabular-nums">
            {formatDuration(duration)}
          </span>
        )}
        {isCurrent ? (
          <span className="absolute top-2 left-2 rounded-md bg-primary px-1.5 py-0.5 font-medium text-[11px] text-primary-foreground">
            Current
          </span>
        ) : null}
      </div>
      <div className="flex min-w-0 flex-col gap-1 px-3.5 py-3">
        <span className="truncate font-medium text-sm" title={file.filename}>
          {file.filename}
        </span>
        <span className="flex items-center gap-1.5 text-muted-foreground text-xs">
          {isYouTube ? (
            <YoutubeIcon className="size-3.5 shrink-0" />
          ) : (
            <FileVideoIcon className="size-3.5 shrink-0" />
          )}
          <span className="truncate">
            {isYouTube ? "YouTube" : "Upload"} ·{" "}
            {formatDistanceToNow(new Date(file.created_at), {
              addSuffix: true,
            })}
          </span>
        </span>
      </div>
    </button>
  );
}

// Every video already in the engine — uploaded or imported in any session —
// with a preview of each, so a new chat can reuse one instead of re-uploading.
export function VideoLibraryDialog({
  open,
  onOpenChange,
  currentVideoId,
  onPick,
}: {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  currentVideoId: string | null;
  onPick: (id: string) => void;
}) {
  const [query, setQuery] = useState("");
  const { uploadVideo, importYouTube, upload, importing, busy } =
    useVideoIntake();
  const fileRef = useRef<HTMLInputElement>(null);
  const { data, isLoading } = useSWR<{ data: EngineFile[] }>(
    open ? VIDEO_FILES_KEY : null,
    fetcher
  );

  const files = useMemo(() => {
    const q = query.trim().toLowerCase();
    return [...(data?.data ?? [])]
      .filter((f) => !q || f.filename.toLowerCase().includes(q))
      .sort((a, b) => b.created_at.localeCompare(a.created_at));
  }, [data, query]);
  const total = data?.data?.length ?? 0;
  // Pasting a YouTube link into the search box offers to import it instead.
  const youTubeUrl = YOUTUBE_URL.test(query.trim()) ? query.trim() : null;

  const handleOpenChange = useCallback(
    (next: boolean) => {
      onOpenChange(next);
      if (!next) {
        setQuery("");
      }
    },
    [onOpenChange]
  );
  const onQueryChange = useCallback(
    (e: ChangeEvent<HTMLInputElement>) => setQuery(e.target.value),
    []
  );
  const pickAndClose = useCallback(
    (id: string) => {
      onPick(id);
      onOpenChange(false);
    },
    [onPick, onOpenChange]
  );
  // Upload and import pick the new video themselves once it lands (see
  // useVideoIntake); the dialog just gets out of the way while they run.
  const openFilePicker = useCallback(() => fileRef.current?.click(), []);
  const onFilePicked = useCallback(
    (e: ChangeEvent<HTMLInputElement>) => {
      const file = e.target.files?.[0];
      e.target.value = "";
      if (file) {
        handleOpenChange(false);
        uploadVideo(file);
      }
    },
    [handleOpenChange, uploadVideo]
  );
  const onImport = useCallback(async () => {
    if (youTubeUrl && (await importYouTube(youTubeUrl))) {
      handleOpenChange(false);
    }
  }, [handleOpenChange, importYouTube, youTubeUrl]);
  const onQueryKeyDown = useCallback(
    (e: KeyboardEvent<HTMLInputElement>) => {
      if (e.key === "Enter" && youTubeUrl) {
        e.preventDefault();
        onImport();
      }
    },
    [onImport, youTubeUrl]
  );

  return (
    <Dialog onOpenChange={handleOpenChange} open={open}>
      <DialogContent className="flex h-[min(82vh,780px)] flex-col gap-0 overflow-hidden p-0 sm:max-w-4xl">
        <DialogHeader className="gap-1.5 border-b px-7 pt-7 pb-5">
          <DialogTitle className="text-xl">Your videos</DialogTitle>
          <DialogDescription>
            Everything you’ve uploaded or imported, across all sessions. Pick
            one to use it in this chat.
          </DialogDescription>
          <div className="mt-3 flex gap-2">
            <InputGroup className="h-10 flex-1">
              <InputGroupAddon>
                {youTubeUrl ? <YoutubeIcon /> : <SearchIcon />}
              </InputGroupAddon>
              <InputGroupInput
                aria-label="Search videos or paste a YouTube URL"
                autoFocus
                onChange={onQueryChange}
                onKeyDown={onQueryKeyDown}
                placeholder={
                  total
                    ? `Search ${total} videos or paste a YouTube URL…`
                    : "Search videos or paste a YouTube URL…"
                }
                value={query}
              />
            </InputGroup>
            <Button
              className="h-10 gap-2 rounded-md px-4"
              disabled={busy}
              onClick={openFilePicker}
              variant="outline"
            >
              {upload ? (
                <Loader2Icon className="size-4 animate-spin" />
              ) : (
                <UploadIcon className="size-4" />
              )}
              Upload
            </Button>
            <input
              accept="video/*"
              className="hidden"
              onChange={onFilePicked}
              ref={fileRef}
              type="file"
            />
          </div>
        </DialogHeader>

        <ScrollArea className="min-h-0 flex-1">
          {youTubeUrl ? (
            <div className="flex h-full min-h-72 flex-col items-center justify-center gap-2 p-10 text-center">
              <div className="flex size-11 items-center justify-center rounded-xl border bg-card">
                <YoutubeIcon className="size-5 text-muted-foreground" />
              </div>
              <p className="mt-2 font-medium text-sm">Import from YouTube</p>
              <p className="max-w-sm truncate text-muted-foreground text-sm">
                {youTubeUrl}
              </p>
              <Button
                className="mt-3 gap-2 rounded-full px-5"
                disabled={busy}
                onClick={onImport}
              >
                {importing ? (
                  <Loader2Icon className="size-4 animate-spin" />
                ) : (
                  <ArrowRightIcon className="size-4" />
                )}
                {importing ? "Importing…" : "Import video"}
              </Button>
            </div>
          ) : isLoading && !data ? (
            <div className="grid grid-cols-2 gap-4 p-7 md:grid-cols-3">
              {SKELETON_KEYS.map((key) => (
                <div className="overflow-hidden rounded-2xl border" key={key}>
                  <Skeleton className="aspect-video w-full rounded-none" />
                  <div className="space-y-2 p-3.5">
                    <Skeleton className="h-4 w-3/4" />
                    <Skeleton className="h-3 w-1/2" />
                  </div>
                </div>
              ))}
            </div>
          ) : files.length === 0 ? (
            <div className="flex h-full min-h-72 flex-col items-center justify-center gap-2 p-10 text-center">
              <div className="flex size-11 items-center justify-center rounded-xl border bg-card">
                <FileVideoIcon className="size-5 text-muted-foreground" />
              </div>
              <p className="mt-2 font-medium text-sm">
                {total ? "No videos match your search" : "No videos yet"}
              </p>
              <p className="max-w-xs text-muted-foreground text-sm">
                {total
                  ? "Try a different name."
                  : "Upload one or paste a YouTube URL to add your first video."}
              </p>
            </div>
          ) : (
            <div className="grid grid-cols-2 gap-4 p-7 md:grid-cols-3">
              {files.map((f) => (
                <LibraryCard
                  file={f}
                  isCurrent={f.id === currentVideoId}
                  key={f.id}
                  onPick={pickAndClose}
                />
              ))}
            </div>
          )}
        </ScrollArea>
      </DialogContent>
    </Dialog>
  );
}

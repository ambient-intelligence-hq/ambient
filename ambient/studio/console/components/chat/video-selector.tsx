"use client";

import { FilmIcon, Loader2Icon } from "lucide-react";
import { memo, useCallback, useState } from "react";
import useSWR from "swr";
import { Button } from "@/components/ui/button";
import { useActiveVideo } from "@/hooks/use-active-chat";
import { useVideoIntake, VIDEO_FILES_KEY } from "@/hooks/use-video-intake";
import { fetcher } from "@/lib/utils";
import { VideoLibraryDialog } from "./video-library-dialog";

type EngineFile = { id: string; filename: string };

// Header video pill. Shows the page's video (the chat's own, or the one picked
// for a new chat — "Select video" until then) and opens the video library, where
// you can pick, upload or import one. Picking goes through selectVideo, which
// opens a new chat when the current one is already bound to a different video.
function PureVideoSelector() {
  const { videoId, selectVideo } = useActiveVideo();
  const { upload, importing } = useVideoIntake();
  const { data } = useSWR<{ data: EngineFile[] }>(VIDEO_FILES_KEY, fetcher);
  const current = data?.data?.find((f) => f.id === videoId);
  const [open, setOpen] = useState(false);
  const openLibrary = useCallback(() => setOpen(true), []);

  let label = current ? current.filename : "Select video";
  if (upload) {
    label =
      upload.progress < 1
        ? `Uploading ${Math.round(upload.progress * 100)}%`
        : "Preparing…";
  } else if (importing) {
    label = "Importing…";
  }

  return (
    <>
      <Button
        className="h-8 gap-1.5 px-2.5 text-sm"
        onClick={openLibrary}
        size="sm"
        title="Browse videos"
        variant="outline"
      >
        {upload || importing ? (
          <Loader2Icon className="size-3.5 animate-spin text-muted-foreground" />
        ) : (
          <FilmIcon className="size-3.5 text-muted-foreground" />
        )}
        <span className="max-w-45 truncate">{label}</span>
      </Button>
      <VideoLibraryDialog
        currentVideoId={videoId}
        onOpenChange={setOpen}
        onPick={selectVideo}
        open={open}
      />
    </>
  );
}

export const VideoSelector = memo(PureVideoSelector);

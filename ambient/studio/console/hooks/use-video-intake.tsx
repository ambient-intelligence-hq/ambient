"use client";

import {
  createContext,
  type ReactNode,
  useCallback,
  useContext,
  useMemo,
  useState,
} from "react";
import { useSWRConfig } from "swr";
import { toast } from "@/components/chat/toast";
import { useActiveVideo } from "@/hooks/use-active-chat";

// Getting a video into the engine — upload from disk (button or drag-and-drop)
// or import from a YouTube URL — then picking it for this page. One provider so
// every entry point (the empty stage, the header selector, the drop zone) shares
// the same in-flight state: a drop onto the stage shows its progress on the
// stage even though the stage's buttons didn't start it.

// SWR key for the engine's file list (library dialog, header selector, player).
export const VIDEO_FILES_KEY = "/api/ambient/files";

const VIDEO_EXTENSIONS = /\.(mp4|mov|m4v|mkv|webm|avi|mpe?g|wmv|flv)$/i;

export function isVideoFile(file: File): boolean {
  return file.type.startsWith("video/") || VIDEO_EXTENSIONS.test(file.name);
}

type EngineFile = { id: string; filename: string };

export type UploadState = {
  name: string;
  // 0–1 while bytes are sent; 1 while the engine prepares the file.
  progress: number;
};

type VideoIntakeContextValue = {
  uploadVideo: (file: File) => Promise<void>;
  // Resolves true once the import is accepted (the video then prepares in the
  // background — the stage shows "Importing from YouTube…").
  importYouTube: (url: string) => Promise<boolean>;
  upload: UploadState | null;
  importing: boolean;
  busy: boolean;
};

const VideoIntakeContext = createContext<VideoIntakeContextValue | null>(null);

// fetch() can't report upload progress; XHR can. Videos are large, so a live
// percentage matters more than anywhere else in the app.
function postWithProgress(
  file: File,
  onProgress: (fraction: number) => void
): Promise<EngineFile> {
  return new Promise((resolve, reject) => {
    const form = new FormData();
    form.append("file", file, file.name);
    const xhr = new XMLHttpRequest();
    xhr.open("POST", VIDEO_FILES_KEY);
    xhr.upload.onprogress = (e) => {
      if (e.lengthComputable) {
        onProgress(e.loaded / e.total);
      }
    };
    xhr.onload = () => {
      if (xhr.status >= 200 && xhr.status < 300) {
        try {
          resolve(JSON.parse(xhr.responseText) as EngineFile);
        } catch (err) {
          reject(err);
        }
      } else {
        reject(new Error(`upload failed (${xhr.status})`));
      }
    };
    xhr.onerror = () => reject(new Error("network error during upload"));
    xhr.send(form);
  });
}

export function VideoIntakeProvider({ children }: { children: ReactNode }) {
  const { selectVideo } = useActiveVideo();
  const { mutate } = useSWRConfig();
  const [upload, setUpload] = useState<UploadState | null>(null);
  const [importing, setImporting] = useState(false);

  const uploadVideo = useCallback(
    async (file: File) => {
      if (!isVideoFile(file)) {
        toast({
          description: `“${file.name}” isn’t a video file.`,
          type: "error",
        });
        return;
      }
      setUpload({ name: file.name, progress: 0 });
      try {
        const meta = await postWithProgress(file, (progress) =>
          setUpload({ name: file.name, progress })
        );
        await mutate(VIDEO_FILES_KEY);
        selectVideo(meta.id);
      } catch {
        toast({
          description: `Couldn’t upload “${file.name}”. Please try again.`,
          type: "error",
        });
      } finally {
        setUpload(null);
      }
    },
    [mutate, selectVideo]
  );

  const importYouTube = useCallback(
    async (url: string) => {
      const trimmed = url.trim();
      if (!trimmed) {
        return false;
      }
      setImporting(true);
      try {
        const res = await fetch(`${VIDEO_FILES_KEY}/import`, {
          body: JSON.stringify({ url: trimmed }),
          headers: { "content-type": "application/json" },
          method: "POST",
        });
        if (!res.ok) {
          const err = (await res.json().catch(() => ({}))) as {
            error?: string;
          };
          toast({
            description:
              err.error ||
              "YouTube import failed. Check the URL and try again.",
            type: "error",
          });
          return false;
        }
        const meta = (await res.json()) as EngineFile;
        await mutate(VIDEO_FILES_KEY);
        selectVideo(meta.id);
        return true;
      } catch {
        toast({
          description:
            "YouTube import failed. Check your connection and try again.",
          type: "error",
        });
        return false;
      } finally {
        setImporting(false);
      }
    },
    [mutate, selectVideo]
  );

  const value = useMemo<VideoIntakeContextValue>(
    () => ({
      busy: upload !== null || importing,
      importing,
      importYouTube,
      upload,
      uploadVideo,
    }),
    [upload, importing, importYouTube, uploadVideo]
  );

  return (
    <VideoIntakeContext.Provider value={value}>
      {children}
    </VideoIntakeContext.Provider>
  );
}

export function useVideoIntake() {
  const context = useContext(VideoIntakeContext);
  if (!context) {
    throw new Error("useVideoIntake must be used within VideoIntakeProvider");
  }
  return context;
}

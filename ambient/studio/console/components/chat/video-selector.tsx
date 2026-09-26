"use client";

import {
  CheckIcon,
  ChevronDownIcon,
  FilmIcon,
  Loader2Icon,
  UploadIcon,
  YoutubeIcon,
} from "lucide-react";
import { usePathname, useRouter } from "next/navigation";
import { memo, useRef, useState } from "react";
import useSWR from "swr";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import { useActiveChat } from "@/hooks/use-active-chat";
import { setSelectedVideo, useSelectedVideo } from "@/hooks/use-selected-video";
import { cn, fetcher } from "@/lib/utils";
import { toast } from "./toast";

type EngineFile = { id: string; filename: string; description_status?: string | null };

function PureVideoSelector() {
  const selectedVideo = useSelectedVideo();
  const { data, mutate, isLoading } = useSWR<{ data: EngineFile[] }>("/api/ambient/files", fetcher);
  const files = data?.data ?? [];
  const inputRef = useRef<HTMLInputElement>(null);
  const [uploading, setUploading] = useState(false);
  const [ytUrl, setYtUrl] = useState("");
  const [importing, setImporting] = useState(false);
  const { chatVideoId, messages } = useActiveChat();
  // Same rule as the player: a chat page shows (and checks) its own video.
  const selected = chatVideoId ?? selectedVideo;
  const current = files.find((f) => f.id === selected);
  const pathname = usePathname();
  const router = useRouter();

  // A chat is bound to the video it was started on (its engine session reads
  // only that video). Picking a different video while viewing a chat with
  // messages starts a new chat about it — otherwise the player would show one
  // video while follow-ups are answered about another.
  const chooseVideo = (id: string) => {
    setSelectedVideo(id);
    const inChat = pathname.includes("/chat/") && messages.length > 0;
    if (inChat && id !== chatVideoId) {
      router.push("/");
    }
  };

  const upload = async (file: File) => {
    setUploading(true);
    try {
      const form = new FormData();
      form.append("file", file);
      const res = await fetch("/api/ambient/files", { method: "POST", body: form });
      if (res.ok) {
        const meta = await res.json();
        await mutate();
        chooseVideo(meta.id);
      }
    } finally {
      setUploading(false);
    }
  };

  const importYouTube = async () => {
    const url = ytUrl.trim();
    if (!url || importing) return;
    setImporting(true);
    try {
      const res = await fetch("/api/ambient/files/import", {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({ url }),
      });
      if (res.ok) {
        const meta = await res.json();
        await mutate();
        chooseVideo(meta.id);
        setYtUrl("");
        toast({
          type: "success",
          description: "Importing from YouTube — preparing the video…",
        });
      } else {
        const err = (await res.json().catch(() => ({}))) as {
          error?: string;
        };
        toast({
          type: "error",
          description: err.error || "YouTube import failed. Check the URL and try again.",
        });
      }
    } catch {
      toast({ type: "error", description: "YouTube import failed. Try again." });
    } finally {
      setImporting(false);
    }
  };

  return (
    <DropdownMenu>
      <DropdownMenuTrigger asChild>
        <Button className="h-8 gap-1.5 px-2.5 text-sm" size="sm" variant="outline">
          <FilmIcon className="size-3.5 text-muted-foreground" />
          <span className="max-w-45 truncate">{current ? current.filename : "Select video"}</span>
          <ChevronDownIcon className="size-3.5 text-muted-foreground" />
        </Button>
      </DropdownMenuTrigger>
      <DropdownMenuContent align="start" className="w-72">
        <DropdownMenuItem
          onSelect={(e) => {
            e.preventDefault();
            inputRef.current?.click();
          }}
        >
          {uploading ? <Loader2Icon className="size-4 animate-spin" /> : <UploadIcon className="size-4" />}
          {uploading ? "Uploading…" : "Upload video"}
        </DropdownMenuItem>
        <input
          accept="video/*"
          className="hidden"
          onChange={(e) => e.target.files?.[0] && upload(e.target.files[0])}
          ref={inputRef}
          type="file"
        />

        {/* Import from a YouTube URL. Kept as a plain row so the menu stays open
            while typing; stopPropagation prevents Radix typeahead. */}
        <div
          className="flex items-center gap-1.5 px-2 py-1.5"
          onKeyDown={(e) => e.stopPropagation()}
        >
          <YoutubeIcon className="size-4 shrink-0 text-muted-foreground" />
          <Input
            className="h-7 flex-1 text-xs"
            disabled={importing}
            onChange={(e) => setYtUrl(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === "Enter") {
                e.preventDefault();
                importYouTube();
              }
            }}
            placeholder="Paste YouTube URL"
            value={ytUrl}
          />
          <Button
            className="h-7 px-2 text-xs"
            disabled={!ytUrl.trim() || importing}
            onClick={importYouTube}
            size="sm"
          >
            {importing ? (
              <Loader2Icon className="size-3.5 animate-spin" />
            ) : (
              "Import"
            )}
          </Button>
        </div>

        {files.length > 0 && <DropdownMenuSeparator />}
        {isLoading && (
          <div className="px-2 py-3 text-center text-muted-foreground text-xs">Loading…</div>
        )}
        {files.map((f) => (
          <DropdownMenuItem
            className="gap-2"
            key={f.id}
            onSelect={() => chooseVideo(f.id)}
          >
            <FilmIcon className="size-4 shrink-0 text-muted-foreground" />
            <span className="min-w-0 flex-1 truncate">{f.filename}</span>
            <CheckIcon className={cn("size-4 shrink-0", selected === f.id ? "opacity-100" : "opacity-0")} />
          </DropdownMenuItem>
        ))}
        {!isLoading && files.length === 0 && (
          <div className="px-2 py-3 text-center text-muted-foreground text-xs">No videos yet — upload one.</div>
        )}
      </DropdownMenuContent>
    </DropdownMenu>
  );
}

export const VideoSelector = memo(PureVideoSelector);

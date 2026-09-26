import { cn } from "@/lib/utils";

const MARK_SRC = `${process.env.NEXT_PUBLIC_BASE_PATH ?? ""}/ambient-mark.jpg`;

/**
 * The Ambient brand mark in a rounded, ring-framed chip — the assistant's
 * identity across the app (message avatars + the chat panel header). Reads
 * cleanly in both light and dark: the ring uses theme tokens and the mark's
 * paper ground sits as a small brand chip either way.
 */
export function AmbientAvatar({ className }: { className?: string }) {
  return (
    <div
      className={cn(
        "flex size-7 shrink-0 items-center justify-center overflow-hidden rounded-lg bg-background ring-1 ring-border/70",
        className
      )}
    >
      {/* biome-ignore lint/performance/noImgElement: tiny (4.7KB) static brand asset, no optimization needed */}
      <img
        alt="Ambient"
        className="size-full object-cover"
        draggable={false}
        src={MARK_SRC}
      />
    </div>
  );
}

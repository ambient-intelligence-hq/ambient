"use client";

// Click-to-seek: turn timestamps in assistant answers into links that jump the
// video player. The player (video-panel) listens for the `ambient:seek` event.
// We linkify the FULL accumulated text at render time (Streamdown re-renders the
// whole string each update), so streamed deltas never split a timestamp.

const SEEK_EVENT = "ambient:seek";

export function emitSeek(seconds: number) {
  window.dispatchEvent(new CustomEvent(SEEK_EVENT, { detail: seconds }));
}

export function onSeek(cb: (seconds: number) => void): () => void {
  const handler = (e: Event) => cb((e as CustomEvent<number>).detail);
  window.addEventListener(SEEK_EVENT, handler);
  return () => window.removeEventListener(SEEK_EVENT, handler);
}

// mm:ss / hh:mm:ss, and "N seconds/secs/sec" -> markdown links with a #seek-<s> href.
export function linkifyTimestamps(md: string): string {
  let out = md.replace(
    /\b(\d{1,2}):([0-5]\d)(?::([0-5]\d))?\b/g,
    (m, a, b, c) => {
      const secs = c ? +a * 3600 + +b * 60 + +c : +a * 60 + +b;
      return `[${m}](#seek-${secs})`;
    }
  );
  out = out.replace(
    /\b(\d+(?:\.\d+)?)\s*(seconds|secs|sec)\b/gi,
    (m, n) => `[${m}](#seek-${n})`
  );
  return out;
}

// Streamdown `components` override: intercept #seek-<s> links.
export const seekComponents = {
  a: ({ href, children, ...rest }: { href?: string; children?: React.ReactNode }) => {
    if (typeof href === "string" && href.startsWith("#seek-")) {
      const s = Number.parseFloat(href.slice(6));
      return (
        <button
          className="mx-0.5 inline-flex items-center rounded bg-primary/15 px-1 font-medium text-primary tabular-nums no-underline transition-colors hover:bg-primary/25"
          onClick={(e) => {
            e.preventDefault();
            emitSeek(s);
          }}
          title={`Jump to ${s}s`}
          type="button"
        >
          {children}
        </button>
      );
    }
    return (
      <a href={href} rel="noreferrer" target="_blank" {...rest}>
        {children}
      </a>
    );
  },
};

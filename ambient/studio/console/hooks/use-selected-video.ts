"use client";

import { useSyncExternalStore } from "react";

// Shared "currently-selected video" state, persisted in localStorage so the
// chat transport (use-active-chat) reads the same value it renders. Backed by a
// custom event so all subscribers (selector, player pane) update together.

const KEY = "ambient.videoId";
const EVENT = "ambient:video-change";

function emit() {
  window.dispatchEvent(new Event(EVENT));
}

export function setSelectedVideo(id: string | null) {
  if (id) localStorage.setItem(KEY, id);
  else localStorage.removeItem(KEY);
  emit();
}

function subscribe(cb: () => void) {
  window.addEventListener(EVENT, cb);
  window.addEventListener("storage", cb);
  return () => {
    window.removeEventListener(EVENT, cb);
    window.removeEventListener("storage", cb);
  };
}

export function useSelectedVideo(): string | null {
  return useSyncExternalStore(
    subscribe,
    () => localStorage.getItem(KEY),
    () => null
  );
}

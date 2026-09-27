"use client";

import { PanelLeftIcon } from "lucide-react";
import { memo } from "react";
import { Button } from "@/components/ui/button";
import { useSidebar } from "@/components/ui/sidebar";
import { ModeToggle } from "./mode-toggle";
import { VideoSelector } from "./video-selector";

function PureChatHeader() {
  const { toggleSidebar } = useSidebar();

  // Always render: the video selector must be reachable even when the sidebar is
  // collapsed (the original template hid the whole header in that state).
  return (
    <header className="sticky top-0 z-10 flex h-14 items-center gap-2 bg-sidebar px-3">
      <Button onClick={toggleSidebar} size="icon-sm" variant="ghost">
        <PanelLeftIcon className="size-4" />
      </Button>

      <VideoSelector />

      {/* Global run-mode pill, centered over the stage. */}
      <div className="-translate-x-1/2 pointer-events-none absolute left-1/2 hidden sm:block">
        <div className="pointer-events-auto">
          <ModeToggle />
        </div>
      </div>
    </header>
  );
}

export const ChatHeader = memo(PureChatHeader);

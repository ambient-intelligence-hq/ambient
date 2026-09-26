"use client";

import { PanelLeftIcon, SettingsIcon } from "lucide-react";
import Link from "next/link";
import { memo } from "react";
import { Button } from "@/components/ui/button";
import { useSidebar } from "@/components/ui/sidebar";
import { useAgentDefs } from "@/lib/ai/agent-defs";
import { ModeToggle } from "./mode-toggle";
import { VideoSelector } from "./video-selector";
import { VisibilitySelector, type VisibilityType } from "./visibility-selector";

function PureChatHeader({
  chatId,
  selectedVisibilityType,
  isReadonly,
}: {
  chatId: string;
  selectedVisibilityType: VisibilityType;
  isReadonly: boolean;
}) {
  const { toggleSidebar } = useSidebar();
  const { active } = useAgentDefs();

  // Always render: the video selector must be reachable even when the sidebar is
  // collapsed (the original template hid the whole header in that state).
  return (
    <header className="sticky top-0 z-10 flex h-14 items-center gap-2 bg-background px-3">
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

      {!isReadonly && (
        <VisibilitySelector
          chatId={chatId}
          selectedVisibilityType={selectedVisibilityType}
        />
      )}

      <Button
        asChild
        className="ml-auto gap-2 text-muted-foreground"
        size="sm"
        title="Agent settings"
        variant="ghost"
      >
        <Link href="/settings">
          <SettingsIcon className="size-4" />
          <span className="hidden max-w-32 truncate sm:inline">
            {active.name}
          </span>
        </Link>
      </Button>
    </header>
  );
}

export const ChatHeader = memo(
  PureChatHeader,
  (prevProps, nextProps) =>
    prevProps.chatId === nextProps.chatId &&
    prevProps.selectedVisibilityType === nextProps.selectedVisibilityType &&
    prevProps.isReadonly === nextProps.isReadonly
);

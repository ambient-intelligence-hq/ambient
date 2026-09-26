"use client";

import { MoreHorizontalIcon, SettingsIcon } from "lucide-react";
import Link from "next/link";
import { Button } from "@/components/ui/button";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import { AmbientAvatar } from "./ambient-mark";

// The chat panel's identity header: the Ambient mark + "Assistant", giving the
// conversation a clear, branded top edge that's separate from the messages.
// Height matches the left video toolbar (h-14) so the two panes line up.
export function AssistantPanelHeader() {
  const base = process.env.NEXT_PUBLIC_BASE_PATH ?? "";
  return (
    <header className="flex h-14 shrink-0 items-center gap-2.5 border-b px-4">
      <AmbientAvatar className="size-8 rounded-xl" />
      <div className="min-w-0">
        <div className="font-semibold text-sm leading-tight">Assistant</div>
        <div className="truncate text-muted-foreground text-xs leading-tight">
          Ambient video agent
        </div>
      </div>
      <div className="ml-auto">
        <DropdownMenu>
          <DropdownMenuTrigger asChild>
            <Button
              aria-label="Assistant options"
              className="size-8 text-muted-foreground"
              size="icon"
              variant="ghost"
            >
              <MoreHorizontalIcon className="size-4" />
            </Button>
          </DropdownMenuTrigger>
          <DropdownMenuContent align="end" className="w-44">
            <DropdownMenuItem asChild>
              <Link href={`${base}/settings`}>
                <SettingsIcon className="size-4" />
                Settings
              </Link>
            </DropdownMenuItem>
          </DropdownMenuContent>
        </DropdownMenu>
      </div>
    </header>
  );
}

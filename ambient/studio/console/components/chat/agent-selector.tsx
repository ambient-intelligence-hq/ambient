"use client";

import { BotIcon, CheckIcon, ChevronDownIcon, SettingsIcon } from "lucide-react";
import Link from "next/link";
import { Button } from "@/components/ui/button";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import { setActiveAgentDefId, useAgentDefs } from "@/lib/ai/agent-defs";
import { cn } from "@/lib/utils";

// The agent picker in the composer. Lists the saved agent definitions (Vanilla +
// customs), highlights the active one, and remembers the selection as last-used
// (setActiveAgentDefId persists it, and it's the default next session).
export function AgentSelector() {
  const { defs, activeId, active } = useAgentDefs();

  return (
    <DropdownMenu>
      <DropdownMenuTrigger asChild>
        <Button
          className="h-7 max-w-[200px] justify-between gap-1.5 rounded-lg px-2 text-[12px] text-muted-foreground transition-colors hover:text-foreground"
          data-testid="agent-selector"
          variant="ghost"
        >
          <BotIcon className="size-3.5 shrink-0" />
          <span className="truncate">{active.name}</span>
          <ChevronDownIcon className="size-3.5 shrink-0 opacity-60" />
        </Button>
      </DropdownMenuTrigger>
      <DropdownMenuContent align="start" className="w-64">
        {defs.map((def) => (
          <DropdownMenuItem
            className="flex items-start gap-2.5 py-2"
            key={def.id}
            onSelect={() => setActiveAgentDefId(def.id)}
          >
            <BotIcon className="mt-0.5 size-4 shrink-0 text-muted-foreground" />
            <span className="min-w-0 flex-1">
              <span className="block truncate font-medium text-sm">
                {def.name}
              </span>
              <span className="block truncate text-muted-foreground text-xs">
                {def.model || "engine default"}
                {def.baseUrl ? " · custom endpoint" : ""}
              </span>
            </span>
            <CheckIcon
              className={cn(
                "mt-0.5 size-4 shrink-0 text-primary",
                def.id === activeId ? "opacity-100" : "opacity-0"
              )}
            />
          </DropdownMenuItem>
        ))}
        <DropdownMenuSeparator />
        <DropdownMenuItem asChild>
          <Link className="gap-2 text-muted-foreground" href="/settings">
            <SettingsIcon className="size-4" />
            Manage agents
          </Link>
        </DropdownMenuItem>
      </DropdownMenuContent>
    </DropdownMenu>
  );
}

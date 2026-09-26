"use client";

import { BotIcon, ZapIcon } from "lucide-react";
import { type AgentMode, useMode } from "@/lib/ai/mode";
import { cn } from "@/lib/utils";

// Global run-mode pill (Agent / Fast), chosen before a session. Sits centered at
// the top of the stage. Mode is separate from the agent definition.
export function ModeToggle() {
  const [mode, setMode] = useMode();
  const options: { id: AgentMode; label: string; icon: typeof BotIcon }[] = [
    { id: "agent", label: "Agent", icon: BotIcon },
    { id: "fast", label: "Fast", icon: ZapIcon },
  ];

  return (
    <div className="inline-flex items-center rounded-full border bg-background/70 p-0.5 shadow-[var(--shadow-card)] backdrop-blur">
      {options.map((opt) => {
        const Icon = opt.icon;
        const active = mode === opt.id;
        return (
          <button
            className={cn(
              "flex items-center gap-1.5 rounded-full px-3.5 py-1 font-medium text-[13px] transition-colors",
              active
                ? "bg-muted text-foreground shadow-[var(--shadow-card)]"
                : "text-muted-foreground hover:text-foreground"
            )}
            key={opt.id}
            onClick={() => setMode(opt.id)}
            type="button"
          >
            <Icon className="size-3.5" />
            {opt.label}
          </button>
        );
      })}
    </div>
  );
}

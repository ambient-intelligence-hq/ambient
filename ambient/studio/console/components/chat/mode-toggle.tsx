"use client";

import { BotIcon, ZapIcon } from "lucide-react";
import { useCallback } from "react";
import {
  Tooltip,
  TooltipContent,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import { type AgentMode, useMode } from "@/lib/ai/mode";
import { cn } from "@/lib/utils";

const OPTIONS: {
  id: AgentMode;
  label: string;
  tagline: string;
  description: string;
  icon: typeof BotIcon;
}[] = [
  {
    description:
      "Digs into clips and frames with tools. Slower, but precise — timestamps, counts, exports.",
    icon: BotIcon,
    id: "agent",
    label: "Agent",
    tagline: "thorough",
  },
  {
    description:
      "One quick pass over the whole video. Great for summaries; may miss fine detail.",
    icon: ZapIcon,
    id: "fast",
    label: "Fast",
    tagline: "quick answers",
  },
];

function ModeOption({
  option,
  active,
  onSelect,
}: {
  option: (typeof OPTIONS)[number];
  active: boolean;
  onSelect: (mode: AgentMode) => void;
}) {
  const Icon = option.icon;
  const select = useCallback(() => onSelect(option.id), [onSelect, option.id]);

  return (
    // A short delay so sweeping the cursor across the pill doesn't flash both.
    <Tooltip delayDuration={300}>
      <TooltipTrigger asChild>
        <button
          aria-pressed={active}
          className={cn(
            "flex items-center gap-1.5 rounded-full px-3.5 py-1 font-medium text-[13px] transition-colors",
            active
              ? "bg-muted text-foreground shadow-[var(--shadow-card)]"
              : "text-muted-foreground hover:text-foreground"
          )}
          onClick={select}
          type="button"
        >
          <Icon className="size-3.5" />
          {option.label}
        </button>
      </TooltipTrigger>
      <TooltipContent
        className="w-56 max-w-56 flex-col items-start gap-1 rounded-xl px-3.5 py-2.5"
        side="bottom"
        sideOffset={8}
      >
        <span className="font-semibold text-[13px]">
          {option.label}
          <span className="font-normal opacity-60"> · {option.tagline}</span>
        </span>
        <span className="text-xs leading-relaxed opacity-75">
          {option.description}
        </span>
      </TooltipContent>
    </Tooltip>
  );
}

// Global run-mode pill (Agent / Fast), chosen before a session. Sits centered at
// the top of the stage. Mode is separate from the agent definition. Each option
// explains its trade-off on hover.
export function ModeToggle() {
  const [mode, setMode] = useMode();

  return (
    <div className="inline-flex items-center rounded-full border bg-background/70 p-0.5 shadow-[var(--shadow-card)] backdrop-blur">
      {OPTIONS.map((opt) => (
        <ModeOption
          active={mode === opt.id}
          key={opt.id}
          onSelect={setMode}
          option={opt}
        />
      ))}
    </div>
  );
}

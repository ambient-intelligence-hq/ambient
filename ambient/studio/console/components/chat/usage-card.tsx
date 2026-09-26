"use client";

import { ChevronDownIcon, GaugeIcon } from "lucide-react";
import {
  Collapsible,
  CollapsibleContent,
  CollapsibleTrigger,
} from "@/components/ui/collapsible";
import type { AmbientUsage } from "@/lib/types";

const num = (n?: number) => (typeof n === "number" ? n.toLocaleString() : "—");
const usd = (n?: number) =>
  typeof n === "number" ? `$${n.toFixed(4)}` : "—";

// Per-run token + cost usage (from the engine's session.status_idle), mirroring
// demo/ui's usage card.
export function UsageCard({ usage }: { usage: AmbientUsage }) {
  if (!usage || (usage.requests == null && usage.total_tokens == null)) {
    return null;
  }
  const bySource = Object.entries(usage.by_source ?? {});
  const metrics: [string, string][] = [
    ["Total tokens", num(usage.total_tokens)],
    ["Est. cost", usd(usage.cost)],
    ["Input", num(usage.input_tokens)],
    ["Output", num(usage.output_tokens)],
    ["Cache read", num(usage.cache_read_input_tokens)],
    ["Requests", num(usage.requests)],
  ];

  return (
    <Collapsible className="w-[min(100%,450px)]">
      <div className="rounded-xl border bg-muted/30">
        <CollapsibleTrigger className="group flex w-full items-center gap-2 px-3 py-2 text-left">
          <GaugeIcon className="size-4 text-muted-foreground" />
          <span className="font-medium text-sm">Usage &amp; cost</span>
          <span className="ml-auto flex items-center gap-2 text-muted-foreground text-xs">
            {usd(usage.cost)} · {num(usage.total_tokens)} tok
            <ChevronDownIcon className="size-4 transition-transform group-data-[state=open]:rotate-180" />
          </span>
        </CollapsibleTrigger>
        <CollapsibleContent>
          <div className="border-t px-3 py-3">
            <div className="grid grid-cols-2 gap-x-4 gap-y-2 sm:grid-cols-3">
              {metrics.map(([label, value]) => (
                <div key={label}>
                  <div className="text-muted-foreground text-xs">{label}</div>
                  <div className="font-medium text-sm tabular-nums">{value}</div>
                </div>
              ))}
            </div>
            {bySource.length > 0 && (
              <div className="mt-3 border-t pt-2">
                <div className="mb-1 text-muted-foreground text-xs uppercase tracking-wide">
                  By source
                </div>
                <div className="flex flex-col gap-1">
                  {bySource.map(([source, s]) => (
                    <div
                      className="flex items-center justify-between gap-3 text-xs"
                      key={source}
                    >
                      <span className="truncate font-medium">{source}</span>
                      <span className="shrink-0 text-muted-foreground tabular-nums">
                        {num(s.total_tokens)} tok · {usd(s.cost)} ·{" "}
                        {num(s.requests)} req
                      </span>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>
        </CollapsibleContent>
      </div>
    </Collapsible>
  );
}

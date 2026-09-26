"use client";

import { ArrowLeftIcon, InfoIcon, UsersIcon } from "lucide-react";
import Link from "next/link";
import { useState } from "react";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { cn } from "@/lib/utils";
import { AgentsPanel } from "./agents-panel";

type SectionId = "agents" | "about";

const SECTIONS: {
  id: SectionId;
  label: string;
  description: string;
  icon: typeof UsersIcon;
}[] = [
  {
    id: "agents",
    label: "Agents",
    description:
      "Define the model, LLM endpoint, and system prompt each agent runs with.",
    icon: UsersIcon,
  },
  {
    id: "about",
    label: "About",
    description: "Studio and engine information.",
    icon: InfoIcon,
  },
];

export function SettingsShell() {
  const [section, setSection] = useState<SectionId>("agents");
  const current = SECTIONS.find((s) => s.id === section) ?? SECTIONS[0];

  return (
    <div className="min-h-dvh bg-sidebar">
      {/* Top bar */}
      <header className="sticky top-0 z-20 flex h-14 items-center gap-3 border-b bg-background/80 px-4 backdrop-blur">
        <Link
          className="flex size-8 items-center justify-center rounded-lg text-muted-foreground transition-colors hover:bg-muted hover:text-foreground"
          href="/"
          title="Back to chat"
        >
          <ArrowLeftIcon className="size-4" />
        </Link>
        <h1 className="font-semibold text-base">Settings</h1>
      </header>

      {/* Body */}
      <div className="mx-auto grid max-w-5xl gap-8 px-4 py-8 md:grid-cols-[200px_minmax(0,1fr)] md:px-6">
        {/* Section nav */}
        <nav className="flex flex-row gap-1 overflow-x-auto md:sticky md:top-20 md:h-fit md:flex-col">
          {SECTIONS.map((s) => {
            const Icon = s.icon;
            const active = s.id === section;
            return (
              <button
                className={cn(
                  "flex items-center gap-2.5 rounded-lg px-3 py-2 text-left font-medium text-sm transition-colors",
                  active
                    ? "bg-background text-foreground shadow-[var(--shadow-card)]"
                    : "text-muted-foreground hover:bg-background/60 hover:text-foreground"
                )}
                key={s.id}
                onClick={() => setSection(s.id)}
                type="button"
              >
                <Icon className="size-4 shrink-0" />
                {s.label}
              </button>
            );
          })}
        </nav>

        {/* Content */}
        <main className="min-w-0">
          <div className="mb-6">
            <h2 className="font-semibold text-xl tracking-tight">
              {current.label}
            </h2>
            <p className="mt-1 text-muted-foreground text-sm">
              {current.description}
            </p>
          </div>

          {section === "agents" && <AgentsPanel />}
          {section === "about" && <AboutPanel />}
        </main>
      </div>
    </div>
  );
}

function AboutPanel() {
  return (
    <Card>
      <CardHeader>
        <CardTitle>Ambient Studio</CardTitle>
        <CardDescription>
          A console for the Ambient video-understanding engine.
        </CardDescription>
      </CardHeader>
      <CardContent className="grid gap-3 text-sm">
        <div className="flex justify-between gap-4 border-t pt-3">
          <span className="text-muted-foreground">LLM endpoint</span>
          <span className="text-right">
            Configured on the engine (shared by all agents)
          </span>
        </div>
        <div className="flex justify-between gap-4 border-t pt-3">
          <span className="text-muted-foreground">Agent definitions</span>
          <span className="text-right">Stored in this browser</span>
        </div>
      </CardContent>
    </Card>
  );
}

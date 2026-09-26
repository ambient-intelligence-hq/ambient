"use client";

import { BotIcon, CheckIcon, CopyIcon, PlusIcon, TrashIcon } from "lucide-react";
import { useEffect, useMemo, useState } from "react";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Textarea } from "@/components/ui/textarea";
import {
  type AgentDef,
  deleteAgentDef,
  newAgentDefId,
  setActiveAgentDefId,
  upsertAgentDef,
  useAgentDefs,
} from "@/lib/ai/agent-defs";
import { cn } from "@/lib/utils";
import { Field } from "./field";

function sameDef(a: AgentDef, b: AgentDef) {
  return (
    a.name === b.name &&
    a.model === b.model &&
    a.system === b.system &&
    a.baseUrl === b.baseUrl &&
    a.apiKey === b.apiKey
  );
}

const EMPTY_DRAFT = (): AgentDef => ({
  id: newAgentDefId(),
  name: "New agent",
  model: "",
  system: "",
  baseUrl: "",
  apiKey: "",
});

export function AgentsPanel() {
  const { defs, activeId } = useAgentDefs();
  const [selectedId, setSelectedId] = useState<string>(activeId);
  const [draft, setDraft] = useState<AgentDef | null>(null);

  const stored = useMemo(
    () => defs.find((d) => d.id === selectedId) ?? defs[0],
    [defs, selectedId]
  );

  useEffect(() => {
    setDraft(stored ? { ...stored } : null);
  }, [stored]);

  const dirty = !!(draft && stored && !sameDef(draft, stored));
  const isActive = draft?.id === activeId;

  const handleNew = () => {
    const def = EMPTY_DRAFT();
    upsertAgentDef(def);
    setSelectedId(def.id);
  };

  const handleDuplicate = () => {
    if (!draft) return;
    const copy: AgentDef = {
      ...draft,
      id: newAgentDefId(),
      name: `${draft.name} copy`,
      builtin: false,
    };
    upsertAgentDef(copy);
    setSelectedId(copy.id);
  };

  const handleSave = () => draft && upsertAgentDef(draft);

  const handleDelete = () => {
    if (!draft || draft.builtin) return;
    deleteAgentDef(draft.id);
    setSelectedId("vanilla");
  };

  return (
    <div className="grid gap-6 lg:grid-cols-[260px_minmax(0,1fr)]">
      {/* Definitions list */}
      <Card className="h-fit">
        <CardHeader className="flex-row items-center justify-between pb-2">
          <CardTitle>Agents</CardTitle>
          <Button
            className="h-7 gap-1.5 px-2 text-muted-foreground"
            onClick={handleNew}
            size="sm"
            variant="ghost"
          >
            <PlusIcon className="size-4" />
            New
          </Button>
        </CardHeader>
        <CardContent className="pb-2">
          <ScrollArea className="max-h-[440px]">
            <div className="flex flex-col gap-1 pr-2">
              {defs.map((def) => {
                const selected = def.id === selectedId;
                return (
                  <button
                    className={cn(
                      "flex items-center gap-2.5 rounded-lg px-2.5 py-2 text-left transition-colors",
                      selected ? "bg-muted" : "hover:bg-muted/60"
                    )}
                    key={def.id}
                    onClick={() => setSelectedId(def.id)}
                    type="button"
                  >
                    <span className="flex size-7 shrink-0 items-center justify-center rounded-md bg-primary/10 text-primary">
                      <BotIcon className="size-3.5" />
                    </span>
                    <span className="min-w-0 flex-1">
                      <span className="block truncate font-medium text-sm">
                        {def.name}
                      </span>
                      <span className="block truncate text-muted-foreground text-xs">
                        {def.model || "engine default"}
                      </span>
                    </span>
                    {def.id === activeId && (
                      <Badge
                        className="shrink-0 gap-1 rounded-full px-1.5 py-0 text-[10px]"
                        variant="secondary"
                      >
                        <CheckIcon className="size-2.5" />
                        Active
                      </Badge>
                    )}
                  </button>
                );
              })}
            </div>
          </ScrollArea>
        </CardContent>
      </Card>

      {/* Editor */}
      {draft && (
        <div className="flex flex-col gap-5">
          <div className="flex flex-wrap items-center justify-between gap-3">
            <div className="flex items-center gap-2">
              <h2 className="font-semibold text-lg">{draft.name || "Untitled"}</h2>
              {isActive && (
                <Badge className="gap-1 rounded-full" variant="secondary">
                  <CheckIcon className="size-3" />
                  Active
                </Badge>
              )}
              {draft.builtin && (
                <Badge className="rounded-full" variant="outline">
                  Built-in
                </Badge>
              )}
            </div>
            <div className="flex items-center gap-2">
              <Button
                className="gap-1.5"
                onClick={handleDuplicate}
                size="sm"
                variant="outline"
              >
                <CopyIcon className="size-4" />
                Duplicate
              </Button>
              {!isActive && (
                <Button
                  onClick={() => {
                    handleSave();
                    setActiveAgentDefId(draft.id);
                  }}
                  size="sm"
                  variant="secondary"
                >
                  Make active
                </Button>
              )}
            </div>
          </div>

          {draft.builtin ? (
            <Card>
              <CardContent className="py-6 text-muted-foreground text-sm">
                <span className="font-medium text-foreground">Vanilla</span> runs
                on the engine&apos;s configured defaults with no overrides.
                Duplicate it to create a custom agent with its own model, endpoint,
                and system prompt.
              </CardContent>
            </Card>
          ) : (
            <>
              {/* General */}
              <Card>
                <CardHeader>
                  <CardTitle>General</CardTitle>
                  <CardDescription>A name for this agent.</CardDescription>
                </CardHeader>
                <CardContent>
                  <Field htmlFor="def-name" label="Name">
                    <Input
                      id="def-name"
                      onChange={(e) =>
                        setDraft({ ...draft, name: e.target.value })
                      }
                      value={draft.name}
                    />
                  </Field>
                </CardContent>
              </Card>

              {/* LLM endpoint */}
              <Card>
                <CardHeader>
                  <CardTitle>LLM endpoint</CardTitle>
                  <CardDescription>
                    The model and provider this agent runs on. Leave a field blank
                    to fall back to the engine&apos;s configured default.
                  </CardDescription>
                </CardHeader>
                <CardContent className="grid gap-5">
                  <Field
                    description={
                      <>
                        e.g. <code>unsloth/Qwen3.8-27B-NVFP4</code> or{" "}
                        <code>z-ai/glm-5.2</code>.
                      </>
                    }
                    htmlFor="def-model"
                    label="Model id"
                  >
                    <Input
                      className="font-mono text-[13px]"
                      id="def-model"
                      onChange={(e) =>
                        setDraft({ ...draft, model: e.target.value })
                      }
                      placeholder="engine default"
                      value={draft.model}
                    />
                  </Field>
                  <Field
                    description="OpenAI-compatible base URL, e.g. https://openrouter.ai/api/v1."
                    htmlFor="def-base-url"
                    label="Base URL"
                  >
                    <Input
                      className="font-mono text-[13px]"
                      id="def-base-url"
                      onChange={(e) =>
                        setDraft({ ...draft, baseUrl: e.target.value })
                      }
                      placeholder="engine default"
                      value={draft.baseUrl}
                    />
                  </Field>
                  <Field
                    description="Stored in this browser and sent to the engine to run this agent. Leave blank to use the engine's key."
                    htmlFor="def-api-key"
                    label="API key"
                  >
                    <Input
                      autoComplete="off"
                      className="font-mono text-[13px]"
                      id="def-api-key"
                      onChange={(e) =>
                        setDraft({ ...draft, apiKey: e.target.value })
                      }
                      placeholder="engine default"
                      type="password"
                      value={draft.apiKey}
                    />
                  </Field>
                </CardContent>
              </Card>

              {/* Behavior */}
              <Card>
                <CardHeader>
                  <CardTitle>Behavior</CardTitle>
                  <CardDescription>
                    A system prompt that steers how the agent answers.
                  </CardDescription>
                </CardHeader>
                <CardContent>
                  <Field htmlFor="def-system" label="System prompt">
                    <Textarea
                      className="min-h-32 resize-y font-mono text-[13px] leading-relaxed"
                      id="def-system"
                      onChange={(e) =>
                        setDraft({ ...draft, system: e.target.value })
                      }
                      placeholder="engine default"
                      value={draft.system}
                    />
                  </Field>
                </CardContent>
              </Card>
            </>
          )}
        </div>
      )}

      {/* Sticky action bar (custom agents only) */}
      {draft && !draft.builtin && (
        <div className="sticky bottom-0 z-10 lg:col-start-2">
          <div className="flex items-center justify-between gap-3 rounded-xl border bg-background/80 px-4 py-3 shadow-[var(--shadow-float)] backdrop-blur">
            <span className="text-muted-foreground text-sm">
              {dirty ? "Unsaved changes" : "All changes saved"}
            </span>
            <div className="flex items-center gap-2">
              <Button
                className="gap-1.5 text-destructive hover:text-destructive"
                onClick={handleDelete}
                size="sm"
                variant="ghost"
              >
                <TrashIcon className="size-4" />
                Delete
              </Button>
              <Button disabled={!dirty} onClick={handleSave} size="sm">
                Save changes
              </Button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

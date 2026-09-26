import { z } from "zod";

const textPartSchema = z.object({
  text: z.string().min(1).max(2000),
  type: z.enum(["text"]),
});

const filePartSchema = z.object({
  mediaType: z.enum(["image/jpeg", "image/png"]),
  name: z.string().min(1).max(100),
  type: z.enum(["file"]),
  url: z.url(),
});

const partSchema = z.union([textPartSchema, filePartSchema]);

const userMessageSchema = z.object({
  id: z.uuid(),
  parts: z.array(partSchema),
  role: z.enum(["user"]),
});

const toolApprovalMessageSchema = z.object({
  id: z.string(),
  parts: z.array(z.record(z.string(), z.unknown())),
  role: z.enum(["user", "assistant"]),
});

export const postRequestBodySchema = z.object({
  id: z.uuid(),
  message: userMessageSchema.optional(),
  messages: z.array(toolApprovalMessageSchema).optional(),
  selectedChatModel: z.string(),
  selectedVisibilityType: z.enum(["public", "private"]),
  // Ambient: the engine video this chat is about, and the global run mode/track.
  videoId: z.string().optional(),
  ambientMode: z.enum(["agent", "fast"]).optional(),
  // Ambient: the active Studio agent definition (LLM setup). When it overrides
  // anything, the route creates/reuses an engine agent with that model + system
  // + endpoint; otherwise (Vanilla) it uses the engine's default agent.
  agentDef: z
    .object({
      id: z.string().optional(),
      model: z.string().optional(),
      system: z.string().optional(),
      baseUrl: z.string().optional(),
      apiKey: z.string().optional(),
    })
    .optional(),
});

export type PostRequestBody = z.infer<typeof postRequestBodySchema>;

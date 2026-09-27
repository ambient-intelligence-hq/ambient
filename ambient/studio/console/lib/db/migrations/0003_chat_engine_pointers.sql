-- Chat → engine pointers that used to live in Redis (see Chat.engineSessionId /
-- Chat.turnAnchor in schema.ts), so the Studio needs no Redis. IF NOT EXISTS so it
-- is safe where the columns were added ahead of the migrator.
ALTER TABLE "Chat" ADD COLUMN IF NOT EXISTS "engineSessionId" varchar(64);--> statement-breakpoint
ALTER TABLE "Chat" ADD COLUMN IF NOT EXISTS "turnAnchor" json;

-- The video each chat is about (see Chat.videoId in schema.ts). IF NOT EXISTS so
-- it is safe on databases where the column was added ahead of the migrator.
ALTER TABLE "Chat" ADD COLUMN IF NOT EXISTS "videoId" varchar(64);

-- Restored: this migration's SQL file was missing from the repo (its journal entry
-- and snapshot were present), which made `drizzle migrate` fail. It adds the
-- per-message metadata column (run usage + runComplete). IF NOT EXISTS because
-- existing dev databases already received the column outside the migrator.
ALTER TABLE "Message_v2" ADD COLUMN IF NOT EXISTS "metadata" json;

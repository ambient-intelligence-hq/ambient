import { config } from "dotenv";
import { drizzle } from "drizzle-orm/postgres-js";
import { migrate } from "drizzle-orm/postgres-js/migrator";
import postgres from "postgres";

config({
  path: ".env.local",
});

// Create the Studio's database if it doesn't exist yet. The Studio shares the
// engine's Postgres server but keeps its own database (e.g. ambient_studio), so a
// fresh install — or a fresh `docker compose up` — starts without it. Best-effort:
// if the role can't reach the `postgres` maintenance DB (e.g. some managed
// Postgres), skip and let the migration report a missing database clearly.
async function ensureDatabase(url: string) {
  const target = new URL(url);
  const name = decodeURIComponent(target.pathname.replace(/^\//, ""));
  if (!name || name === "postgres") {
    return;
  }
  const admin = new URL(url);
  admin.pathname = "/postgres";
  const sql = postgres(admin.toString(), { max: 1, onnotice: () => undefined });
  try {
    const rows = await sql`select 1 from pg_database where datname = ${name}`;
    if (rows.length === 0) {
      await sql.unsafe(`create database "${name.replaceAll('"', '""')}"`);
      console.log(`Created database ${name}`);
    }
  } catch (err) {
    console.warn(`Could not ensure database ${name} exists:`, (err as Error).message);
  } finally {
    await sql.end();
  }
}

const runMigrate = async () => {
  if (!process.env.POSTGRES_URL) {
    console.log("POSTGRES_URL not defined, skipping migrations");
    process.exit(0);
  }

  await ensureDatabase(process.env.POSTGRES_URL);

  const connection = postgres(process.env.POSTGRES_URL, { max: 1 });
  const db = drizzle(connection);

  console.log("Running migrations...");

  const start = Date.now();
  await migrate(db, { migrationsFolder: "./lib/db/migrations" });
  const end = Date.now();

  console.log("Migrations completed in", end - start, "ms");
  process.exit(0);
};

runMigrate().catch((err) => {
  console.error("Migration failed");
  console.error(err);
  process.exit(1);
});

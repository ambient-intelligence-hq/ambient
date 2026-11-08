import { genSaltSync, hashSync } from "bcrypt-ts";
import { config } from "dotenv";
import { eq } from "drizzle-orm";
import { drizzle } from "drizzle-orm/postgres-js";
import { migrate } from "drizzle-orm/postgres-js/migrator";
import postgres from "postgres";
import { user } from "./schema";

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
    console.warn(
      `Could not ensure database ${name} exists:`,
      (err as Error).message
    );
  } finally {
    await sql.end();
  }
}

// Create or update the account from STUDIO_ADMIN_EMAIL / STUDIO_ADMIN_PASSWORD on
// every start, so changing the password variable and restarting rotates it. This is
// how a closed install (STUDIO_ACCESS=closed) gets its first account.
async function ensureAdminUser(db: ReturnType<typeof drizzle>) {
  const email = process.env.STUDIO_ADMIN_EMAIL?.trim();
  const password = process.env.STUDIO_ADMIN_PASSWORD;

  if (!email || !password) {
    if (process.env.STUDIO_ACCESS === "closed") {
      console.warn(
        "STUDIO_ACCESS=closed without STUDIO_ADMIN_EMAIL and STUDIO_ADMIN_PASSWORD: no admin account was set up."
      );
    }
    return;
  }
  if (password.length < 12) {
    throw new Error("STUDIO_ADMIN_PASSWORD must be at least 12 characters");
  }

  const hashed = hashSync(password, genSaltSync(10));
  const updated = await db
    .update(user)
    .set({ password: hashed })
    .where(eq(user.email, email))
    .returning({ id: user.id });
  if (updated.length === 0) {
    await db.insert(user).values({ email, password: hashed });
  }
  console.log("Admin account ready");
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

  await ensureAdminUser(db);
  process.exit(0);
};

runMigrate().catch((err) => {
  console.error("Migration failed");
  console.error(err);
  process.exit(1);
});

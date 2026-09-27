# syntax=docker/dockerfile:1.9
#
# Ambient Studio (Next.js console) image — published as
# ghcr.io/ambient-intelligence-hq/ambient-studio. Build context: this directory.
#
# Runtime config comes from the environment (see .env.example): POSTGRES_URL,
# AUTH_SECRET, ENGINE_URL, ENGINE_API_KEY. On start it creates/migrates its
# database, then serves on :3000. Liveness: GET /ping -> "pong".

############################  base  ############################
FROM node:24-bookworm-slim AS base
ENV NEXT_TELEMETRY_DISABLED=1 \
    HUSKY=0
# pnpm pinned to the version in package.json's `packageManager`.
RUN npm install -g pnpm@10.32.1

############################  deps  ############################
FROM base AS deps
WORKDIR /app
COPY package.json pnpm-lock.yaml ./
RUN --mount=type=cache,id=pnpm-store,target=/root/.local/share/pnpm/store \
    pnpm install --frozen-lockfile

############################  build  ###########################
FROM base AS build
WORKDIR /app
COPY --from=deps /app/node_modules ./node_modules
COPY . .
# `next build` directly — `pnpm build` would first migrate a database, and
# there is none at image-build time (migrations run when the container starts).
RUN pnpm exec next build
# Bundle the migration script (drizzle + postgres inlined) into one plain-JS file
# so the runtime image needs neither tsx nor the full node_modules to run it.
# esbuild is already in the lockfile (via tsx).
RUN ESBUILD="$(find node_modules/.pnpm -path '*/node_modules/esbuild/bin/esbuild' -type f | head -1)" \
    && "$ESBUILD" lib/db/migrate.ts --bundle --platform=node --format=cjs \
         --outfile=migrate.cjs --log-level=warning

############################  runtime  #########################
FROM node:24-bookworm-slim AS runtime
LABEL org.opencontainers.image.source="https://github.com/ambient-intelligence-hq/ambient" \
      org.opencontainers.image.description="Ambient Studio — web UI for the Ambient engine"
WORKDIR /app
ENV NODE_ENV=production \
    NEXT_TELEMETRY_DISABLED=1 \
    PORT=3000 \
    HOSTNAME=0.0.0.0

COPY --from=build --chown=node:node /app/.next/standalone ./
COPY --from=build --chown=node:node /app/.next/static ./.next/static
COPY --from=build --chown=node:node /app/public ./public
COPY --from=build --chown=node:node /app/migrate.cjs ./migrate.cjs
COPY --from=build --chown=node:node /app/lib/db/migrations ./lib/db/migrations

USER node
EXPOSE 3000
# Create + migrate the database, then serve. `exec` hands PID 1 to the server so
# `docker stop` shuts it down cleanly.
CMD ["sh", "-c", "node migrate.cjs && exec node server.js"]

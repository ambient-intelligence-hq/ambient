#!/usr/bin/env bash
#
# Ambient installer — a video-understanding agent (engine + Studio) on Docker.
#
#   curl -fsSL https://ambient-intelligence-hq.github.io/ambient/install.sh | bash
#
# Needs only Docker (Docker Desktop on macOS/Windows, Docker Engine + the compose
# plugin on Linux). Writes ~/.ambient/{docker-compose.yml,.env,bin/ambientctl},
# pulls the images, starts the stack, and opens the Studio.
#
# Re-running it is safe: it keeps your generated secrets and settings, refreshes
# docker-compose.yml + ambientctl, pulls newer images, and restarts.
#
# Non-interactive:
#   curl -fsSL https://ambient-intelligence-hq.github.io/ambient/install.sh | AMBIENT_LLM_KEY=sk-or-... bash -s -- --yes
#
# Options:
#   -y, --yes          accept defaults; don't prompt (key from AMBIENT_LLM_KEY)
#   --dir DIR          install directory (default: ~/.ambient)
#   --version TAG      image tag (default: latest)
#   --no-start         write the config only; don't pull or start anything
#   --no-open          don't open the browser at the end
#   -h, --help         show this help
#
# Environment (all optional):
#   AMBIENT_LLM_KEY        LLM API key (OpenRouter by default)
#   AMBIENT_LLM_BASE_URL   OpenAI-compatible base URL (default: OpenRouter)
#   AMBIENT_MODEL          model that plans + answers (the agent; default below)
#   AMBIENT_VIDEO_MODEL    model that watches the clips (must accept video)
#   AMBIENT_BIND           address the ports bind to (default: 127.0.0.1 — this
#                          machine only; use 0.0.0.0 on a server behind a proxy)
#   AMBIENT_STUDIO_PORT    Studio port (default: 3000)
#   AMBIENT_ENGINE_PORT    engine/API port (default: 8080)
#   AMBIENT_PROJECT        docker compose project name (default: ambient)
#   AMBIENT_ENGINE_IMAGE / AMBIENT_STUDIO_IMAGE   image repos (without tag)
#   AMBIENT_SKIP_PULL=1    use images already present locally

set -euo pipefail

# Everything runs inside main(), called on the last line — so a download that is
# cut off halfway through `curl | bash` never executes a partial script.

DEFAULT_BASE_URL="https://openrouter.ai/api/v1"
DEFAULT_MODEL="qwen/qwen3.8-27b"
DEFAULT_VIDEO_MODEL="qwen/qwen3.8-27b"
DEFAULT_ENGINE_IMAGE="ghcr.io/ambient-intelligence-hq/ambient-engine"
DEFAULT_STUDIO_IMAGE="ghcr.io/ambient-intelligence-hq/ambient-studio"

# ---------------------------------------------------------------- output ----
if [ -t 1 ]; then
  BOLD=$'\033[1m'; DIM=$'\033[2m'; RED=$'\033[31m'; GREEN=$'\033[32m'; YELLOW=$'\033[33m'; RESET=$'\033[0m'
else
  BOLD=""; DIM=""; RED=""; GREEN=""; YELLOW=""; RESET=""
fi
say()  { printf '%s\n' "$*"; }
step() { printf '\n%s==>%s %s%s%s\n' "$GREEN" "$RESET" "$BOLD" "$*" "$RESET"; }
ok()   { printf '  %s✓%s %s\n' "$GREEN" "$RESET" "$*"; }
warn() { printf '  %s!%s %s\n' "$YELLOW" "$RESET" "$*" >&2; }
die()  { printf '\n%serror:%s %s\n' "$RED" "$RESET" "$*" >&2; exit 1; }

usage() {
  if [ -f "$0" ]; then
    sed -n '2,/^set -euo/p' "$0" | sed -e '/^set -euo/d' -e 's/^# \{0,1\}//'
  else  # piped (`curl | bash -s -- --help`): no file to read the header from
    say "usage: install.sh [--yes] [--dir DIR] [--version TAG] [--no-start] [--no-open]"
    say "env:   AMBIENT_LLM_KEY AMBIENT_LLM_BASE_URL AMBIENT_MODEL AMBIENT_VIDEO_MODEL"
    say "       AMBIENT_BIND AMBIENT_STUDIO_PORT AMBIENT_ENGINE_PORT AMBIENT_PROJECT"
  fi
}

# ----------------------------------------------------------------- input ----
HAVE_TTY=0
if (exec </dev/tty) 2>/dev/null; then HAVE_TTY=1; fi

# ask VAR "Question" "default"  — reads from the terminal (stdin is the script)
ask() {
  local __var="$1" __prompt="$2" __default="${3:-}" __reply=""
  if [ "$HAVE_TTY" = 1 ]; then
    if [ -n "$__default" ]; then
      printf '  %s %s[%s]%s: ' "$__prompt" "$DIM" "$__default" "$RESET" >/dev/tty
    else
      printf '  %s: ' "$__prompt" >/dev/tty
    fi
    IFS= read -r __reply </dev/tty || true
  fi
  printf -v "$__var" '%s' "${__reply:-$__default}"
}

# ask_secret VAR "Question"  — hidden input; empty = skip
ask_secret() {
  local __var="$1" __prompt="$2" __reply=""
  if [ "$HAVE_TTY" = 1 ]; then
    printf '  %s: ' "$__prompt" >/dev/tty
    IFS= read -rs __reply </dev/tty || true
    printf '\n' >/dev/tty
  fi
  printf -v "$__var" '%s' "$__reply"
}

# ---------------------------------------------------------------- helpers ---
gen_secret() {
  if command -v openssl >/dev/null 2>&1; then
    openssl rand -hex 32
  else
    head -c 32 /dev/urandom | od -An -tx1 | tr -d ' \n'
  fi
}

# Value of KEY in an env file ("" if absent). Handles KEY=value and KEY='value'.
env_get() {
  local file="$1" key="$2" line
  [ -f "$file" ] || return 0
  line="$(grep -E "^${key}=" "$file" | tail -1 || true)"
  line="${line#*=}"
  line="${line#\'}"; line="${line%\'}"
  line="${line#\"}"; line="${line%\"}"
  printf '%s' "$line"
}

port_busy() { (: </dev/tcp/127.0.0.1/"$1") >/dev/null 2>&1; }

# resolve_port LABEL PORT EXPLICIT [TAKEN] — echoes a usable port. An explicitly
# requested port that's busy is an error; otherwise scan upward for a free one
# (skipping TAKEN, the port already picked for the other service).
resolve_port() {
  local label="$1" port="$2" explicit="$3" taken="${4:-}" p
  if ! port_busy "$port" && [ "$port" != "$taken" ]; then
    printf '%s' "$port"; return 0
  fi
  if [ -n "$explicit" ]; then
    printf '\n%serror:%s %s port %s is already in use. Free it, or pick another (or unset it to auto-pick).\n' \
      "$RED" "$RESET" "$label" "$port" >&2
    return 1
  fi
  for p in $(seq $((port + 1)) $((port + 100))); do
    if [ "$p" != "$taken" ] && ! port_busy "$p"; then
      warn "$label port $port is in use — using $p instead"
      printf '%s' "$p"; return 0
    fi
  done
  printf '\n%serror:%s no free %s port in %s-%s. Set AMBIENT_%s_PORT.\n' \
    "$RED" "$RESET" "$label" "$port" $((port + 100)) \
    "$([ "$label" = Studio ] && echo STUDIO || echo ENGINE)" >&2
  return 1
}

open_url() {
  if command -v open >/dev/null 2>&1; then open "$1" >/dev/null 2>&1 || true
  elif command -v xdg-open >/dev/null 2>&1; then xdg-open "$1" >/dev/null 2>&1 || true
  fi
}

# ------------------------------------------------------------- templates ----
write_compose() {
  cat >"$1" <<'COMPOSE'
# Ambient — written by install.sh. Settings live in .env (see `ambientctl config`);
# re-running the installer regenerates this file, so don't edit it by hand.

services:
  postgres:
    image: postgres:16-alpine
    restart: unless-stopped
    environment:
      POSTGRES_USER: ambient
      POSTGRES_PASSWORD: ${POSTGRES_PASSWORD:?run the installer to generate .env}
      POSTGRES_DB: ambient
    volumes:
      - postgres:/var/lib/postgresql/data
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U ambient -d ambient"]
      interval: 5s
      timeout: 5s
      retries: 30

  redis:
    image: redis:7-alpine
    restart: unless-stopped
    # Append-only so queued video-description jobs survive a restart.
    command: ["redis-server", "--appendonly", "yes"]
    volumes:
      - redis:/data
    healthcheck:
      test: ["CMD", "redis-cli", "ping"]
      interval: 5s
      timeout: 3s
      retries: 30

  engine:
    image: ${AMBIENT_ENGINE_IMAGE}:${AMBIENT_VERSION}
    restart: unless-stopped
    depends_on:
      postgres: { condition: service_healthy }
      redis: { condition: service_healthy }
    environment:
      DATABASE_URL: postgresql://ambient:${POSTGRES_PASSWORD}@postgres:5432/ambient
      REDIS_URL: redis://redis:6379/0
      API_KEY: ${ENGINE_API_KEY}
      LLM_BASE_URL: ${LLM_BASE_URL}
      LLM_API_KEY: ${LLM_API_KEY}
      AGENT_MODEL: ${AGENT_MODEL}
      LLM_MODEL: ${LLM_MODEL}
      # Media runs on this machine with ffmpeg (no cloud sandbox, no S3).
      SANDBOX_BACKEND: inprocess
      VIDEO_FOLDER: /data/videos
      YOUTUBE_IMPORT_ENABLED: "true"
      ENABLE_BASH_TOOL: ${ENABLE_BASH_TOOL}
    volumes:
      - videos:/data
    ports:
      - "${AMBIENT_BIND}:${AMBIENT_ENGINE_PORT}:8080"
    healthcheck:
      test: ["CMD", "python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8080/v1/healthz', timeout=3)"]
      interval: 10s
      timeout: 5s
      start_period: 60s
      retries: 12

  studio:
    image: ${AMBIENT_STUDIO_IMAGE}:${AMBIENT_VERSION}
    restart: unless-stopped
    depends_on:
      postgres: { condition: service_healthy }
      engine: { condition: service_healthy }
    environment:
      # Its own database on the same server; created on first start.
      POSTGRES_URL: postgresql://ambient:${POSTGRES_PASSWORD}@postgres:5432/ambient_studio
      AUTH_SECRET: ${AUTH_SECRET}
      ENGINE_URL: http://engine:8080
      ENGINE_API_KEY: ${ENGINE_API_KEY}
    ports:
      - "${AMBIENT_BIND}:${AMBIENT_STUDIO_PORT}:3000"
    healthcheck:
      test: ["CMD", "node", "-e", "fetch('http://127.0.0.1:3000/ping').then(r=>process.exit(r.ok?0:1)).catch(()=>process.exit(1))"]
      interval: 10s
      timeout: 5s
      start_period: 60s
      retries: 12

volumes:
  postgres:
  redis:
  videos:
COMPOSE
}

write_ambientctl() {
  local target="$1" dir="$2"
  {
    printf '#!/usr/bin/env bash\n'
    printf '# ambientctl — manage the Ambient install in %s (written by install.sh).\n' "$dir"
    printf 'AMBIENT_DIR="${AMBIENT_DIR:-%s}"\n' "$dir"
    cat <<'CTL'
set -euo pipefail

cd "$AMBIENT_DIR" 2>/dev/null || { echo "ambientctl: no install at $AMBIENT_DIR" >&2; exit 1; }
ENV_FILE="$AMBIENT_DIR/.env"
dc() { docker compose "$@"; }

env_get() {
  local line
  line="$(grep -E "^$1=" "$ENV_FILE" | tail -1 || true)"
  line="${line#*=}"; line="${line#\'}"; line="${line%\'}"
  printf '%s' "$line"
}

env_set() {  # env_set KEY VALUE — replace or append, keeping the file private
  local key="$1" value="$2" tmp
  tmp="$(mktemp "$AMBIENT_DIR/.env.XXXXXX")"
  if grep -qE "^${key}=" "$ENV_FILE"; then
    awk -v k="$key" -v v="$value" 'BEGIN{FS=OFS="="} $1==k {print k"="v; next} {print}' "$ENV_FILE" >"$tmp"
  else
    cat "$ENV_FILE" >"$tmp"; printf '%s=%s\n' "$key" "$value" >>"$tmp"
  fi
  chmod 600 "$tmp"; mv "$tmp" "$ENV_FILE"
}

urls() {
  local bind port_s port_e host
  bind="$(env_get AMBIENT_BIND)"; port_s="$(env_get AMBIENT_STUDIO_PORT)"; port_e="$(env_get AMBIENT_ENGINE_PORT)"
  host="localhost"; [ "$bind" = "0.0.0.0" ] && host="<this-host>"
  echo "Studio:  http://$host:$port_s"
  echo "API:     http://$host:$port_e   (Anthropic SDK base_url; key: ambientctl config get ENGINE_API_KEY)"
}

redact() { case "$1" in *KEY*|*SECRET*|*PASSWORD*) local v="$2"; [ -n "$v" ] && echo "${v:0:6}…" || echo "(not set)";; *) echo "$2";; esac; }

cmd="${1:-help}"; shift || true
case "$cmd" in
  status)
    dc ps
    echo; urls ;;
  logs)
    dc logs -f --tail=200 "$@" ;;
  start)
    dc up -d --wait
    urls ;;
  stop)
    dc stop ;;
  restart)
    dc restart "$@" ;;
  update)
    # Newer images for the current tag, then recreate what changed. The Studio
    # migrates its database on start. (To pick up a newer stack definition,
    # re-run the install command — it keeps your settings.)
    dc pull
    dc up -d --wait
    urls ;;
  config)
    sub="${1:-show}"; shift || true
    case "$sub" in
      show)
        while IFS= read -r line; do
          case "$line" in ''|\#*) continue;; esac
          k="${line%%=*}"; v="${line#*=}"; v="${v#\'}"; v="${v%\'}"
          printf '%-22s %s\n' "$k" "$(redact "$k" "$v")"
        done <"$ENV_FILE" ;;
      get)
        [ $# -eq 1 ] || { echo "usage: ambientctl config get KEY" >&2; exit 2; }
        env_get "$1"; echo ;;
      set)
        [ $# -eq 2 ] || { echo "usage: ambientctl config set KEY VALUE" >&2; exit 2; }
        env_set "$1" "$2"
        echo "Updated $1. Applying…"
        dc up -d --wait ;;
      *) echo "usage: ambientctl config [show | get KEY | set KEY VALUE]" >&2; exit 2 ;;
    esac ;;
  open)
    url="http://localhost:$(env_get AMBIENT_STUDIO_PORT)"
    if command -v open >/dev/null 2>&1; then open "$url"; elif command -v xdg-open >/dev/null 2>&1; then xdg-open "$url"; else echo "$url"; fi ;;
  uninstall)
    if [ "${1:-}" = "--purge" ]; then
      printf 'This deletes all Ambient data (videos, chats, database) in Docker volumes and %s. Type "delete" to confirm: ' "$AMBIENT_DIR"
      read -r answer
      [ "$answer" = "delete" ] || { echo "Aborted."; exit 1; }
      dc down -v --remove-orphans
      cd /; rm -rf "$AMBIENT_DIR"
      rm -f "$HOME/.local/bin/ambientctl"
      echo "Ambient removed, including its data."
    else
      dc down --remove-orphans
      echo "Stopped and removed the containers. Your data is kept in Docker volumes;"
      echo "'ambientctl start' brings everything back, 'ambientctl uninstall --purge' deletes it."
    fi ;;
  version)
    echo "tag: $(env_get AMBIENT_VERSION)"
    dc images ;;
  help|-h|--help)
    cat <<USAGE
ambientctl — manage your Ambient install ($AMBIENT_DIR)

  status                 containers + URLs
  logs [service]         follow logs (engine, studio, postgres, redis)
  start | stop           start (waits until healthy) / stop everything
  restart [service]      restart one or all services
  update                 pull newer images for this tag and restart
  config show            current settings (secrets redacted)
  config get KEY         print one setting (e.g. ENGINE_API_KEY)
  config set KEY VALUE   change a setting and apply it
                         (LLM_API_KEY, LLM_BASE_URL, AGENT_MODEL, LLM_MODEL, …)
  open                   open the Studio in your browser
  version                image versions
  uninstall [--purge]    remove containers (keep data) / everything incl. data
USAGE
    ;;
  *) echo "ambientctl: unknown command '$cmd' (see: ambientctl help)" >&2; exit 2 ;;
esac
CTL
  } >"$target"
  chmod 755 "$target"
}

# ------------------------------------------------------------------ main ----
main() {
  local assume_yes=0 no_start=0 no_open=0
  local dir="${AMBIENT_DIR:-$HOME/.ambient}"
  local version="${AMBIENT_VERSION:-}"

  while [ $# -gt 0 ]; do
    case "$1" in
      -y|--yes) assume_yes=1 ;;
      --dir) shift; dir="${1:?--dir needs a path}" ;;
      --version) shift; version="${1:?--version needs a tag}" ;;
      --no-start) no_start=1 ;;
      --no-open) no_open=1 ;;
      -h|--help) usage; exit 0 ;;
      *) die "unknown option: $1 (see --help)" ;;
    esac
    shift
  done
  [ "$assume_yes" = 1 ] && HAVE_TTY=0

  say "${BOLD}Ambient${RESET} — video-understanding agent (engine + Studio)"

  # -- prerequisites ---------------------------------------------------------
  step "Checking prerequisites"
  command -v docker >/dev/null 2>&1 || die "Docker is not installed.
  macOS / Windows: install Docker Desktop — https://www.docker.com/products/docker-desktop/
  Linux:           curl -fsSL https://get.docker.com | sh"
  docker info >/dev/null 2>&1 || die "Docker is installed but not running (or you lack permission).
  Start Docker Desktop, or on Linux: sudo systemctl start docker
  (and add yourself to the docker group: sudo usermod -aG docker \$USER)"
  docker compose version >/dev/null 2>&1 || die "Docker Compose v2 ('docker compose') is required.
  It ships with Docker Desktop; on Linux install the docker-compose-plugin package."
  ok "Docker $(docker version --format '{{.Server.Version}}' 2>/dev/null || echo '?'), $(docker compose version --short 2>/dev/null || echo 'compose v2')"

  mkdir -p "$dir/bin"
  dir="$(cd "$dir" && pwd)"
  local env_file="$dir/.env"
  local existing=0; [ -f "$env_file" ] && existing=1

  # -- configuration ---------------------------------------------------------
  # Keep everything from a previous install (secrets above all: a new database
  # password would lock you out of your existing data); ask only for what's new.
  local project bind port_s port_e engine_image studio_image
  project="${AMBIENT_PROJECT:-$(env_get "$env_file" COMPOSE_PROJECT_NAME)}"; project="${project:-ambient}"
  bind="${AMBIENT_BIND:-$(env_get "$env_file" AMBIENT_BIND)}"; bind="${bind:-127.0.0.1}"
  port_s="${AMBIENT_STUDIO_PORT:-$(env_get "$env_file" AMBIENT_STUDIO_PORT)}"; port_s="${port_s:-3000}"
  port_e="${AMBIENT_ENGINE_PORT:-$(env_get "$env_file" AMBIENT_ENGINE_PORT)}"; port_e="${port_e:-8080}"
  engine_image="${AMBIENT_ENGINE_IMAGE:-$(env_get "$env_file" AMBIENT_ENGINE_IMAGE)}"; engine_image="${engine_image:-$DEFAULT_ENGINE_IMAGE}"
  studio_image="${AMBIENT_STUDIO_IMAGE:-$(env_get "$env_file" AMBIENT_STUDIO_IMAGE)}"; studio_image="${studio_image:-$DEFAULT_STUDIO_IMAGE}"
  version="${version:-$(env_get "$env_file" AMBIENT_VERSION)}"; version="${version:-latest}"

  local base_url model video_model llm_key bash_tool pg_pass engine_key auth_secret
  base_url="${AMBIENT_LLM_BASE_URL:-$(env_get "$env_file" LLM_BASE_URL)}"
  model="${AMBIENT_MODEL:-$(env_get "$env_file" AGENT_MODEL)}"
  video_model="${AMBIENT_VIDEO_MODEL:-$(env_get "$env_file" LLM_MODEL)}"
  llm_key="${AMBIENT_LLM_KEY:-$(env_get "$env_file" LLM_API_KEY)}"
  bash_tool="$(env_get "$env_file" ENABLE_BASH_TOOL)"; bash_tool="${bash_tool:-true}"
  pg_pass="$(env_get "$env_file" POSTGRES_PASSWORD)"; pg_pass="${pg_pass:-$(gen_secret)}"
  engine_key="$(env_get "$env_file" ENGINE_API_KEY)"; engine_key="${engine_key:-$(gen_secret)}"
  auth_secret="$(env_get "$env_file" AUTH_SECRET)"; auth_secret="${auth_secret:-$(gen_secret)}"

  # -- ports -----------------------------------------------------------------
  # Checked before any prompt, so nothing typed is wasted. A port you asked for
  # (AMBIENT_*_PORT) must be free; a default or previously saved one that's taken
  # is swapped for the next free port. A running install already holds its own
  # ports, so they aren't checked.
  local running=0
  if [ "$existing" = 1 ] && (cd "$dir" && docker compose ps -q 2>/dev/null | grep -q .); then
    running=1
  fi
  if [ "$no_start" = 0 ] && [ "$running" = 0 ]; then
    port_s="$(resolve_port "Studio" "$port_s" "${AMBIENT_STUDIO_PORT:-}")" || exit 1
    port_e="$(resolve_port "API" "$port_e" "${AMBIENT_ENGINE_PORT:-}" "$port_s")" || exit 1
    ok "Ports: Studio $port_s, API $port_e"
  fi

  step "Configuring"
  if [ "$existing" = 1 ]; then
    ok "Existing install in $dir — keeping its settings and secrets"
  fi
  if [ -z "$base_url" ]; then
    ask base_url "LLM endpoint (OpenAI-compatible)" "$DEFAULT_BASE_URL"
  fi
  if [ -z "$model" ]; then
    ask model "Agent model (plans + answers)" "$DEFAULT_MODEL"
  fi
  if [ -z "$video_model" ]; then
    ask video_model "Video model (watches the clips)" "$DEFAULT_VIDEO_MODEL"
  fi
  if [ -z "$llm_key" ]; then
    if [ "$HAVE_TTY" = 1 ]; then
      say "  An API key for $base_url (OpenRouter keys: https://openrouter.ai/keys)."
      ask_secret llm_key "API key (leave empty to add later)"
    fi
  fi
  base_url="${base_url:-$DEFAULT_BASE_URL}"
  model="${model:-$DEFAULT_MODEL}"
  video_model="${video_model:-$DEFAULT_VIDEO_MODEL}"
  ok "LLM: $base_url · agent $model · video $video_model · key $([ -n "$llm_key" ] && echo set || echo "${YELLOW}not set${RESET}")"


  # -- files -----------------------------------------------------------------
  step "Writing $dir"
  write_compose "$dir/docker-compose.yml"
  ok "docker-compose.yml"

  local tmp_env
  tmp_env="$(mktemp "$dir/.env.XXXXXX")"
  chmod 600 "$tmp_env"
  cat >"$tmp_env" <<ENV
# Ambient settings — written by install.sh.
# Change them with:  ambientctl config set KEY VALUE   (applies immediately)
COMPOSE_PROJECT_NAME=$project
AMBIENT_VERSION=$version
AMBIENT_ENGINE_IMAGE=$engine_image
AMBIENT_STUDIO_IMAGE=$studio_image
# 127.0.0.1 = reachable from this machine only.
AMBIENT_BIND=$bind
AMBIENT_STUDIO_PORT=$port_s
AMBIENT_ENGINE_PORT=$port_e

# LLM — any OpenAI-compatible chat-completions endpoint.
LLM_BASE_URL=$base_url
LLM_API_KEY=$llm_key
# Agent: plans and answers. Video model: watches clips (must accept video input).
AGENT_MODEL=$model
LLM_MODEL=$video_model

# Lets the agent run shell commands inside the engine container.
ENABLE_BASH_TOOL=$bash_tool

# Generated secrets. Keep them: a different POSTGRES_PASSWORD locks you out of
# the existing database. ENGINE_API_KEY is the API key for the Anthropic SDK.
POSTGRES_PASSWORD=$pg_pass
ENGINE_API_KEY=$engine_key
AUTH_SECRET=$auth_secret
ENV
  mv "$tmp_env" "$env_file"
  ok ".env (private: only you can read it)"

  write_ambientctl "$dir/bin/ambientctl" "$dir"
  local ctl="$dir/bin/ambientctl" ctl_hint="$dir/bin/ambientctl"
  mkdir -p "$HOME/.local/bin" 2>/dev/null || true
  if ln -sf "$ctl" "$HOME/.local/bin/ambientctl" 2>/dev/null; then
    case ":$PATH:" in
      *":$HOME/.local/bin:"*) ctl_hint="ambientctl" ;;
    esac
  fi
  ok "ambientctl"

  if [ "$no_start" = 1 ]; then
    say ""; say "Config written. Start with: $ctl_hint start"
    return 0
  fi

  # -- start -----------------------------------------------------------------
  cd "$dir"
  if [ "${AMBIENT_SKIP_PULL:-0}" != 1 ]; then
    step "Pulling images (first time: a few minutes)"
    local pull_log
    pull_log="$(mktemp)"
    if ! docker compose pull 2>&1 | tee "$pull_log"; then
      # GHCR answers "unauthorized"/"denied" both for a private package and for
      # an image or tag that doesn't exist, so name the exact refs.
      if grep -qiE 'unauthorized|denied|manifest unknown|not found' "$pull_log"; then
        rm -f "$pull_log"
        die "The registry refused these images:
    $engine_image:$version
    $studio_image:$version
  Either the tag doesn't exist, or the images aren't public. Try a published
  tag (--version <tag>), or log in first: docker login ghcr.io"
      fi
      rm -f "$pull_log"
      die "Could not pull the images. Check your connection, then re-run the installer."
    fi
    rm -f "$pull_log"
  fi

  step "Starting Ambient"
  if ! docker compose up -d --wait; then
    warn "Something didn't come up healthy. Recent logs:"
    docker compose ps >&2 || true
    docker compose logs --tail=40 >&2 || true
    die "Start failed. Fix the issue above, then run: $ctl_hint start"
  fi
  ok "All services healthy"

  local host="localhost"; [ "$bind" = "0.0.0.0" ] && host="<this-host>"
  local studio_url="http://$host:$port_s"

  say ""
  say "${GREEN}${BOLD}Ambient is running.${RESET}"
  say ""
  say "  Studio   $studio_url"
  say "  API      http://$host:$port_e   (Anthropic SDK base_url)"
  say "  API key  $ctl_hint config get ENGINE_API_KEY"
  say "  Data     $dir  +  Docker volumes (${project}_videos, ${project}_postgres)"
  if [ -z "$llm_key" ]; then
    say ""
    warn "No LLM API key yet — add one before asking questions:"
    say "       $ctl_hint config set LLM_API_KEY <your-key>"
  fi
  say ""
  say "  Manage:  $ctl_hint status | logs | stop | start | update | config | uninstall"
  if [ "$ctl_hint" != "ambientctl" ]; then
    say "  ${DIM}(add ~/.local/bin to your PATH to run it as just 'ambientctl')${RESET}"
  fi

  if [ "$no_open" = 0 ] && [ "$HAVE_TTY" = 1 ] && [ "$bind" != "0.0.0.0" ]; then
    open_url "$studio_url"
  fi
}

main "$@"

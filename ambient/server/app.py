from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from ambient.server.broker import Broker
from ambient.server.ingest import IngestWorker
from ambient.server.routes import  health, managed_agents
from ambient.server.routes.managed_agents import seed_defaults
from ambient.server.store import Store
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
log = logging.getLogger(__name__)

@asynccontextmanager
async def _lifespan(app: FastAPI):
    # Shared, durable state: Postgres (sessions/events/metadata) + Redis (run
    # ownership lease + SSE fan-out). Agents/environments/files now live in the
    # store, not in-memory, so any worker / replica can serve any request.
    store = Store()
    await store.connect()
    await seed_defaults(store)
    broker = Broker()
    await broker.connect()
    # Background video-description ingestion (Redis stream consumer + sweeper).
    ingest = IngestWorker(store=store)
    await ingest.start()
    app.state.store = store
    app.state.broker = broker
    app.state.ingest = ingest
    # Process-local cache of live SessionRunner objects. NOT the source of truth:
    # ownership is the Redis lease, state is Postgres. A runner missing here is
    # rehydrated on demand.
    app.state.runners = {}  # session_id -> SessionRunner
    try:
        yield
    finally:
        for runner in list(app.state.runners.values()):
            await runner.terminate()
        await ingest.stop()
        await broker.close()
        await store.close()


app = FastAPI(title="Ambient Agent API", version="0.1.0", lifespan=_lifespan)
app.include_router(health.router)
# Anthropic Managed Agents protocol (the Anthropic SDK talks to these routes).
app.include_router(managed_agents.router)

# Single-page web UI (demo/web). Served same-origin so its fetch/SSE calls hit
# the /v1 routes without CORS, and the api key travels as a normal header.
# Mounted last so it never shadows the /v1 API routes; `html=True` serves
# index.html for "/".
_WEB_DIR = Path(__file__).resolve().parents[2] / "demo" / "web"
if (_WEB_DIR / "index.html").exists():
    app.mount("/", StaticFiles(directory=str(_WEB_DIR), html=True), name="web")
"""Redis-backed cross-process coordination for the managed-agents server.

Two responsibilities, both keyed by session id:

  1. **Run ownership lease.** A run (the SessionRunner loop) must execute in
     exactly one worker at a time. `claim` takes the lease with `SET NX PX`;
     `renew` extends it while a run is in flight; `release` drops it (compare-and-
     delete so we only release our own lease). A dead worker's lease self-expires
     after `lease_ms`, so a session never locks permanently.

  2. **Event fan-out.** The owning worker `publish`es each runner event to a
     per-session channel. Any worker's SSE handler `subscribe`s to it, so a stream
     opened on worker B receives events from a run executing on worker A.
"""
from __future__ import annotations

import json
import uuid
from typing import Any, AsyncIterator, Optional

import redis.asyncio as aioredis

from ambient.config import settings

# Release the lease only if we still own it (avoids dropping a lease a later owner
# acquired after ours expired).
_RELEASE_LUA = """
if redis.call('get', KEYS[1]) == ARGV[1] then
    return redis.call('del', KEYS[1])
else
    return 0
end
"""


def _owner_key(session_id: str) -> str:
    return f"session:{session_id}:owner"


def _events_channel(session_id: str) -> str:
    return f"session:{session_id}:events"


class Broker:
    def __init__(self, url: Optional[str] = None, lease_ms: int = 600_000) -> None:
        self.url = url or settings.redis_url
        self.lease_ms = lease_ms
        # Stable per-process identity stamped on every lease we hold.
        self.worker_id = settings.worker_id or uuid.uuid4().hex
        self._redis: Optional[aioredis.Redis] = None
        self._release = None  # registered Lua script

    async def connect(self) -> None:
        self._redis = aioredis.from_url(self.url, decode_responses=True)
        self._release = self._redis.register_script(_RELEASE_LUA)

    async def close(self) -> None:
        if self._redis is not None:
            await self._redis.aclose()
            self._redis = None

    # --- ownership lease --------------------------------------------------

    async def claim(self, session_id: str) -> bool:
        """Take the run-ownership lease. True if acquired (no run in flight)."""
        ok = await self._redis.set(
            _owner_key(session_id), self.worker_id, nx=True, px=self.lease_ms
        )
        return bool(ok)

    async def renew(self, session_id: str) -> None:
        """Extend our lease (best-effort; only if we still hold it)."""
        # SET XX keeps the key only if it exists; re-stamp value + TTL.
        await self._redis.set(
            _owner_key(session_id), self.worker_id, xx=True, px=self.lease_ms
        )

    async def release(self, session_id: str) -> None:
        await self._release(keys=[_owner_key(session_id)], args=[self.worker_id])

    async def owner(self, session_id: str) -> Optional[str]:
        return await self._redis.get(_owner_key(session_id))

    # --- event fan-out ----------------------------------------------------

    async def publish(self, session_id: str, event: dict[str, Any]) -> None:
        await self._redis.publish(_events_channel(session_id), json.dumps(event))

    async def subscribe(self, session_id: str) -> "BrokerSubscription":
        pubsub = self._redis.pubsub()
        await pubsub.subscribe(_events_channel(session_id))
        return BrokerSubscription(pubsub)


class BrokerSubscription:
    """Async iterator over a session's published events. Use as a context manager."""

    def __init__(self, pubsub) -> None:
        self._pubsub = pubsub

    async def get(self, timeout: float = 15.0) -> Optional[dict[str, Any]]:
        """Next event, or None on timeout (lets callers poll for disconnects)."""
        msg = await self._pubsub.get_message(ignore_subscribe_messages=True, timeout=timeout)
        if msg is None:
            return None
        return json.loads(msg["data"])

    async def __aenter__(self) -> "BrokerSubscription":
        return self

    async def __aexit__(self, *exc) -> None:
        await self.close()

    async def close(self) -> None:
        try:
            await self._pubsub.aclose()
        except Exception:
            pass

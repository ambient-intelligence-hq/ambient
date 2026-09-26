from fastapi import Header, Query

from ambient.config import settings
from ambient.server.errors import unauthorized


async def require_api_key(
    x_api_key: str | None = Header(default=None),
    authorization: str | None = Header(default=None),
    api_key: str | None = Query(default=None),
) -> None:
    """Auth for the Managed Agents protocol.

    The Anthropic SDK sends the key as `x-api-key`. We also accept
    `Authorization: Bearer <key>` so the same token works for curl/Postman, and
    an `?api_key=` query param so browser-native elements (Studio's `<video>` /
    thumbnail `<img>` on `/files/{id}/content`) — which can't set custom headers —
    can still authenticate.
    """
    token: str | None = None
    if x_api_key:
        token = x_api_key.strip()
    elif authorization and authorization.lower().startswith("bearer "):
        token = authorization.split(" ", 1)[1].strip()
    elif api_key:
        token = api_key.strip()
    if not token:
        raise unauthorized("missing x-api-key (or Authorization: Bearer) header")
    if token != settings.api_key:
        raise unauthorized("api key does not match settings.api_key")

from fastapi import Header

from ambient.config import settings
from ambient.server.errors import unauthorized


async def require_api_key(
    x_api_key: str | None = Header(default=None),
    authorization: str | None = Header(default=None),
) -> None:
    """Auth for the Managed Agents protocol.

    The Anthropic SDK sends the key as `x-api-key`. We also accept
    `Authorization: Bearer <key>` so the same token works for curl/Postman.
    """
    token: str | None = None
    if x_api_key:
        token = x_api_key.strip()
    elif authorization and authorization.lower().startswith("bearer "):
        token = authorization.split(" ", 1)[1].strip()
    if not token:
        raise unauthorized("missing x-api-key (or Authorization: Bearer) header")
    if token != settings.api_key:
        raise unauthorized("api key does not match settings.api_key")

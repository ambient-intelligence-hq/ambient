"""Ambient server package.

Intentionally does not import the FastAPI app at package import time. That keeps
`import ambient.server.runner` (which the CLI drives with in-memory adapters)
from pulling in FastAPI/uvicorn and the whole HTTP stack. The ASGI app lives at
`ambient.server.app:app` — the import string uvicorn and `python -m
ambient.server` use — so nothing here needs to eagerly bind it.
"""

import uvicorn

from ambient.config import settings


def main() -> None:
    # `server_workers > 1` runs multiple processes sharing Postgres + Redis; it
    # is mutually exclusive with reload (not used here).
    uvicorn.run(
        "ambient.server.app:app",
        host=settings.server_host,
        port=settings.server_port,
        log_level="info",
        workers=settings.server_workers,
    )


if __name__ == "__main__":
    main()

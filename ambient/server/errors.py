from typing import Any, Optional

from fastapi import HTTPException


class APIError(HTTPException):
    def __init__(
        self,
        status_code: int,
        code: str,
        message: str,
        details: Optional[dict[str, Any]] = None,
    ):
        super().__init__(
            status_code=status_code,
            detail={"error": {"code": code, "message": message, "details": details or {}}},
        )


def unauthorized(message: str = "missing or invalid bearer token") -> APIError:
    return APIError(401, "unauthorized", message)


def not_found(resource: str, id: str) -> APIError:
    return APIError(404, "not_found", f"{resource} {id!r} not found")


def conflict(message: str, details: Optional[dict[str, Any]] = None) -> APIError:
    return APIError(409, "conflict", message, details)


def bad_request(message: str, details: Optional[dict[str, Any]] = None) -> APIError:
    return APIError(400, "bad_request", message, details)

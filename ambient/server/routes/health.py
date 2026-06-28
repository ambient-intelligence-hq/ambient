from fastapi import APIRouter

router = APIRouter()


@router.get("/v1/healthz")
async def healthz() -> dict:
    return {"status": "ok"}

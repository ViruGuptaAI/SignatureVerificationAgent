from fastapi import APIRouter
from fastapi.responses import JSONResponse

from app.azure_client import get_client
from app.services.blob_storage import check_blob_health

router = APIRouter()


@router.get("/health")
async def health():
    """Liveness probe — confirms the server is running."""
    return {"status": "ok"}


@router.get("/health/ready")
async def readiness():
    """Readiness probe — confirms Azure OpenAI and Blob Storage are reachable."""
    checks = {}

    # 1. Azure OpenAI client
    try:
        client = get_client()
        await client.models.list()
        checks["azure_openai"] = "ok"
    except Exception as exc:
        checks["azure_openai"] = f"error: {exc}"

    # 2. Azure Blob Storage
    try:
        checks["blob_storage"] = await check_blob_health()
    except Exception as exc:
        checks["blob_storage"] = f"error: {exc}"

    all_ok = all(v == "ok" for v in checks.values())
    return JSONResponse(
        status_code=200 if all_ok else 503,
        content={"status": "ready" if all_ok else "degraded", "checks": checks},
    )

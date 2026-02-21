from fastapi import APIRouter
from fastapi.responses import JSONResponse

from app.azure_client import get_client
from app.config import LOGS_DIR

router = APIRouter()


@router.get("/health")
async def health():
    """Liveness probe — confirms the server is running."""
    return {"status": "ok"}


@router.get("/health/ready")
async def readiness():
    """Readiness probe — confirms Azure OpenAI is reachable and logs dir is writable."""
    checks = {}

    # 1. Azure OpenAI client
    try:
        client = get_client()
        await client.models.list()
        checks["azure_openai"] = "ok"
    except Exception as exc:
        checks["azure_openai"] = f"error: {exc}"

    # 2. Logs directory writable
    try:
        test_file = LOGS_DIR / ".health_check"
        test_file.write_text("ok", encoding="utf-8")
        test_file.unlink()
        checks["logs_dir"] = "ok"
    except Exception as exc:
        checks["logs_dir"] = f"error: {exc}"

    all_ok = all(v == "ok" for v in checks.values())
    return JSONResponse(
        status_code=200 if all_ok else 503,
        content={"status": "ready" if all_ok else "degraded", "checks": checks},
    )

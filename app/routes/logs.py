"""
GET /api/logs/{request_id}  — Retrieve a previously saved batch result by its request ID.
"""

import json
import re

from fastapi import APIRouter, HTTPException

from app.config import LOGS_DIR

router = APIRouter()

# Only allow UUID-shaped IDs to prevent path-traversal attacks
_UUID_RE = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$", re.IGNORECASE
)


@router.get("/api/logs/{request_id}")
async def get_log(request_id: str):
    """Return the JSON log for a given batch request ID."""

    if not _UUID_RE.match(request_id):
        raise HTTPException(status_code=400, detail="Invalid request ID format. Expected a UUID.")

    log_path = LOGS_DIR / f"{request_id}.json"

    if not log_path.exists():
        raise HTTPException(status_code=404, detail=f"No log found for request ID '{request_id}'.")

    try:
        return json.loads(log_path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Error reading log: {exc}")

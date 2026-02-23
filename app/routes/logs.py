"""
GET /api/logs/{request_id}  — Retrieve a previously saved result by its request ID from Azure Blob Storage.
"""

import json
import re

from fastapi import APIRouter, HTTPException

from app.services.blob_storage import download_log

router = APIRouter()

# Only allow UUID-shaped IDs to prevent injection attacks
_UUID_RE = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$", re.IGNORECASE
)


@router.get("/api/logs/{request_id}")
async def get_log(request_id: str):
    """Return the JSON log for a given request ID from blob storage."""

    if not _UUID_RE.match(request_id):
        raise HTTPException(status_code=400, detail="Invalid request ID format. Expected a UUID.")

    try:
        data = await download_log(request_id)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Error reading log from blob storage: {exc}")

    if data is None:
        raise HTTPException(status_code=404, detail=f"No log found for request ID '{request_id}'.")

    try:
        return json.loads(data)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Error parsing log JSON: {exc}")

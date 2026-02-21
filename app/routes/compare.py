from typing import Literal

from fastapi import APIRouter, File, HTTPException, Query, UploadFile

from app.config import ALLOWED_TYPES, ALLOWED_MODELS, MAX_IMAGE_SIZE
from app.models import CompareResponse
from app.services.comparison import compare_signatures

router = APIRouter()


@router.post("/api/VerifySignature", response_model=CompareResponse)
async def verify_signature(
    image1: UploadFile = File(..., description="First signature image"),
    image2: UploadFile = File(..., description="Second signature image"),
    preprocess: bool = True,
    model: ALLOWED_MODELS = Query("gpt-4.1", description="Model to use for comparison"),
    reasoning_effort: Literal["low", "medium", "high"] = Query(
        "medium", description="Reasoning effort for gpt-5 models (low, medium, high)"
    ),
):
    """Compare two uploaded signature images and return a structured similarity assessment."""

    # Validate content types
    for img, label in [(image1, "image1"), (image2, "image2")]:
        if img.content_type not in ALLOWED_TYPES:
            raise HTTPException(
                status_code=400,
                detail=f"{label} has unsupported content type '{img.content_type}'. "
                       f"Allowed: {', '.join(sorted(ALLOWED_TYPES))}",
            )

    image1_bytes = await image1.read()
    image2_bytes = await image2.read()

    if not image1_bytes or not image2_bytes:
        raise HTTPException(status_code=400, detail="Both images must be non-empty.")

    for data, label in [(image1_bytes, "image1"), (image2_bytes, "image2")]:
        if len(data) > MAX_IMAGE_SIZE:
            raise HTTPException(
                status_code=413,
                detail=f"{label} exceeds the 1 MB size limit ({len(data) / 1024 / 1024:.2f} MB).",
            )

    return await compare_signatures(
        image1_bytes, image1.filename or "image1.png",
        image2_bytes, image2.filename or "image2.png",
        preprocess=preprocess,
        model=model,
        reasoning_effort=reasoning_effort,
    )

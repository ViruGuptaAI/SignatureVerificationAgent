import base64
import json
import logging
import mimetypes
import os
import time
from contextlib import asynccontextmanager

from openai import AsyncAzureOpenAI
from azure.identity.aio import (
    AzureCliCredential,
    ManagedIdentityCredential,
    get_bearer_token_provider,
)
from dotenv import load_dotenv
from fastapi import FastAPI, Query, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Literal

from systemInstructions import signatureMatcher
from image_preprocessing import preprocess_signature_pair

load_dotenv(override=True)

logger = logging.getLogger("signature_agent")

# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------

class SignatureResult(BaseModel):
    """Structured response from the signature comparison agent."""
    signature_matched: bool
    confidence_score: float = Field(..., ge=0.0, le=1.0)
    reasoning: str


class TimingMetrics(BaseModel):
    """Latency metrics captured during the streaming API call."""
    stream_opened_ms: float = Field(..., description="Time to establish the stream connection")
    ttft_ms: float = Field(..., description="Time from stream open to first content token")
    ttfb_ms: float = Field(..., description="Time from request start to first content byte")
    ttlb_ms: float = Field(..., description="Time from request start to last byte received")


class CompareResponse(BaseModel):
    """Full API response including usage metadata."""
    result: SignatureResult
    usage: dict | None = None
    timing: TimingMetrics
    elapsed_ms: float


# ---------------------------------------------------------------------------
# Azure OpenAI helpers
# ---------------------------------------------------------------------------

_client: AsyncAzureOpenAI | None = None


def _get_credential():
    """Use Managed Identity in Azure App Service, fall back to CLI locally."""
    if os.getenv("WEBSITE_SITE_NAME"):
        return ManagedIdentityCredential()
    return AzureCliCredential()


def _build_client() -> AsyncAzureOpenAI:
    return AsyncAzureOpenAI(
        azure_endpoint=os.getenv("AZURE_ENDPOINT"),
        azure_deployment=os.getenv("AZURE_DEPLOYMENT", "gpt-4.1-mini"),
        azure_ad_token_provider=get_bearer_token_provider(
            _get_credential(), "https://cognitiveservices.azure.com/.default"
        ),
        api_version="2025-03-01-preview",
        max_retries=int(os.getenv("AZURE_MAX_RETRIES", 3))
    )


def _encode_bytes(data: bytes, filename: str) -> str:
    """Return a base64 data-URI from raw bytes."""
    mime, _ = mimetypes.guess_type(filename)
    mime = mime or "image/png"
    b64 = base64.b64encode(data).decode()
    return f"data:{mime};base64,{b64}"


# ---------------------------------------------------------------------------
# Lifespan – create / tear down the shared OpenAI client
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    global _client
    _client = _build_client()
    yield
    await _client.close()


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Signature Matching Agent API",
    description="Upload two signature images to compare their visual similarity.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Core comparison logic
# ---------------------------------------------------------------------------

ALLOWED_TYPES = {"image/png", "image/jpeg", "image/jpg", "image/webp", "image/gif"}


async def _compare_signatures(image1_bytes: bytes, image1_name: str,
                               image2_bytes: bytes, image2_name: str,
                               preprocess: bool = True,
                               reasoning_effort: Literal["low", "medium", "high"] = "medium") -> CompareResponse:
    """Send both images to the model and return the structured result."""
    model = os.getenv("AZURE_DEPLOYMENT", "gpt-4.1-mini")

    # Optional image preprocessing (grayscale + denoise + autocrop)
    if preprocess:
        image1_bytes, image2_bytes = preprocess_signature_pair(image1_bytes, image2_bytes)
        # After preprocessing, images are PNGs regardless of original format
        image1_name = "image1.png"
        image2_name = "image2.png"

    data_uri_1 = _encode_bytes(image1_bytes, image1_name)
    data_uri_2 = _encode_bytes(image2_bytes, image2_name)

    content = [
        {
            "type": "input_text",
            "text": (
                "Analyze both the signatures in the images and determine if they match."
            ),
        },
        {"type": "input_image", "image_url": data_uri_1, "detail": "high"},
        {"type": "input_image", "image_url": data_uri_2, "detail": "high"},
    ]

    # Build the JSON-schema definition for structured output
    json_schema = {
        "type": "object",
        "properties": {
            "signature_matched": {"type": "boolean"},
            "confidence_score": {"type": "number"},
            "reasoning": {"type": "string"},
        },
        "required": ["signature_matched", "confidence_score", "reasoning"],
        "additionalProperties": False,
    }

    common_kwargs: dict = dict(
        model=model,
        instructions=signatureMatcher,
        input=[{"role": "user", "content": content}],
        text={"format": {"type": "json_schema", "name": "signature_result", "strict": True, "schema": json_schema}},
    )

    if "gpt-5" in model.lower():
        common_kwargs["reasoning"] = {"effort": reasoning_effort}
    else:
        common_kwargs["temperature"] = 0.2

    # --- Streaming call with timing instrumentation ---
    t0 = time.perf_counter()

    stream = await _client.responses.create(**common_kwargs, stream=True)
    async with stream as stream:
        stream_opened_ms = (time.perf_counter() - t0) * 1000

        ttft_ms: float | None = None
        final_response = None
        collected_text = ""

        async for event in stream:
            if event.type == "response.output_text.delta":
                if ttft_ms is None:
                    ttft_ms = (time.perf_counter() - t0) * 1000 - stream_opened_ms
                collected_text += event.delta
            elif event.type == "response.completed":
                final_response = event.response

    ttlb_ms = (time.perf_counter() - t0) * 1000
    ttft_ms = ttft_ms or 0.0
    ttfb_ms = stream_opened_ms + ttft_ms

    timing = TimingMetrics(
        stream_opened_ms=round(stream_opened_ms, 1),
        ttft_ms=round(ttft_ms, 1),
        ttfb_ms=round(ttfb_ms, 1),
        ttlb_ms=round(ttlb_ms, 1),
    )

    # Extract usage from the completed response
    usage = None
    if final_response and final_response.usage:
        u = final_response.usage
        reasoning_tokens = getattr(
            getattr(u, "output_tokens_details", None), "reasoning_tokens", 0
        )
        usage = {
            "input_tokens": u.input_tokens,
            "output_tokens": u.output_tokens,
            "reasoning_tokens": reasoning_tokens,
            "total_tokens": u.total_tokens,
        }

    # Structured output guarantees valid JSON — parse directly
    raw_text = collected_text.strip()
    try:
        parsed = json.loads(raw_text)
        result = SignatureResult(**parsed)
    except (json.JSONDecodeError, Exception) as exc:
        raise HTTPException(
            status_code=502,
            detail=f"Model returned invalid JSON: {exc}\nRaw output: {raw_text}",
        )

    # Log metrics summary
    token_info = ""
    if usage:
        token_info = (
            f"tokens: {usage['total_tokens']} "
            f"(in:{usage['input_tokens']} out:{usage['output_tokens']} "
            f"reasoning:{usage['reasoning_tokens']})"
        )
    logger.info(
        "--- Stream opened: %.0f ms | TTFT: %.0f ms | TTFB: %.0f ms | TTLB: %.0f ms | %s ---",
        timing.stream_opened_ms, timing.ttft_ms, timing.ttfb_ms, timing.ttlb_ms,
        token_info,
    )

    return CompareResponse(
        result=result, usage=usage, timing=timing, elapsed_ms=round(ttlb_ms, 1)
    )


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.post("/compare", response_model=CompareResponse)
async def compare_signatures(
    image1: UploadFile = File(..., description="First signature image"),
    image2: UploadFile = File(..., description="Second signature image"),
    preprocess: bool = True,
    reasoning_effort: Literal["low", "medium", "high"] = Query("medium", description="Reasoning effort for o-series/gpt-5 models (low, medium, high)"),
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

    MAX_IMAGE_SIZE = 1 * 1024 * 1024  # 1 MB
    for data, label in [(image1_bytes, "image1"), (image2_bytes, "image2")]:
        if len(data) > MAX_IMAGE_SIZE:
            raise HTTPException(
                status_code=413,
                detail=f"{label} exceeds the 1 MB size limit ({len(data) / 1024 / 1024:.2f} MB).",
            )

    return await _compare_signatures(
        image1_bytes, image1.filename or "image1.png",
        image2_bytes, image2.filename or "image2.png",
        preprocess=preprocess,
        reasoning_effort=reasoning_effort,
    )


@app.get("/health")
async def health():
    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("backend:app", host="127.0.0.1", port=8000, log_level="info", workers=4, reload=True)
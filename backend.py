import base64
import json
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
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from systemInstructions import signatureMatcher
from image_preprocessing import preprocess_signature_pair

load_dotenv(override=True)

# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------

class SignatureResult(BaseModel):
    """Structured response from the signature comparison agent."""
    signature_matched: bool
    confidence_score: float = Field(..., ge=0.0, le=1.0)
    reasoning: str


class CompareResponse(BaseModel):
    """Full API response including usage metadata."""
    result: SignatureResult
    usage: dict | None = None
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
                               preprocess: bool = True) -> CompareResponse:
    """Send both images to the model and return the structured result."""
    model = os.getenv("AZURE_DEPLOYMENT", "gpt-4.1-mini")
    start = time.perf_counter()

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
                "Analyze both the signatures in the images and determine if they match. "
                "Return ONLY the JSON object as specified in the instructions."
            ),
        },
        {"type": "input_image", "image_url": data_uri_1, "detail": "high"},
        {"type": "input_image", "image_url": data_uri_2, "detail": "high"},
    ]

    common_kwargs: dict = dict(
        model=model,
        instructions=signatureMatcher,
        input=[{"role": "user", "content": content}],
    )

    if "gpt-5" in model.lower():
        common_kwargs["reasoning"] = {"effort": "medium"}
    else:
        common_kwargs["temperature"] = 0.2  # low temp for deterministic structured output

    response = await _client.responses.create(**common_kwargs)

    elapsed_ms = (time.perf_counter() - start) * 1000

    # Extract usage
    u = response.usage
    reasoning_tokens = getattr(
        getattr(u, "output_tokens_details", None), "reasoning_tokens", 0
    )
    usage = {
        "input_tokens": u.input_tokens,
        "output_tokens": u.output_tokens,
        "reasoning_tokens": reasoning_tokens,
        "total_tokens": u.total_tokens,
    }

    # Parse the model's JSON reply
    raw_text = response.output_text.strip()
    # Strip markdown code fences if present
    if raw_text.startswith("```"):
        raw_text = raw_text.split("\n", 1)[1] if "\n" in raw_text else raw_text[3:]
        if raw_text.endswith("```"):
            raw_text = raw_text[:-3].strip()

    try:
        parsed = json.loads(raw_text)
        result = SignatureResult(**parsed)
    except (json.JSONDecodeError, Exception) as exc:
        raise HTTPException(
            status_code=502,
            detail=f"Model returned invalid JSON: {exc}\nRaw output: {raw_text}",
        )

    return CompareResponse(result=result, usage=usage, elapsed_ms=round(elapsed_ms, 1))


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.post("/compare", response_model=CompareResponse)
async def compare_signatures(
    image1: UploadFile = File(..., description="First signature image"),
    image2: UploadFile = File(..., description="Second signature image"),
    preprocess: bool = True,
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

    return await _compare_signatures(
        image1_bytes, image1.filename or "image1.png",
        image2_bytes, image2.filename or "image2.png",
        preprocess=preprocess,
    )


@app.get("/health")
async def health():
    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("backend:app", host="127.0.0.1", port=8000, reload=True, log_level="info")
import asyncio
import base64
import hashlib
import json
import logging
import mimetypes
import os
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

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

from systemInstructions import signatureMatcher, batchSummaryPrompt
from image_preprocessing import preprocess_signature_pair

load_dotenv(override=True)

logger = logging.getLogger("signature_agent")
logger.setLevel(logging.DEBUG)
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(asctime)s [%(name)s] %(message)s"))
    logger.addHandler(handler)

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
    image1: str = Field(..., description="Filename of the first image sent to the model")
    image2: str = Field(..., description="Filename of the second image sent to the model")
    result: SignatureResult
    usage: dict | None = None
    timing: TimingMetrics
    elapsed_ms: float


class IndividualResult(BaseModel):
    """Result of a single reference-vs-test comparison."""
    reference_filename: str
    test_filename: str
    signature_matched: bool
    confidence_score: float = Field(..., ge=0.0, le=1.0)
    reasoning: str
    usage: dict | None = None
    elapsed_ms: float
    error: str | None = None


class BatchVerdict(BaseModel):
    """Aggregated verdict from multiple reference comparisons."""
    signature_matched: bool
    avg_confidence: float = Field(..., ge=0.0, le=1.0)
    match_ratio: str = Field(..., description="e.g. '7/10'")
    decision_method: str = "majority_vote"
    reasoning: str = Field(..., description="LLM-generated summary of all individual reasonings")
    inconclusive: bool = False


class BatchCompareResponse(BaseModel):
    """Full batch API response."""
    request_id: str = Field(..., description="Unique UUID for this invocation")
    verdict: BatchVerdict
    individual_results: list[IndividualResult]
    total_usage: dict | None = None
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
# Logs directory
# ---------------------------------------------------------------------------

LOGS_DIR = Path(__file__).parent / "logs"
LOGS_DIR.mkdir(exist_ok=True)


# ---------------------------------------------------------------------------
# Core comparison logic
# ---------------------------------------------------------------------------

ALLOWED_TYPES = {"image/png", "image/jpeg", "image/jpg", "image/webp", "image/gif"}
ALLOWED_MODELS = Literal["gpt-4.1", "gpt-5.2", "gpt-5-mini"]


async def _compare_signatures(image1_bytes: bytes, image1_name: str,
                               image2_bytes: bytes, image2_name: str,
                               preprocess: bool = True,
                               model: str = "gpt-4.1",
                               reasoning_effort: Literal["low", "medium", "high"] = "medium") -> CompareResponse:
    """Send both images to the model and return the structured result."""

    img1_hash_raw = hashlib.md5(image1_bytes).hexdigest()[:8]
    img2_hash_raw = hashlib.md5(image2_bytes).hexdigest()[:8]
    logger.info(
        "--- _compare_signatures: image1 = %s [%d bytes, md5=%s] | image2 = %s [%d bytes, md5=%s] | model = %s ---",
        image1_name, len(image1_bytes), img1_hash_raw,
        image2_name, len(image2_bytes), img2_hash_raw, model,
    )

    # Keep original filenames for the response; preprocessing changes format internally
    original_image1_name = image1_name
    original_image2_name = image2_name

    # Optional image preprocessing (grayscale + denoise + autocrop)
    # Runs in a thread pool so the CPU-bound Pillow work doesn't block the event loop
    if preprocess:
        loop = asyncio.get_running_loop()
        image1_bytes, image2_bytes = await loop.run_in_executor(
            None, preprocess_signature_pair, image1_bytes, image2_bytes
        )
        # After preprocessing, images are PNGs regardless of original format
        image1_name = "image1.png"
        image2_name = "image2.png"

    img1_hash_final = hashlib.md5(image1_bytes).hexdigest()[:8]
    img2_hash_final = hashlib.md5(image2_bytes).hexdigest()[:8]
    logger.info(
        "--- Sending to model: image1 md5=%s | image2 md5=%s (same=%s) ---",
        img1_hash_final, img2_hash_final, img1_hash_final == img2_hash_final,
    )

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
        store=False,
    )

    if "gpt-5" in model.lower():
        common_kwargs["reasoning"] = {"effort": reasoning_effort}
    else:
        common_kwargs["temperature"] = 0

    # --- Streaming call with timing instrumentation ---
    STREAM_TIMEOUT_SECONDS = int(os.getenv("STREAM_TIMEOUT_SECONDS", 60))
    t0 = time.perf_counter()

    try:
        stream = await asyncio.wait_for(
            _client.responses.create(**common_kwargs, stream=True),
            timeout=STREAM_TIMEOUT_SECONDS,
        )
        async with stream as stream:
            stream_opened_ms = (time.perf_counter() - t0) * 1000

            ttft_ms: float | None = None
            final_response = None
            collected_text = ""

            async for event in stream:
                if (time.perf_counter() - t0) > STREAM_TIMEOUT_SECONDS:
                    raise asyncio.TimeoutError()
                if event.type == "response.output_text.delta":
                    if ttft_ms is None:
                        ttft_ms = (time.perf_counter() - t0) * 1000 - stream_opened_ms
                    collected_text += event.delta
                elif event.type == "response.completed":
                    final_response = event.response
    except asyncio.TimeoutError:
        raise HTTPException(
            status_code=504,
            detail=f"OpenAI request timed out after {STREAM_TIMEOUT_SECONDS} seconds.",
        )

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
        image1=original_image1_name, image2=original_image2_name,
        result=result, usage=usage, timing=timing, elapsed_ms=round(ttlb_ms, 1)
    )


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.post("/api/VerifySignature", response_model=CompareResponse)
async def compare_signatures(
    image1: UploadFile = File(..., description="First signature image"),
    image2: UploadFile = File(..., description="Second signature image"),
    preprocess: bool = True,
    model: ALLOWED_MODELS = Query("gpt-4.1", description="Model to use for comparison"),
    reasoning_effort: Literal["low", "medium", "high"] = Query("medium", description="Reasoning effort for gpt-5 models (low, medium, high)"),
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
        model=model,
        reasoning_effort=reasoning_effort,
    )


@app.post("/api/VerifySignatureBatch", response_model=BatchCompareResponse)
async def compare_signatures_batch(
    test_image: UploadFile = File(..., description="Single test signature to verify"),
    ref_1: UploadFile = File(..., description="Reference signature 1 (required)"),
    ref_2: UploadFile = File(..., description="Reference signature 2 (required)"),
    ref_3: UploadFile | None = File(None, description="Reference signature 3 (optional)"),
    ref_4: UploadFile | None = File(None, description="Reference signature 4 (optional)"),
    ref_5: UploadFile | None = File(None, description="Reference signature 5 (optional)"),
    ref_6: UploadFile | None = File(None, description="Reference signature 6 (optional)"),
    ref_7: UploadFile | None = File(None, description="Reference signature 7 (optional)"),
    ref_8: UploadFile | None = File(None, description="Reference signature 8 (optional)"),
    ref_9: UploadFile | None = File(None, description="Reference signature 9 (optional)"),
    ref_10: UploadFile | None = File(None, description="Reference signature 10 (optional)"),
    preprocess: bool = True,
    model: ALLOWED_MODELS = Query("gpt-4.1", description="Model to use for comparison"),
    reasoning_effort: Literal["low", "medium", "high"] = Query("medium", description="Reasoning effort for gpt-5 models"),
):
    """Compare a test signature against 2–10 reference signatures and return an aggregated verdict."""

    # --- Collect provided reference images ---
    reference_images: list[UploadFile] = [
        img for img in [ref_1, ref_2, ref_3, ref_4, ref_5, ref_6, ref_7, ref_8, ref_9, ref_10]
        if img is not None
    ]

    if len(reference_images) < 2:
        raise HTTPException(status_code=400, detail="At least 2 reference images are required.")

    # --- Validate content types ---
    all_images = [(test_image, "test_image")] + [
        (img, f"reference_images[{i}]") for i, img in enumerate(reference_images)
    ]
    for img, label in all_images:
        if img.content_type not in ALLOWED_TYPES:
            raise HTTPException(
                status_code=400,
                detail=f"{label} has unsupported content type '{img.content_type}'. "
                       f"Allowed: {', '.join(sorted(ALLOWED_TYPES))}",
            )

    # --- Read and validate sizes ---
    MAX_IMAGE_SIZE = 1 * 1024 * 1024  # 1 MB

    test_bytes = await test_image.read()
    if not test_bytes:
        raise HTTPException(status_code=400, detail="test_image must be non-empty.")
    if len(test_bytes) > MAX_IMAGE_SIZE:
        raise HTTPException(
            status_code=413,
            detail=f"test_image exceeds the 1 MB size limit ({len(test_bytes) / 1024 / 1024:.2f} MB).",
        )

    ref_data: list[tuple[bytes, str]] = []
    for i, ref_img in enumerate(reference_images):
        data = await ref_img.read()
        label = f"reference_images[{i}]"
        if not data:
            raise HTTPException(status_code=400, detail=f"{label} must be non-empty.")
        if len(data) > MAX_IMAGE_SIZE:
            raise HTTPException(
                status_code=413,
                detail=f"{label} exceeds the 1 MB size limit ({len(data) / 1024 / 1024:.2f} MB).",
            )
        ref_data.append((data, ref_img.filename or f"reference_{i}.png"))

    # --- Run all comparisons concurrently ---
    t0 = time.perf_counter()

    async def _single_comparison(ref_bytes: bytes, ref_name: str, index: int) -> IndividualResult:
        """Run one test-vs-reference comparison, catching errors gracefully."""
        test_name = test_image.filename or "test.png"
        ref_hash = hashlib.md5(ref_bytes).hexdigest()[:8]
        test_hash = hashlib.md5(test_bytes).hexdigest()[:8]
        logger.info(
            "--- Batch call %d: image1 (reference) = %s [%d bytes, md5=%s] | image2 (test) = %s [%d bytes, md5=%s] ---",
            index, ref_name, len(ref_bytes), ref_hash, test_name, len(test_bytes), test_hash,
        )
        try:
            resp = await _compare_signatures(
                image1_bytes=ref_bytes, image1_name=ref_name,
                image2_bytes=test_bytes, image2_name=test_image.filename or "test.png",
                preprocess=preprocess,
                model=model,
                reasoning_effort=reasoning_effort,
            )
            return IndividualResult(
                reference_filename=ref_name,
                test_filename=test_name,
                signature_matched=resp.result.signature_matched,
                confidence_score=resp.result.confidence_score,
                reasoning=resp.result.reasoning,
                usage=resp.usage,
                elapsed_ms=resp.elapsed_ms,
            )
        except Exception as exc:
            return IndividualResult(
                reference_filename=ref_name,
                test_filename=test_name,
                signature_matched=False,
                confidence_score=0.0,
                reasoning="",
                usage=None,
                elapsed_ms=0.0,
                error=str(exc),
            )

    tasks = [
        _single_comparison(ref_bytes, ref_name, i)
        for i, (ref_bytes, ref_name) in enumerate(ref_data)
    ]
    individual_results: list[IndividualResult] = await asyncio.gather(*tasks)

    # --- Aggregate results (majority vote) ---
    successful = [r for r in individual_results if r.error is None]
    if not successful:
        raise HTTPException(status_code=502, detail="All comparisons failed.")

    match_count = sum(1 for r in successful if r.signature_matched)
    total_count = len(successful)
    majority_matched = match_count > total_count / 2

    # Average confidence across ALL comparisons (not just the winning side)
    avg_confidence = sum(r.confidence_score for r in successful) / total_count

    # Flag as inconclusive if the split is nearly even (within 1 vote of 50/50)
    inconclusive = abs(match_count - (total_count - match_count)) <= 1 and total_count >= 2

    # --- Summarize all individual reasonings via LLM ---
    reasoning_texts = []
    for i, r in enumerate(successful):
        label = "MATCHED" if r.signature_matched else "NOT MATCHED"
        reasoning_texts.append(
            f"Comparison {i+1} ({r.reference_filename} vs {r.test_filename}) — {label} ({r.confidence_score}): {r.reasoning}"
        )
    all_reasonings = "\n\n".join(reasoning_texts)

    summary_prompt = batchSummaryPrompt(
        majority_matched=majority_matched,
        match_count=match_count,
        total_count=total_count,
        avg_confidence=avg_confidence,
        all_reasonings=all_reasonings,
    )

    try:
        summary_resp = await _client.responses.create(
            model=model,
            input=[{"role": "user", "content": summary_prompt}],
            temperature=0,
            store=False,
        )
        summary_reasoning = summary_resp.output_text.strip()
        # Track summary usage
        summary_usage = None
        if summary_resp.usage:
            su = summary_resp.usage
            reasoning_tokens = getattr(
                getattr(su, "output_tokens_details", None), "reasoning_tokens", 0
            )
            summary_usage = {
                "input_tokens": su.input_tokens,
                "output_tokens": su.output_tokens,
                "reasoning_tokens": reasoning_tokens,
                "total_tokens": su.total_tokens,
            }
    except Exception as exc:
        logger.warning("Summary LLM call failed: %s — falling back to concatenation", exc)
        summary_reasoning = all_reasonings
        summary_usage = None

    verdict = BatchVerdict(
        signature_matched=majority_matched,
        avg_confidence=round(avg_confidence, 4),
        match_ratio=f"{match_count}/{total_count}",
        decision_method="majority_vote",
        reasoning=summary_reasoning,
        inconclusive=inconclusive,
    )

    # --- Aggregate usage (include summary call) ---
    total_usage = None
    usages = [r.usage for r in successful if r.usage]
    if summary_usage:
        usages.append(summary_usage)
    if usages:
        total_usage = {
            "input_tokens": sum(u["input_tokens"] for u in usages),
            "output_tokens": sum(u["output_tokens"] for u in usages),
            "reasoning_tokens": sum(u["reasoning_tokens"] for u in usages),
            "total_tokens": sum(u["total_tokens"] for u in usages),
        }

    total_elapsed = (time.perf_counter() - t0) * 1000

    request_id = str(uuid.uuid4())

    logger.info(
        "--- Batch [%s]: %d references | verdict=%s | confidence=%.2f | match_ratio=%s | inconclusive=%s | %.0f ms ---",
        request_id, total_count, majority_matched, avg_confidence, verdict.match_ratio, inconclusive, total_elapsed,
    )

    response = BatchCompareResponse(
        request_id=request_id,
        verdict=verdict,
        individual_results=individual_results,
        total_usage=total_usage,
        elapsed_ms=round(total_elapsed, 1),
    )

    # --- Persist response log ---
    try:
        log_path = LOGS_DIR / f"{request_id}.json"
        log_path.write_text(response.model_dump_json(indent=2), encoding="utf-8")
        logger.info("--- Batch log saved: %s ---", log_path)
    except Exception as exc:
        logger.warning("Failed to write batch log: %s", exc)

    return response


@app.get("/health")
async def health():
    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("backend:app", host="0.0.0.0", port=8000, log_level="info", workers=4, reload=True)
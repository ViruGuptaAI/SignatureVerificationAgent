import asyncio
import hashlib
import json
import os
import time
from typing import Literal

from fastapi import HTTPException

from app.azure_client import get_client, encode_bytes
from app.config import logger
from app.models import SignatureResult, TimingMetrics, CompareResponse
from app.services.preprocessing import preprocess_signature_pair
from app.prompts import signatureMatcher

# ---------------------------------------------------------------------------
# Global semaphore — caps concurrent LLM calls across all requests to stay
# within the TPM budget.  30 users × 10 refs = 300 parallel calls without
# this gate; the semaphore serialises overflow so retries stay manageable.
# ---------------------------------------------------------------------------

_llm_semaphore = asyncio.Semaphore(int(os.getenv("MAX_CONCURRENT_LLM_CALLS", "20")))


def get_llm_semaphore() -> asyncio.Semaphore:
    """Expose the semaphore so routes (e.g. batch summary) can reuse it."""
    return _llm_semaphore


async def compare_signatures(
    image1_bytes: bytes,
    image1_name: str,
    image2_bytes: bytes,
    image2_name: str,
    preprocess: bool = True,
    model: str = "gpt-4.1",
    reasoning_effort: Literal["low", "medium", "high"] = "medium",
) -> CompareResponse:
    """Send both images to the model and return the structured result."""

    client = get_client()

    img1_hash_raw = hashlib.md5(image1_bytes).hexdigest()[:8]
    img2_hash_raw = hashlib.md5(image2_bytes).hexdigest()[:8]
    logger.info(
        "--- compare_signatures: image1 = %s [%d bytes, md5=%s] | image2 = %s [%d bytes, md5=%s] | model = %s ---",
        image1_name, len(image1_bytes), img1_hash_raw,
        image2_name, len(image2_bytes), img2_hash_raw, model,
    )

    # Keep original filenames for the response
    original_image1_name = image1_name
    original_image2_name = image2_name

    # Optional image preprocessing (grayscale + denoise + autocrop)
    # Runs in a thread pool so the CPU-bound Pillow work doesn't block the event loop
    if preprocess:
        loop = asyncio.get_running_loop()
        image1_bytes, image2_bytes = await loop.run_in_executor(
            None, preprocess_signature_pair, image1_bytes, image2_bytes
        )
        image1_name = "image1.png"
        image2_name = "image2.png"

    img1_hash_final = hashlib.md5(image1_bytes).hexdigest()[:8]
    img2_hash_final = hashlib.md5(image2_bytes).hexdigest()[:8]
    logger.info(
        "--- Sending to model: image1 md5=%s | image2 md5=%s (same=%s) ---",
        img1_hash_final, img2_hash_final, img1_hash_final == img2_hash_final,
    )

    data_uri_1 = encode_bytes(image1_bytes, image1_name)
    data_uri_2 = encode_bytes(image2_bytes, image2_name)

    content = [
        {
            "type": "input_text",
            "text": "Analyze both the signatures in the images and determine if they match.",
        },
        {"type": "input_image", "image_url": data_uri_1, "detail": "high"},
        {"type": "input_image", "image_url": data_uri_2, "detail": "high"},
    ]

    # JSON-schema for structured output
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
        text={
            "format": {
                "type": "json_schema",
                "name": "signature_result",
                "strict": True,
                "schema": json_schema,
            }
        },
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
        async with _llm_semaphore:
            stream = await asyncio.wait_for(
                client.responses.create(**common_kwargs, stream=True),
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

    # Extract usage
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

    # Parse structured JSON output
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
        image1=original_image1_name,
        image2=original_image2_name,
        result=result,
        usage=usage,
        timing=timing,
        elapsed_ms=round(ttlb_ms, 1),
    )

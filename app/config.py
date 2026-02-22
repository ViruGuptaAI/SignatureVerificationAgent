import logging
import os
from pathlib import Path
from typing import Literal

# ---------------------------------------------------------------------------
# Logger
# ---------------------------------------------------------------------------

logger = logging.getLogger("signature_agent")
logger.setLevel(logging.DEBUG)
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(asctime)s [%(name)s] %(message)s"))
    logger.addHandler(handler)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ALLOWED_TYPES: set[str] = {"image/png", "image/jpeg", "image/jpg", "image/webp", "image/gif"}
ALLOWED_MODELS = Literal["gpt-4.1", "gpt-5.2", "gpt-5-mini"]
MAX_IMAGE_SIZE: int = 1 * 1024 * 1024  # 1 MB

# ---------------------------------------------------------------------------
# Logs directory (project-root/logs)
# ---------------------------------------------------------------------------

LOGS_DIR = Path(__file__).resolve().parent.parent / "logs"
LOGS_DIR.mkdir(exist_ok=True)

# ---------------------------------------------------------------------------
# Model pricing (INR per 1M tokens) — loaded from environment
# ---------------------------------------------------------------------------

def _cost(var: str) -> float:
    return float(os.getenv(var, "0"))

MODEL_PRICING: dict[str, dict[str, float]] = {
    "gpt-4.1": {
        "input":  _cost("MODEL_GPT41_INPUT"),
        "cached": _cost("MODEL_GPT41_CACHED"),
        "output": _cost("MODEL_GPT41_OUTPUT"),
    },
    "gpt-5.2": {
        "input":  _cost("MODEL_GPT52_INPUT"),
        "cached": _cost("MODEL_GPT52_CACHED"),
        "output": _cost("MODEL_GPT52_OUTPUT"),
    },
    "gpt-5-mini": {
        "input":  _cost("MODEL_GPT5MINI_INPUT"),
        "cached": _cost("MODEL_GPT5MINI_CACHED"),
        "output": _cost("MODEL_GPT5MINI_OUTPUT"),
    },
}


def calculate_cost_inr(usage: dict | None, model: str) -> float | None:
    """
    Calculate the cost of an API call in INR given a usage dict and model name.

    Usage dict must have: input_tokens, output_tokens, and optionally
    input_tokens_details.cached_tokens (or cached_tokens at top level).

    Returns None if usage is missing or pricing is unavailable.
    """
    if not usage:
        return None

    pricing = MODEL_PRICING.get(model)
    if not pricing:
        return None

    input_tokens = usage.get("input_tokens", 0)
    output_tokens = usage.get("output_tokens", 0)

    # Cached tokens may come from different locations depending on the SDK response
    cached_tokens = usage.get("cached_tokens", 0)
    if not cached_tokens:
        details = usage.get("input_tokens_details") or {}
        cached_tokens = details.get("cached_tokens", 0)

    # Non-cached input tokens = total input - cached
    regular_input = max(0, input_tokens - cached_tokens)

    cost = (
        (regular_input / 1_000_000) * pricing["input"]
        + (cached_tokens / 1_000_000) * pricing["cached"]
        + (output_tokens / 1_000_000) * pricing["output"]
    )
    return round(cost, 6)

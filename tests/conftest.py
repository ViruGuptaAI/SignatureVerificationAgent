"""
Shared fixtures for the Signature Matching Agent test suite.

Run all tests:   pytest tests/ -v
Run one file :   pytest tests/test_models.py -v
"""

from __future__ import annotations

import asyncio
import io
import json
import os
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient
from PIL import Image

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "Data"


# ---------------------------------------------------------------------------
# Async event loop (one per session — faster)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def event_loop():
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


# ---------------------------------------------------------------------------
# Sample image bytes (tiny 50x20 white image with a black stroke)
# ---------------------------------------------------------------------------

def _make_test_image(width: int = 100, height: int = 50, color: str = "white") -> bytes:
    """Create a minimal PNG image in memory."""
    img = Image.new("RGB", (width, height), color)
    # Draw a simple diagonal line to simulate a signature stroke
    for i in range(min(width, height)):
        img.putpixel((i, i), (0, 0, 0))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


@pytest.fixture
def sample_image_bytes() -> bytes:
    """A small synthetic signature image (PNG bytes)."""
    return _make_test_image()


@pytest.fixture
def sample_image_pair() -> tuple[bytes, bytes]:
    """Two distinct synthetic images for pair-based tests."""
    return _make_test_image(100, 50), _make_test_image(120, 60, "lightgray")


@pytest.fixture
def real_image_bytes() -> bytes | None:
    """Load VR1.jpg from Data/ if it exists, otherwise skip."""
    path = DATA_DIR / "VR1.jpg"
    if not path.exists():
        pytest.skip("Data/VR1.jpg not found — skipping real-image test")
    return path.read_bytes()


# ---------------------------------------------------------------------------
# Mock Azure OpenAI response (avoids real API calls)
# ---------------------------------------------------------------------------

MOCK_SIGNATURE_RESULT = {
    "signature_matched": True,
    "confidence_score": 0.85,
    "reasoning": "Mock reasoning: strokes are similar in flow and pressure.",
}


def _build_mock_stream():
    """Build an async context-manager mock that yields streaming events."""

    class FakeEvent:
        def __init__(self, etype, delta=None, response=None):
            self.type = etype
            self.delta = delta
            self.response = response

    class FakeUsage:
        input_tokens = 500
        output_tokens = 100
        total_tokens = 600
        output_tokens_details = None

    class FakeResponse:
        usage = FakeUsage()

    events = [
        FakeEvent("response.output_text.delta", delta=json.dumps(MOCK_SIGNATURE_RESULT)),
        FakeEvent("response.completed", response=FakeResponse()),
    ]

    class MockStream:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            pass

        def __aiter__(self):
            return self

        async def __anext__(self):
            if events:
                return events.pop(0)
            raise StopAsyncIteration

    return MockStream()


@pytest.fixture
def mock_openai_client():
    """Return a mocked AsyncAzureOpenAI client whose responses.create streams fake data."""
    client = AsyncMock()
    client.responses.create = AsyncMock(side_effect=lambda **kw: _build_mock_stream() if kw.get("stream") else _build_mock_non_stream())
    client.models.list = AsyncMock(return_value=[])
    client.close = AsyncMock()
    return client


def _build_mock_non_stream():
    """For non-streaming calls like the batch summary."""
    class FakeUsage:
        input_tokens = 200
        output_tokens = 80
        total_tokens = 280
        output_tokens_details = None

    resp = MagicMock()
    resp.output_text = "Mock summary reasoning."
    resp.usage = FakeUsage()
    # Make it awaitable
    f = asyncio.Future()
    f.set_result(resp)
    return f


# ---------------------------------------------------------------------------
# FastAPI TestClient with mocked OpenAI
# ---------------------------------------------------------------------------

@pytest.fixture
def test_client(mock_openai_client):
    """
    A synchronous TestClient for the FastAPI app with Azure OpenAI mocked out.
    No real credentials or network calls needed.
    """
    # Patch the client before importing the app (lifespan would call build_client)
    with (
        patch("app.azure_client.build_client", return_value=mock_openai_client),
        # Mock blob storage so tests never hit real Azure Blob Storage
        patch("app.services.comparison.upload_log", new_callable=AsyncMock),
        patch("app.routes.batch.upload_log", new_callable=AsyncMock),
        patch("app.routes.logs.download_log", new_callable=AsyncMock, return_value=None),
        patch("app.routes.health.check_blob_health", new_callable=AsyncMock, return_value="ok"),
        patch("app.main.close_blob_client", new_callable=AsyncMock),
    ):
        from app.main import app
        # Manually set the client so routes can use get_client()
        from app.azure_client import set_client
        set_client(mock_openai_client)

        with TestClient(app) as tc:
            yield tc

        set_client(None)

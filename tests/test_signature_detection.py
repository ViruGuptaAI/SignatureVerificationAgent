"""
Tests for app/services/signature_detection.py.

Covers:
- SignatureDetectionResult dataclass
- _ensure_supported_format (WEBP→JPEG conversion, passthrough for PNG/JPEG)
- _detect_ink_region (Strategy C fallback)
- detect_and_crop_signature with mocked Document Intelligence client
"""

from __future__ import annotations

import io
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from PIL import Image

from app.services.signature_detection import (
    SignatureDetectionResult,
    _ensure_supported_format,
    _detect_ink_region,
    detect_and_crop_signature,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_image(width: int = 100, height: int = 50, fmt: str = "PNG", color: str = "white") -> bytes:
    """Create a simple image in the given format."""
    img = Image.new("RGB", (width, height), color)
    # Draw a thick dark stroke across the middle (>10px tall to pass bbox guard)
    for x in range(10, width - 10):
        for y in range(height // 2 - 8, height // 2 + 8):
            img.putpixel((x, y), (0, 0, 0))
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    return buf.getvalue()


def _make_blank_image(width: int = 100, height: int = 50, color: str = "white") -> bytes:
    """Pure solid-color image — no ink at all."""
    img = Image.new("RGB", (width, height), color)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


# ---------------------------------------------------------------------------
# SignatureDetectionResult dataclass
# ---------------------------------------------------------------------------

class TestSignatureDetectionResult:
    def test_defaults(self):
        r = SignatureDetectionResult(signature_found=False)
        assert r.confidence == 0.0
        assert r.cropped_bytes is None
        assert r.bbox is None

    def test_full_result(self):
        r = SignatureDetectionResult(
            signature_found=True,
            confidence=0.92,
            cropped_bytes=b"fake-png",
            bbox=(10, 20, 90, 45),
        )
        assert r.signature_found is True
        assert r.confidence == 0.92
        assert r.bbox == (10, 20, 90, 45)


# ---------------------------------------------------------------------------
# _ensure_supported_format
# ---------------------------------------------------------------------------

class TestEnsureSupportedFormat:
    def test_png_passthrough(self):
        """PNG should pass through unchanged."""
        png = _make_image(fmt="PNG")
        out = _ensure_supported_format(png)
        assert out is png  # same object reference — no conversion

    def test_jpeg_passthrough(self):
        """JPEG should pass through unchanged."""
        jpeg = _make_image(fmt="JPEG")
        out = _ensure_supported_format(jpeg)
        assert out is jpeg

    def test_webp_converted_to_jpeg(self):
        """WebP should be auto-converted to JPEG."""
        webp = _make_image(fmt="WEBP")
        out = _ensure_supported_format(webp)
        assert out is not webp
        # Verify the output is valid JPEG
        img = Image.open(io.BytesIO(out))
        assert img.format == "JPEG"


# ---------------------------------------------------------------------------
# _detect_ink_region (Strategy C)
# ---------------------------------------------------------------------------

class TestDetectInkRegion:
    def test_detects_dark_strokes_on_white(self):
        """A white image with a dark horizontal stroke should be detected."""
        img_bytes = _make_image(200, 100, "PNG", "white")
        result = _detect_ink_region(img_bytes, padding=5)
        assert result is not None
        confidence, cropped, bbox = result
        assert confidence > 0
        assert isinstance(cropped, bytes)
        assert len(bbox) == 4

    def test_blank_image_returns_none(self):
        """A solid white image with no strokes should return None."""
        blank = _make_blank_image(200, 100, "white")
        result = _detect_ink_region(blank, padding=5)
        assert result is None


# ---------------------------------------------------------------------------
# detect_and_crop_signature — mocked DI client
# ---------------------------------------------------------------------------

class TestDetectAndCropSignature:
    @pytest.mark.asyncio
    async def test_no_endpoint_returns_not_found(self):
        """When DOCUMENT_INTELLIGENCE_ENDPOINT is unset, should return not-found."""
        with patch.dict("os.environ", {}, clear=False):
            # Remove the env var if it exists
            import os
            old = os.environ.pop("DOCUMENT_INTELLIGENCE_ENDPOINT", None)
            try:
                result = await detect_and_crop_signature(b"fake-image")
                assert result.signature_found is False
            finally:
                if old is not None:
                    os.environ["DOCUMENT_INTELLIGENCE_ENDPOINT"] = old

    @pytest.mark.asyncio
    async def test_falls_back_to_ink_detection(self):
        """When DI returns no signatures and no handwriting, Strategy C (ink) should fire."""
        # Build a mock DI result with empty pages and no styles
        mock_page = MagicMock()
        mock_page.signatures = None
        mock_page.words = []
        mock_page.lines = []
        mock_page.page_number = 1
        mock_page.unit = "pixel"
        mock_page.width = 200
        mock_page.height = 100

        mock_result = MagicMock()
        mock_result.pages = [mock_page]
        mock_result.styles = []

        mock_poller = AsyncMock()
        mock_poller.result = AsyncMock(return_value=mock_result)

        mock_client_instance = AsyncMock()
        mock_client_instance.begin_analyze_document = AsyncMock(return_value=mock_poller)
        mock_client_instance.__aenter__ = AsyncMock(return_value=mock_client_instance)
        mock_client_instance.__aexit__ = AsyncMock(return_value=False)

        img_bytes = _make_image(200, 100, "PNG", "white")

        with (
            patch.dict("os.environ", {"DOCUMENT_INTELLIGENCE_ENDPOINT": "https://fake.endpoint.com"}),
            patch(
                "app.services.signature_detection.DocumentIntelligenceClient",
                return_value=mock_client_instance,
            ) if False else
            patch(
                "azure.ai.documentintelligence.aio.DocumentIntelligenceClient",
                return_value=mock_client_instance,
            ),
            patch(
                "app.services.signature_detection._get_credential",
                return_value=MagicMock(),
            ),
        ):
            result = await detect_and_crop_signature(img_bytes)
            # Strategy C should catch the ink in the test image
            assert result.signature_found is True
            assert result.cropped_bytes is not None

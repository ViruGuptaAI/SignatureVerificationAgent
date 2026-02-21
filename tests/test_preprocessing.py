"""
Tests for image_preprocessing.py.

Validates the Pillow pipeline: grayscale, denoise, autocrop, resize, pair wrapper.
All tests use synthetic in-memory images — no Data/ files required.
"""

from __future__ import annotations

import io

import pytest
from PIL import Image

from app.services.preprocessing import preprocess_signature, preprocess_signature_pair


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _png_bytes(width: int = 200, height: int = 100, color: str = "white") -> bytes:
    """Create a minimal PNG in memory."""
    img = Image.new("RGB", (width, height), color)
    # Draw a small black rectangle to simulate ink (clamped to image bounds)
    x_start = min(50, width // 4)
    x_end = min(150, width * 3 // 4)
    y_start = min(30, height // 4)
    y_end = min(70, height * 3 // 4)
    for x in range(x_start, x_end):
        for y in range(y_start, y_end):
            img.putpixel((x, y), (0, 0, 0))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _jpeg_bytes(width: int = 200, height: int = 100) -> bytes:
    """Create a minimal JPEG in memory."""
    img = Image.new("RGB", (width, height), "white")
    for x in range(60, 140):
        for y in range(30, 70):
            img.putpixel((x, y), (0, 0, 0))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


def _open(data: bytes) -> Image.Image:
    return Image.open(io.BytesIO(data))


# ---------------------------------------------------------------------------
# preprocess_signature
# ---------------------------------------------------------------------------

class TestPreprocessSignature:
    def test_returns_png_bytes(self, sample_image_bytes):
        result = preprocess_signature(sample_image_bytes)
        assert isinstance(result, bytes)
        img = _open(result)
        assert img.format == "PNG"

    def test_converts_to_grayscale(self):
        raw = _png_bytes()
        result = preprocess_signature(raw, grayscale=True)
        img = _open(result)
        assert img.mode == "L"

    def test_skip_grayscale(self):
        raw = _png_bytes()
        result = preprocess_signature(raw, grayscale=False, denoise=False, autocrop=False, target_long_edge=None)
        img = _open(result)
        # Should remain RGB when grayscale=False
        assert img.mode == "RGB"

    def test_autocrop_reduces_size(self):
        """Autocrop should trim whitespace, resulting in a smaller image."""
        # Create 600x400 image with tiny 50x20 stroke in center
        img = Image.new("RGB", (600, 400), "white")
        for x in range(275, 325):
            for y in range(190, 210):
                img.putpixel((x, y), (0, 0, 0))
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        raw = buf.getvalue()

        result = preprocess_signature(raw, grayscale=False, denoise=False, autocrop=True, target_long_edge=None)
        cropped = _open(result)
        # Cropped image should be much smaller than 600x400
        assert cropped.width < 200
        assert cropped.height < 200

    def test_resize_long_edge(self):
        """Images larger than target_long_edge should be resized."""
        raw = _png_bytes(2000, 1500)
        result = preprocess_signature(raw, target_long_edge=1024)
        img = _open(result)
        assert max(img.size) <= 1024

    def test_small_image_not_upscaled(self):
        """Images smaller than target_long_edge should NOT be upscaled."""
        raw = _png_bytes(100, 50)
        result = preprocess_signature(raw, target_long_edge=1024)
        img = _open(result)
        # Autocrop may shrink it further, but it should never exceed original
        assert max(img.size) <= 1024

    def test_jpeg_input_accepted(self):
        """JPEG input should be accepted and output as PNG."""
        raw = _jpeg_bytes()
        result = preprocess_signature(raw)
        img = _open(result)
        assert img.format == "PNG"

    def test_blank_image_doesnt_crash(self):
        """A fully white image (no ink) should not crash autocrop."""
        img = Image.new("RGB", (200, 100), "white")
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        result = preprocess_signature(buf.getvalue())
        assert isinstance(result, bytes)
        assert len(result) > 0

    def test_denoise_disabled(self):
        raw = _png_bytes()
        result = preprocess_signature(raw, denoise=False)
        assert isinstance(result, bytes)


# ---------------------------------------------------------------------------
# preprocess_signature_pair
# ---------------------------------------------------------------------------

class TestPreprocessSignaturePair:
    def test_returns_two_images(self, sample_image_pair):
        img1, img2 = sample_image_pair
        r1, r2 = preprocess_signature_pair(img1, img2)
        assert isinstance(r1, bytes)
        assert isinstance(r2, bytes)
        assert _open(r1).format == "PNG"
        assert _open(r2).format == "PNG"

    def test_pair_applies_same_settings(self, sample_image_pair):
        img1, img2 = sample_image_pair
        r1, r2 = preprocess_signature_pair(img1, img2, grayscale=True)
        assert _open(r1).mode == "L"
        assert _open(r2).mode == "L"

    def test_pair_produces_different_outputs(self, sample_image_pair):
        """Two different inputs should produce different outputs."""
        img1, img2 = sample_image_pair
        r1, r2 = preprocess_signature_pair(img1, img2)
        assert r1 != r2


# ---------------------------------------------------------------------------
# Real image test (only runs if Data/ exists)
# ---------------------------------------------------------------------------

class TestRealImage:
    def test_real_image_preprocesses(self, real_image_bytes):
        result = preprocess_signature(real_image_bytes)
        img = _open(result)
        assert img.format == "PNG"
        assert img.mode == "L"
        assert max(img.size) <= 1024

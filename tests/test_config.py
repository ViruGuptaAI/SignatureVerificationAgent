"""
Tests for app/config.py.

Validates constants and logger setup.
"""

from app.config import ALLOWED_TYPES, MAX_IMAGE_SIZE, logger


class TestConstants:
    def test_allowed_types_contains_common_formats(self):
        assert "image/png" in ALLOWED_TYPES
        assert "image/jpeg" in ALLOWED_TYPES
        assert "image/webp" in ALLOWED_TYPES

    def test_allowed_types_excludes_non_images(self):
        assert "application/pdf" not in ALLOWED_TYPES
        assert "text/plain" not in ALLOWED_TYPES

    def test_max_image_size_is_1mb(self):
        assert MAX_IMAGE_SIZE == 1 * 1024 * 1024


class TestLogger:
    def test_logger_name(self):
        assert logger.name == "signature_agent"

    def test_logger_has_handler(self):
        assert len(logger.handlers) >= 1

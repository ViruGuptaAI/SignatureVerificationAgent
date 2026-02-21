"""
Tests for app/config.py.

Validates constants, logger setup, and logs directory.
"""

from pathlib import Path

from app.config import ALLOWED_TYPES, MAX_IMAGE_SIZE, LOGS_DIR, logger


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


class TestLogsDir:
    def test_logs_dir_exists(self):
        assert LOGS_DIR.exists()
        assert LOGS_DIR.is_dir()

    def test_logs_dir_is_writable(self, tmp_path):
        """The actual logs dir should be writable."""
        test_file = LOGS_DIR / ".test_write_check"
        try:
            test_file.write_text("ok")
            assert test_file.read_text() == "ok"
        finally:
            test_file.unlink(missing_ok=True)

    def test_logs_dir_at_project_root(self):
        """logs/ should be at the project root, not inside app/."""
        assert LOGS_DIR.name == "logs"
        # Parent of logs/ should contain app/ and requirements.txt
        project_root = LOGS_DIR.parent
        assert (project_root / "app").exists()

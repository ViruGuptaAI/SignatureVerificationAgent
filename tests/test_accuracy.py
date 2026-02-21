"""
Live accuracy verification against a running backend.

This is a migration of the old verify_accuracy.py into the pytest framework.
Unlike the other test files, this one hits the REAL running server and makes
REAL Azure OpenAI calls — it's slow and costs tokens.

Usage:
    # Start the server first:  python backend.py
    # Then run:
    pytest tests/test_accuracy.py -v -s --tb=short

    # Custom URL:
    pytest tests/test_accuracy.py -v -s --server-url http://localhost:8000

Prerequisites:
    The backend must be running and accessible.
"""

from __future__ import annotations

import os
import time

import httpx
import pytest

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "Data")

# Ground-truth test cases: (image1, image2, expected_match, label)
TEST_CASES: list[tuple[str, str, bool | None, str]] = [
    ("VR1.jpg", "VR2.jpg", True, "VR1 vs VR2 (same person)"),
    ("VR1.jpg", "VF1.jpg", False, "VR1 vs VF1 (forged)"),
    ("VR1.jpg", "AR1.jpg", False, "VR1 vs AR1 (different person)"),
    ("VF1.jpg", "AR1.jpg", False, "VF1 vs AR1 (different person)"),
]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def pytest_addoption(parser):
    parser.addoption("--server-url", default="http://127.0.0.1:8000", help="Backend URL")


@pytest.fixture(scope="module")
def server_url(request):
    return request.config.getoption("--server-url")


@pytest.fixture(scope="module")
def check_server(server_url):
    """Skip all tests in this module if the server is not running."""
    try:
        resp = httpx.get(f"{server_url}/health", timeout=5)
        resp.raise_for_status()
    except Exception:
        pytest.skip(f"Backend not reachable at {server_url} — start it first")


@pytest.fixture(scope="module")
def check_data_dir():
    """Skip if the Data/ folder is missing."""
    if not os.path.isdir(DATA_DIR):
        pytest.skip("Data/ directory not found")


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@pytest.mark.usefixtures("check_server", "check_data_dir")
class TestAccuracy:
    """
    Live accuracy tests.
    
    Each test sends a pair of images to /api/VerifySignature and checks
    whether the model's verdict matches the expected ground truth.
    """

    @pytest.mark.parametrize(
        "img1_name, img2_name, expected, label",
        TEST_CASES,
        ids=[t[3] for t in TEST_CASES],
    )
    def test_pair_with_preprocessing(self, server_url, img1_name, img2_name, expected, label):
        img1_path = os.path.join(DATA_DIR, img1_name)
        img2_path = os.path.join(DATA_DIR, img2_name)

        if not os.path.isfile(img1_path) or not os.path.isfile(img2_path):
            pytest.skip(f"Image file(s) missing for: {label}")

        with open(img1_path, "rb") as f1, open(img2_path, "rb") as f2:
            files = [
                ("image1", (img1_name, f1, "image/jpeg")),
                ("image2", (img2_name, f2, "image/jpeg")),
            ]
            start = time.perf_counter()
            resp = httpx.post(
                f"{server_url}/api/VerifySignature",
                files=files,
                params={"preprocess": "true"},
                timeout=120.0,
            )
            elapsed_ms = (time.perf_counter() - start) * 1000

        assert resp.status_code == 200, f"API returned {resp.status_code}: {resp.text}"
        data = resp.json()
        result = data["result"]

        matched = result["signature_matched"]
        score = result["confidence_score"]
        print(f"\n  {label}: matched={matched}, confidence={score:.2f}, time={elapsed_ms:.0f}ms")

        if expected is not None:
            assert matched == expected, (
                f"{label}: expected matched={expected}, got matched={matched} "
                f"(confidence={score:.2f})\n"
                f"Reasoning: {result['reasoning'][:200]}"
            )

    @pytest.mark.parametrize(
        "img1_name, img2_name, expected, label",
        TEST_CASES,
        ids=[t[3] for t in TEST_CASES],
    )
    def test_pair_without_preprocessing(self, server_url, img1_name, img2_name, expected, label):
        img1_path = os.path.join(DATA_DIR, img1_name)
        img2_path = os.path.join(DATA_DIR, img2_name)

        if not os.path.isfile(img1_path) or not os.path.isfile(img2_path):
            pytest.skip(f"Image file(s) missing for: {label}")

        with open(img1_path, "rb") as f1, open(img2_path, "rb") as f2:
            files = [
                ("image1", (img1_name, f1, "image/jpeg")),
                ("image2", (img2_name, f2, "image/jpeg")),
            ]
            resp = httpx.post(
                f"{server_url}/api/VerifySignature",
                files=files,
                params={"preprocess": "false"},
                timeout=120.0,
            )

        assert resp.status_code == 200
        data = resp.json()
        result = data["result"]

        matched = result["signature_matched"]
        score = result["confidence_score"]
        print(f"\n  {label} [raw]: matched={matched}, confidence={score:.2f}")

        if expected is not None:
            assert matched == expected, (
                f"{label} [raw]: expected matched={expected}, got matched={matched} "
                f"(confidence={score:.2f})"
            )

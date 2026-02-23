"""
Tests for FastAPI endpoints (mocked Azure OpenAI — no real API calls).

Uses the `test_client` fixture from conftest.py which patches the OpenAI client.

Coverage:
- POST /api/VerifySignature  — happy path, bad content type, empty file, oversized file
- POST /api/VerifySignatureBatch — happy path, too few references
- GET  /health               — liveness probe
- GET  /health/ready          — readiness probe (mocked)
"""

from __future__ import annotations

import io

import pytest
from PIL import Image


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _png_bytes(w: int = 80, h: int = 40) -> bytes:
    img = Image.new("RGB", (w, h), "white")
    for i in range(min(w, h)):
        img.putpixel((i, i), (0, 0, 0))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _upload_pair(client, img1: bytes = None, img2: bytes = None, **params):
    """POST /api/VerifySignature with two PNG images."""
    img1 = img1 or _png_bytes()
    img2 = img2 or _png_bytes()
    files = [
        ("image1", ("sig1.png", io.BytesIO(img1), "image/png")),
        ("image2", ("sig2.png", io.BytesIO(img2), "image/png")),
    ]
    return client.post("/api/VerifySignature", files=files, params=params)


# ---------------------------------------------------------------------------
# /api/VerifySignature
# ---------------------------------------------------------------------------

class TestVerifySignature:
    def test_happy_path(self, test_client):
        resp = _upload_pair(test_client)
        assert resp.status_code == 200
        data = resp.json()
        assert "result" in data
        assert "signature_matched" in data["result"]
        assert "confidence_score" in data["result"]
        assert "reasoning" in data["result"]
        assert "timing" in data
        assert "elapsed_ms" in data

    def test_returns_filenames(self, test_client):
        resp = _upload_pair(test_client)
        data = resp.json()
        assert data["image1"] == "sig1.png"
        assert data["image2"] == "sig2.png"

    def test_bad_content_type_rejected(self, test_client):
        files = [
            ("image1", ("sig.png", io.BytesIO(_png_bytes()), "image/png")),
            ("image2", ("doc.pdf", io.BytesIO(b"fakepdf"), "application/pdf")),
        ]
        resp = test_client.post("/api/VerifySignature", files=files)
        assert resp.status_code == 400
        assert "unsupported content type" in resp.json()["detail"].lower()

    def test_empty_image_rejected(self, test_client):
        files = [
            ("image1", ("sig.png", io.BytesIO(_png_bytes()), "image/png")),
            ("image2", ("empty.png", io.BytesIO(b""), "image/png")),
        ]
        resp = test_client.post("/api/VerifySignature", files=files)
        assert resp.status_code == 400

    def test_oversized_image_rejected(self, test_client):
        # 2 MB of zeros (will be rejected before hitting the model)
        big = b"\x00" * (2 * 1024 * 1024)
        files = [
            ("image1", ("sig.png", io.BytesIO(_png_bytes()), "image/png")),
            ("image2", ("big.png", io.BytesIO(big), "image/png")),
        ]
        resp = test_client.post("/api/VerifySignature", files=files)
        assert resp.status_code == 413

    def test_preprocess_false(self, test_client):
        resp = _upload_pair(test_client, preprocess="false")
        assert resp.status_code == 200

    def test_model_param(self, test_client):
        resp = _upload_pair(test_client, model="gpt-4.1")
        assert resp.status_code == 200


# ---------------------------------------------------------------------------
# /api/VerifySignatureBatch
# ---------------------------------------------------------------------------

class TestVerifySignatureBatch:
    def _post_batch(self, client, num_refs: int = 3):
        """Helper to POST a batch with N reference images."""
        files = [("test_image", ("test.png", io.BytesIO(_png_bytes()), "image/png"))]
        for i in range(num_refs):
            field = f"ref_{i + 1}"
            files.append((field, (f"ref{i + 1}.png", io.BytesIO(_png_bytes()), "image/png")))
        return client.post("/api/VerifySignatureBatch", files=files)

    def test_happy_path(self, test_client):
        resp = self._post_batch(test_client, num_refs=3)
        assert resp.status_code == 200
        data = resp.json()
        assert "verdict" in data
        assert "individual_results" in data
        assert "request_id" in data
        assert data["verdict"]["decision_method"] == "Majority Vote"

    def test_two_refs_minimum(self, test_client):
        resp = self._post_batch(test_client, num_refs=2)
        assert resp.status_code == 200

    def test_one_ref_rejected(self, test_client):
        """Must have at least 2 reference images. ref_2 is required, so FastAPI returns 422."""
        files = [
            ("test_image", ("test.png", io.BytesIO(_png_bytes()), "image/png")),
            ("ref_1", ("ref1.png", io.BytesIO(_png_bytes()), "image/png")),
        ]
        resp = test_client.post("/api/VerifySignatureBatch", files=files)
        assert resp.status_code in (400, 422)

    def test_batch_result_count(self, test_client):
        resp = self._post_batch(test_client, num_refs=4)
        data = resp.json()
        assert len(data["individual_results"]) == 4


# ---------------------------------------------------------------------------
# /health & /health/ready
# ---------------------------------------------------------------------------

class TestHealth:
    def test_liveness(self, test_client):
        resp = test_client.get("/health")
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"

    def test_readiness(self, test_client):
        resp = test_client.get("/health/ready")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ready"
        assert "azure_openai" in data["checks"]
        assert "blob_storage" in data["checks"]

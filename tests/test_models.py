"""
Tests for Pydantic models in app/models.py.

Validates serialization, deserialization, and field constraints.
"""

import pytest
from pydantic import ValidationError

from app.models import (
    SignatureDetectionInfo,
    SignatureResult,
    TimingMetrics,
    CompareResponse,
    IndividualResult,
    BatchVerdict,
    BatchCompareResponse,
)


# ---------------------------------------------------------------------------
# SignatureDetectionInfo
# ---------------------------------------------------------------------------

class TestSignatureDetectionInfo:
    def test_valid_creation(self):
        info = SignatureDetectionInfo(signature_found=True, detection_confidence=0.9, was_cropped=True, crop_bbox=(10, 20, 90, 45))
        assert info.signature_found is True
        assert info.detection_confidence == 0.9
        assert info.was_cropped is True
        assert info.crop_bbox == (10, 20, 90, 45)

    def test_confidence_bounds(self):
        with pytest.raises(ValidationError):
            SignatureDetectionInfo(signature_found=True, detection_confidence=1.5)
        with pytest.raises(ValidationError):
            SignatureDetectionInfo(signature_found=True, detection_confidence=-0.1)


# ---------------------------------------------------------------------------
# SignatureResult
# ---------------------------------------------------------------------------

class TestSignatureResult:
    def test_valid_creation(self):
        r = SignatureResult(signature_matched=True, confidence_score=0.85, reasoning="Similar strokes")
        assert r.signature_matched is True
        assert r.confidence_score == 0.85
        assert "strokes" in r.reasoning

    def test_confidence_score_bounds(self):
        """Score must be between 0.0 and 1.0."""
        with pytest.raises(ValidationError):
            SignatureResult(signature_matched=False, confidence_score=1.5, reasoning="x")
        with pytest.raises(ValidationError):
            SignatureResult(signature_matched=False, confidence_score=-0.1, reasoning="x")

    def test_edge_scores(self):
        """0.0 and 1.0 are valid boundary values."""
        r0 = SignatureResult(signature_matched=False, confidence_score=0.0, reasoning="Different")
        r1 = SignatureResult(signature_matched=True, confidence_score=1.0, reasoning="Identical")
        assert r0.confidence_score == 0.0
        assert r1.confidence_score == 1.0

    def test_json_roundtrip(self):
        r = SignatureResult(signature_matched=True, confidence_score=0.9, reasoning="Test")
        data = r.model_dump_json()
        restored = SignatureResult.model_validate_json(data)
        assert restored == r


# ---------------------------------------------------------------------------
# TimingMetrics
# ---------------------------------------------------------------------------

class TestTimingMetrics:
    def test_valid_creation(self):
        t = TimingMetrics(stream_opened_ms=50.0, ttft_ms=120.0, ttfb_ms=170.0, ttlb_ms=500.0)
        assert t.stream_opened_ms == 50.0

    def test_json_roundtrip(self):
        t = TimingMetrics(stream_opened_ms=10, ttft_ms=20, ttfb_ms=30, ttlb_ms=100)
        restored = TimingMetrics.model_validate_json(t.model_dump_json())
        assert restored == t


# ---------------------------------------------------------------------------
# CompareResponse
# ---------------------------------------------------------------------------

class TestCompareResponse:
    def test_full_response(self):
        resp = CompareResponse(
            request_id="00000000-0000-0000-0000-000000000001",
            image1="ref.jpg",
            image2="test.jpg",
            result=SignatureResult(signature_matched=True, confidence_score=0.8, reasoning="ok"),
            usage={"input_tokens": 100, "output_tokens": 50, "reasoning_tokens": 0, "total_tokens": 150},
            timing=TimingMetrics(stream_opened_ms=10, ttft_ms=20, ttfb_ms=30, ttlb_ms=100),
            elapsed_ms=100.5,
        )
        assert resp.image1 == "ref.jpg"
        assert resp.result.signature_matched is True

    def test_usage_optional(self):
        resp = CompareResponse(
            request_id="00000000-0000-0000-0000-000000000002",
            image1="a.png", image2="b.png",
            result=SignatureResult(signature_matched=False, confidence_score=0.1, reasoning="no"),
            usage=None,
            timing=TimingMetrics(stream_opened_ms=10, ttft_ms=20, ttfb_ms=30, ttlb_ms=100),
            elapsed_ms=50,
        )
        assert resp.usage is None


# ---------------------------------------------------------------------------
# IndividualResult
# ---------------------------------------------------------------------------

class TestIndividualResult:
    def test_successful_result(self):
        r = IndividualResult(
            reference_filename="ref1.jpg", test_filename="test.jpg",
            signature_matched=True, confidence_score=0.75,
            reasoning="Similar", elapsed_ms=1200.0,
        )
        assert r.error is None

    def test_error_result(self):
        r = IndividualResult(
            reference_filename="ref1.jpg", test_filename="test.jpg",
            signature_matched=False, confidence_score=0.0,
            reasoning="", elapsed_ms=0.0, error="Timeout",
        )
        assert r.error == "Timeout"


# ---------------------------------------------------------------------------
# BatchVerdict
# ---------------------------------------------------------------------------

class TestBatchVerdict:
    def test_majority_match(self):
        v = BatchVerdict(
            signature_matched=True, avg_confidence=0.82,
            match_ratio="7/10", reasoning="Summary text",
        )
        assert v.decision_method == "Majority Vote"
        assert v.inconclusive is False

    def test_inconclusive_flag(self):
        v = BatchVerdict(
            signature_matched=False, avg_confidence=0.5,
            match_ratio="3/6", reasoning="Split", inconclusive=True,
        )
        assert v.inconclusive is True


# ---------------------------------------------------------------------------
# BatchCompareResponse
# ---------------------------------------------------------------------------

class TestBatchCompareResponse:
    def test_full_batch_response(self):
        resp = BatchCompareResponse(
            request_id="abc-123",
            verdict=BatchVerdict(
                signature_matched=True, avg_confidence=0.85,
                match_ratio="4/5", reasoning="Clear match",
            ),
            individual_results=[
                IndividualResult(
                    reference_filename=f"ref{i}.jpg", test_filename="test.jpg",
                    signature_matched=True, confidence_score=0.85,
                    reasoning="ok", elapsed_ms=1000.0,
                )
                for i in range(5)
            ],
            total_usage={"input_tokens": 5000, "output_tokens": 500, "reasoning_tokens": 0, "total_tokens": 5500},
            elapsed_ms=5000.0,
        )
        assert len(resp.individual_results) == 5
        assert resp.request_id == "abc-123"

    def test_json_roundtrip(self):
        resp = BatchCompareResponse(
            request_id="uuid-test",
            verdict=BatchVerdict(
                signature_matched=False, avg_confidence=0.3,
                match_ratio="1/4", reasoning="Low match",
            ),
            individual_results=[],
            elapsed_ms=100.0,
        )
        restored = BatchCompareResponse.model_validate_json(resp.model_dump_json())
        assert restored.request_id == "uuid-test"

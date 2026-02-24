from pydantic import BaseModel, Field


class SignatureDetectionInfo(BaseModel):
    """Metadata from Azure Document Intelligence signature detection."""
    signature_found: bool = Field(..., description="Whether Document Intelligence detected a signature in the image")
    detection_confidence: float = Field(0.0, ge=0.0, le=1.0, description="Confidence of the signature detection")
    was_cropped: bool = Field(False, description="Whether the image was cropped to the signature region")
    crop_bbox: tuple[int, int, int, int] | None = Field(None, description="Pixel bounding box (left, top, right, bottom) if cropped")


class SignatureResult(BaseModel):
    """Structured response from the signature comparison agent."""
    signature_matched: bool
    confidence_score: float = Field(..., ge=0.0, le=1.0)
    reasoning: str


class TimingMetrics(BaseModel):
    """Latency metrics captured during the streaming API call."""
    stream_opened_ms: float = Field(..., description="Time to establish the stream connection")
    ttft_ms: float = Field(..., description="Time from stream open to first content token")
    ttfb_ms: float = Field(..., description="Time from request start to first content byte")
    ttlb_ms: float = Field(..., description="Time from request start to last byte received")


class CompareResponse(BaseModel):
    """Full API response including usage metadata."""
    request_id: str = Field(..., description="Unique UUID for this invocation")
    image1: str = Field(..., description="Filename of the first image sent to the model")
    image2: str = Field(..., description="Filename of the second image sent to the model")
    result: SignatureResult
    usage: dict | None = None
    timing: TimingMetrics
    elapsed_ms: float
    cost_inr: float | None = Field(None, description="Estimated cost of this call in INR")
    signature_detection: dict[str, SignatureDetectionInfo] | None = Field(
        None, description="Document Intelligence detection results keyed by image name (only when detect_signature=True)"
    )


class IndividualResult(BaseModel):
    """Result of a single reference-vs-test comparison."""
    reference_filename: str
    test_filename: str
    signature_matched: bool
    confidence_score: float = Field(..., ge=0.0, le=1.0)
    reasoning: str
    usage: dict | None = None
    elapsed_ms: float
    error: str | None = None
    cost_inr: float | None = Field(None, description="Estimated cost of this call in INR")
    signature_detection: dict[str, SignatureDetectionInfo] | None = Field(
        None, description="Document Intelligence detection results keyed by image name"
    )


class BatchVerdict(BaseModel):
    """Aggregated verdict from multiple reference comparisons."""
    signature_matched: bool
    avg_confidence: float = Field(..., ge=0.0, le=1.0)
    match_ratio: str = Field(..., description="e.g. '7/10'")
    decision_method: str = "Majority Vote"
    reasoning: str = Field(..., description="LLM-generated summary of all individual reasonings")
    inconclusive: bool = False


class BatchCompareResponse(BaseModel):
    """Full batch API response."""
    request_id: str = Field(..., description="Unique UUID for this invocation")
    verdict: BatchVerdict
    individual_results: list[IndividualResult]
    total_usage: dict | None = None
    elapsed_ms: float
    total_cost_inr: float | None = Field(None, description="Total estimated cost in INR for all comparisons + summary")

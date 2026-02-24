# API Reference

Interactive docs available at **http://localhost:8000/docs** (Swagger UI) when the server is running.

---

## `POST /api/VerifySignature`

Compare two signature images.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `image1` | `file` (required) | — | First signature image |
| `image2` | `file` (required) | — | Second signature image |
| `preprocess` | `bool` | `true` | Apply image preprocessing pipeline |
| `detect_signature` | `bool` | `false` | Use Document Intelligence to detect and crop signatures before comparison |
| `model` | `string` | `gpt-4.1` | Model: `gpt-4.1`, `gpt-5-mini` |
| `reasoning_effort` | `string` | `medium` | For GPT-5 models: `low`, `medium`, `high` |

**Response:**

```json
{
  "image1": "reference.png",
  "image2": "test.png",
  "result": {
    "signature_matched": true,
    "confidence_score": 0.85,
    "reasoning": "Both signatures exhibit consistent stroke flow..."
  },
  "usage": {
    "input_tokens": 1200,
    "output_tokens": 350,
    "reasoning_tokens": 0,
    "cached_tokens": 0,
    "total_tokens": 1550
  },
  "timing": {
    "stream_opened_ms": 120.5,
    "ttft_ms": 450.2,
    "ttfb_ms": 570.7,
    "ttlb_ms": 2100.3
  },
  "elapsed_ms": 2250.1,
  "cost_inr": 0.0523,
  "signature_detection": null
}
```

---

## `POST /api/VerifySignatureBatch`

Compare one test signature against 2–10 references using majority vote.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `test_image` | `file` (required) | — | The test signature to verify |
| `ref_1` | `file` (required) | — | Reference signature 1 |
| `ref_2` | `file` (required) | — | Reference signature 2 |
| `ref_3` – `ref_10` | `file` (optional) | — | Additional references (up to 10 total) |
| `preprocess` | `bool` | `true` | Apply image preprocessing |
| `detect_signature` | `bool` | `false` | Use Document Intelligence to detect and crop signatures |
| `model` | `string` | `gpt-4.1` | Model to use |
| `reasoning_effort` | `string` | `medium` | Reasoning effort for GPT-5 |

**Response:**

```json
{
  "request_id": "a1b2c3d4-...",
  "verdict": {
    "signature_matched": true,
    "avg_confidence": 0.83,
    "match_ratio": "4/5",
    "decision_method": "majority_vote",
    "reasoning": "The majority of comparisons indicate consistent stroke patterns...",
    "inconclusive": false
  },
  "individual_results": [
    {
      "reference_filename": "ref_1.png",
      "test_filename": "test.png",
      "signature_matched": true,
      "confidence_score": 0.87,
      "reasoning": "...",
      "usage": { "input_tokens": 1200, "output_tokens": 350, "reasoning_tokens": 0, "cached_tokens": 0, "total_tokens": 1550 },
      "elapsed_ms": 2100.5,
      "error": null,
      "cost_inr": 0.0523
    }
  ],
  "total_usage": { "input_tokens": 6000, "output_tokens": 1750, "reasoning_tokens": 0, "cached_tokens": 0, "total_tokens": 7750 },
  "elapsed_ms": 4500.2,
  "total_cost_inr": 1.06
}
```

When `detect_signature=true`, both the single and batch responses include a `signature_detection` field:

```json
"signature_detection": {
  "reference.png": {
    "signature_found": true,
    "detection_confidence": 1.0,
    "was_cropped": true,
    "crop_bbox": [306, 383, 498, 450]
  },
  "test.png": {
    "signature_found": true,
    "detection_confidence": 0.88,
    "was_cropped": true,
    "crop_bbox": [0, 0, 800, 401]
  }
}
```

---

## `GET /health`

Liveness probe. Returns `{"status": "ok"}`.

---

## `GET /health/ready`

Readiness probe — checks Azure OpenAI reachability and logs directory write access.

| Status | Meaning |
|---|---|
| `200` | All checks pass |
| `503` | Degraded — one or more checks failed |

---

## `GET /api/logs/{request_id}`

Retrieve a previously saved result by UUID request ID.

Returns the same JSON that was saved when the comparison originally ran.

| Status | Meaning |
|---|---|
| `200` | Log found and returned |
| `400` | Invalid request ID format (not a UUID) |
| `404` | No log found for the given ID |

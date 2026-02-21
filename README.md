# Signature Matching Agent

An API service for automated handwritten signature verification, powered by Azure OpenAI vision models. Given two or more signature images, the agent evaluates visual similarity across 8 comparison dimensions — stroke flow, pressure patterns, curvature, slant, loops, alignment, letterform consistency, and structural proportions — and returns a structured verdict with confidence scoring and detailed reasoning.

Built with FastAPI, async Azure OpenAI streaming, and Entra ID authentication. Designed for integration into document processing workflows where manual signature review is a bottleneck.

---

## Features

- **Single-pair comparison** — Upload two signature images, get a match/no-match verdict with confidence score and detailed reasoning.
- **Batch comparison** — Compare one test signature against 2–10 reference signatures with majority-vote aggregation and an LLM-generated summary.
- **Image preprocessing** — Optional Pillow pipeline (grayscale → denoise → autocrop → resize) offloaded to a thread pool so the event loop stays unblocked.
- **Model selection** — Choose between `gpt-4.1`, `gpt-5.2`, or `gpt-5-mini` per request.
- **Streaming with timeout** — Responses are streamed from Azure OpenAI with a 60-second timeout to prevent hanging requests.
- **Health probes** — Liveness (`/health`) and readiness (`/health/ready`) endpoints for container orchestration.
- **Structured output** — Pydantic models enforce JSON schema on every response; results are logged to `logs/` as JSON files.
- **Azure AD authentication** — Uses `ManagedIdentityCredential` in Azure, falls back to `AzureCliCredential` locally.

---

## Project Structure

```
SignatureMatchingAgent/
├── backend.py                  # Thin entry point (backward compat)
├── requirements.txt            # Python dependencies
├── .env                        # Environment variables (not committed)
├── .gitignore
│
├── app/                        # Application package
│   ├── __init__.py
│   ├── main.py                 # FastAPI app, lifespan, CORS, router wiring
│   ├── config.py               # Logger, constants (ALLOWED_TYPES, MAX_IMAGE_SIZE)
│   ├── azure_client.py         # AsyncAzureOpenAI singleton, credential, base64 util
│   ├── models.py               # Pydantic models (request/response contracts)
│   ├── prompts.py              # System instructions & prompt templates
│   │
│   ├── routes/
│   │   ├── __init__.py
│   │   ├── compare.py          # POST /api/VerifySignature
│   │   ├── batch.py            # POST /api/VerifySignatureBatch
│   │   └── health.py           # GET /health, GET /health/ready
│   │
│   └── services/
│       ├── __init__.py
│       ├── comparison.py       # Core compare_signatures logic (streaming, parsing)
│       └── preprocessing.py    # Pillow image preprocessing pipeline
│
├── tests/                      # Test suite (48 tests)
│   ├── __init__.py
│   ├── conftest.py             # Shared fixtures, mock OpenAI client
│   ├── test_models.py          # 14 unit tests — Pydantic models
│   ├── test_preprocessing.py   # 12 unit tests — Pillow pipeline
│   ├── test_config.py          # 7 unit tests — constants, logger, logs dir
│   ├── test_api.py             # 13 integration tests — API routes (mocked)
│   └── test_accuracy.py        # 8 E2E tests — live server + real images
│
├── Data/                       # Test signature images (not committed)
└── Logs/                       # JSON result logs (not committed)
```

---

## Prerequisites

| Requirement | Details |
|---|---|
| **Python** | 3.12+ |
| **Azure OpenAI** | A deployed resource with a vision-capable model (GPT-4.1 or later) |
| **Azure CLI** | Logged in (`az login`) for local credential fallback |
| **Signature images** | Place test images in `Data/` for E2E tests |

---

## Setup

### 1. Clone the repository

```bash
git clone <repo-url>
cd SignatureMatchingAgent
```

### 2. Create and activate a virtual environment

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure environment variables

Create a `.env` file in the project root:

```env
AZURE_ENDPOINT=https://<your-resource-name>.openai.azure.com/
```

> The service authenticates via Azure AD (Entra ID) — no API key needed.  
> Locally it uses `AzureCliCredential` (requires `az login`).  
> In Azure App Service it uses `ManagedIdentityCredential` automatically.

### 5. Run the server

```bash
python backend.py
```

The server starts on **http://0.0.0.0:8000** with 4 Uvicorn workers.

Alternatively:

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 4
```

---

## API Reference

### `POST /api/VerifySignature`

Compare two signature images.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `image1` | `file` (required) | — | First signature image |
| `image2` | `file` (required) | — | Second signature image |
| `preprocess` | `bool` | `true` | Apply image preprocessing pipeline |
| `model` | `string` | `gpt-4.1` | Model: `gpt-4.1`, `gpt-5.2`, `gpt-5-mini` |
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
  "usage": { "input_tokens": 1200, "output_tokens": 350 },
  "timing": {
    "stream_opened_ms": 120.5,
    "ttft_ms": 450.2,
    "ttfb_ms": 570.7,
    "ttlb_ms": 2100.3
  },
  "elapsed_ms": 2250.1
}
```

### `POST /api/VerifySignatureBatch`

Compare one test signature against 2–10 references using majority vote.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `test_image` | `file` (required) | — | The test signature to verify |
| `ref_1` | `file` (required) | — | Reference signature 1 |
| `ref_2` | `file` (required) | — | Reference signature 2 |
| `ref_3` – `ref_10` | `file` (optional) | — | Additional references |
| `preprocess` | `bool` | `true` | Apply image preprocessing |
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
      "usage": { ... },
      "elapsed_ms": 2100.5,
      "error": null
    }
  ],
  "total_usage": { "input_tokens": 6000, "output_tokens": 1750 },
  "elapsed_ms": 4500.2
}
```

### `GET /health`

Liveness probe — returns `{"status": "ok"}` if the server is running.

### `GET /health/ready`

Readiness probe — checks Azure OpenAI reachability and logs directory write access. Returns `200` if all checks pass, `503` if degraded.

---

## Image Preprocessing Pipeline

When `preprocess=true` (default), each uploaded image goes through:

1. **EXIF orientation fix** — corrects rotated phone photos
2. **Grayscale conversion** — removes colour noise, preserves pressure info
3. **Gaussian blur** (σ=0.5) — smooths paper texture without erasing thin strokes
4. **Auto-crop** — trims whitespace with 20px padding around the signature
5. **Resize** — scales to 1024px on the longest edge (no upscaling)

The pipeline runs in a thread pool via `asyncio.run_in_executor` to avoid blocking the event loop.

---

## Testing

### Run offline tests (no server or Azure needed)

```bash
python -m pytest tests/test_models.py tests/test_preprocessing.py tests/test_config.py tests/test_api.py -v
```

This runs **48 tests** — unit tests for models, preprocessing, and config, plus integration tests for all API endpoints with a mocked OpenAI client.

### Run E2E accuracy tests (requires running server + `Data/` images)

```bash
# Terminal 1: start the server
python backend.py

# Terminal 2: run accuracy tests
python -m pytest tests/test_accuracy.py -v -s
```

These hit the live server with real signature images from `Data/` and validate against expected match/no-match outcomes. Tests are skipped automatically if the server isn't running or `Data/` is missing.

### Run everything at once

```bash
python -m pytest tests/ -v
```

---

## Environment Variables

| Variable | Required | Default | Description |
|---|---|---|---|
| `AZURE_ENDPOINT` | **Yes** | — | Azure OpenAI resource endpoint URL |
| `AZURE_MAX_RETRIES` | No | `3` | Max retries for Azure OpenAI API calls |
| `WEBSITE_SITE_NAME` | Auto | — | Set automatically in Azure App Service; triggers `ManagedIdentityCredential` |

---

## Constraints & Limits

| Constraint | Value |
|---|---|
| Max image size | 1 MB per file |
| Allowed image types | PNG, JPEG, WebP, GIF |
| Batch references | 2–10 per request |
| Stream timeout | 60 seconds |
| Confidence score range | 0.0 – 1.0 |
| OpenAI settings | `temperature=0`, `store=false` |

---

## Swagger UI

Once the server is running, open **http://localhost:8000/docs** for the interactive API documentation. All endpoints, parameters, and response schemas are documented there.

---

## Contributing

1. Create a feature branch from `main`
2. Make changes and ensure all 48 tests pass
3. Submit a pull request

```bash
# Verify before pushing
python -m pytest tests/test_models.py tests/test_preprocessing.py tests/test_config.py tests/test_api.py -v
```

# Architecture & Project Structure

```
SignatureMatchingAgent/
├── backend.py                  # Entry point — starts Uvicorn with 4 workers
├── requirements.txt
├── .env                        # Environment variables (not committed)
│
├── app/
│   ├── main.py                 # FastAPI app, lifespan, CORS, router wiring
│   ├── config.py               # Logger, constants, model pricing, cost calculator
│   ├── azure_client.py         # AsyncAzureOpenAI singleton, credential, base64 encoding
│   ├── models.py               # Pydantic models (request/response contracts)
│   ├── prompts.py              # System instructions & prompt templates
│   │
│   ├── routes/
│   │   ├── compare.py          # POST /api/VerifySignature
│   │   ├── batch.py            # POST /api/VerifySignatureBatch
│   │   ├── health.py           # GET /health, GET /health/ready
│   │   └── logs.py             # GET /api/logs/{request_id}
│   │
│   └── services/
│       ├── comparison.py           # Core compare_signatures — streaming, parsing, cost
│       ├── preprocessing.py        # Pillow image preprocessing pipeline
│       └── signature_detection.py  # Azure Document Intelligence signature detection & cropping
│
├── static/                     # Frontend (served at /static)
│   ├── index.html              # SPA — 1:1 verify, verify against references, audit log
│   ├── app.js                  # Tab switching, uploads, API calls, result rendering
│   └── style.css               # Dark theme, tooltips, responsive layout
│
├── tests/                      # 55+ tests
│   ├── conftest.py             # Shared fixtures, mock OpenAI client
│   ├── test_models.py          # 14 unit tests — Pydantic models
│   ├── test_preprocessing.py   # 12 unit tests — Pillow pipeline
│   ├── test_config.py          # 7 unit tests — constants, logger, logs dir
│   ├── test_api.py             # 13 integration tests — API routes (mocked)
│   └── test_accuracy.py        # 8 E2E tests — live server + real images
│
├── Data/                       # Test signature images (not committed)
├── Logs/                       # JSON result logs (not committed)
└── docs/                       # Documentation
```

---

## Request Flow

### 1:1 Verify

```
Client → POST /api/VerifySignature
       → compare.py (route) → comparison.py (service)
       → [optional] signature_detection.py (Azure Document Intelligence — detect & crop)
       → [optional] preprocessing.py (thread pool — grayscale, denoise, autocrop, resize)
       → Azure OpenAI streaming call
       → parse structured JSON → calculate cost
       → return CompareResponse
```

### Verify Against References

```
Client → POST /api/VerifySignatureBatch
       → batch.py (route)
       → N concurrent compare_signatures() calls (semaphore-gated)
       → majority vote aggregation
       → LLM summary call
       → save JSON log to Logs/
       → return BatchCompareResponse
```

---

## Key Design Decisions

- **Semaphore on LLM calls** — `MAX_CONCURRENT_LLM_CALLS` (default 20) prevents token-per-minute overload when many references run in parallel.
- **Streaming with timeout** — Every LLM call is streamed with a 60s timeout; timing metrics (TTFB, TTFT, TTLB) are captured per call.
- **Optional signature detection** — When `detect_signature=true`, Azure Document Intelligence analyses each image to detect and crop signature regions before preprocessing. Uses a 3-strategy cascade: page.signatures → handwriting style → ink-region fallback. Unsupported formats (e.g. WebP) are auto-converted to JPEG.
- **Thread pool for preprocessing** — CPU-bound Pillow work runs via `run_in_executor` so the async event loop stays unblocked.
- **Structured output** — JSON schema enforcement on every LLM response via the Responses API `text.format` parameter.
- **Cost tracking** — Token usage is extracted from every response, cost is computed from `.env` pricing, and both are surfaced in the API response and UI.

---

## Authentication

| Environment | Credential |
|---|---|
| Local dev | `AzureCliCredential` — requires `az login` |
| Azure App Service | `ManagedIdentityCredential` — automatic via `WEBSITE_SITE_NAME` env var |

The OpenAI client singleton is created once at app startup (in `azure_client.py`) and reused across all requests.

Document Intelligence uses a separate per-request client (in `signature_detection.py`) with the same credential strategy.

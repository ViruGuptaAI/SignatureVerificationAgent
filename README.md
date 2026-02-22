# Signature Matching Agent

Automated handwritten signature verification using Azure OpenAI vision models. Upload two or more signature images → get a match/no-match verdict with confidence scoring and detailed reasoning.

Built with FastAPI + async Azure OpenAI streaming + Entra ID auth.

---

## Quickstart

```bash
git clone <repo-url> && cd SignatureMatchingAgent
python -m venv .venv && .venv\Scripts\activate   # Windows
pip install -r requirements.txt
```

Create `.env`:

```env
AZURE_ENDPOINT=https://<your-resource>.openai.azure.com/
```

Run:

```bash
python backend.py          # starts on http://localhost:8000
```

- **Web UI** → http://localhost:8000/static/index.html
- **Swagger** → http://localhost:8000/docs

> Auth is via Azure AD — no API key. Locally uses `AzureCliCredential` (`az login`), in Azure uses `ManagedIdentityCredential`.

---

## Features

- **1:1 verify** — Two images → verdict + confidence + reasoning
- **Verify against references** — One test vs 2–10 references, majority-vote aggregation, LLM summary
- **Image preprocessing** — Grayscale → denoise → autocrop → resize (optional, on by default)
- **Model selection** — `gpt-4.1` or `gpt-5-mini` per request
- **Token & cost tracking** — Per-call token breakdown and INR cost estimation
- **Audit log** — Look up any past result by request ID
- **Health probes** — `/health` and `/health/ready`

---

## Documentation

| Topic | Link |
|---|---|
| API reference | [docs/api.md](docs/api.md) |
| Architecture & project structure | [docs/architecture.md](docs/architecture.md) |
| Image preprocessing pipeline | [docs/preprocessing.md](docs/preprocessing.md) |
| Configuration & environment variables | [docs/configuration.md](docs/configuration.md) |
| Testing | [docs/testing.md](docs/testing.md) |
| Contributing | [CONTRIBUTING.md](CONTRIBUTING.md) |

---

## Constraints

| Limit | Value |
|---|---|
| Max image size | 1 MB |
| Allowed types | PNG, JPEG, WebP, GIF |
| References per request | 2–10 |
| Stream timeout | 60 s |
| Confidence range | 0.0–1.0 |
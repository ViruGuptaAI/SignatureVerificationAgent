# Configuration & Environment Variables

All configuration is via environment variables, loaded from `.env` at startup.

---

## Required

| Variable | Description |
|---|---|
| `AZURE_ENDPOINT` | Azure OpenAI resource endpoint URL (e.g. `https://my-resource.openai.azure.com/`) |

---

## Optional

| Variable | Default | Description |
|---|---|---|
| `AZURE_MAX_RETRIES` | `3` | Max retries for Azure OpenAI API calls |
| `MAX_CONCURRENT_LLM_CALLS` | `20` | Semaphore limit for parallel LLM calls |
| `STREAM_TIMEOUT_SECONDS` | `60` | Timeout per streaming LLM call |

---

## Auto-detected

| Variable | Description |
|---|---|
| `WEBSITE_SITE_NAME` | Set automatically in Azure App Service; presence triggers `ManagedIdentityCredential` instead of `AzureCliCredential` |

---

## Model Pricing (INR per 1M tokens)

Used for cost estimation displayed in the UI. Set to `0` to disable.

| Variable | Description |
|---|---|
| `MODEL_GPT41_INPUT` | GPT-4.1 input token price |
| `MODEL_GPT41_CACHED` | GPT-4.1 cached input token price |
| `MODEL_GPT41_OUTPUT` | GPT-4.1 output token price |
| `MODEL_GPT52_INPUT` | GPT-5.2 input token price |
| `MODEL_GPT52_CACHED` | GPT-5.2 cached input token price |
| `MODEL_GPT52_OUTPUT` | GPT-5.2 output token price |
| `MODEL_GPT5MINI_INPUT` | GPT-5 Mini input token price |
| `MODEL_GPT5MINI_CACHED` | GPT-5 Mini cached input token price |
| `MODEL_GPT5MINI_OUTPUT` | GPT-5 Mini output token price |

---

## Example `.env`

```env
AZURE_ENDPOINT=https://my-resource.openai.azure.com/

# Pricing (INR per 1M tokens)
MODEL_GPT41_INPUT=168
MODEL_GPT41_CACHED=42
MODEL_GPT41_OUTPUT=672
MODEL_GPT5MINI_INPUT=25.2
MODEL_GPT5MINI_CACHED=6.3
MODEL_GPT5MINI_OUTPUT=100.8
```

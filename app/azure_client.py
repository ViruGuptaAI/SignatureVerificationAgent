import base64
import mimetypes
import os

from openai import AsyncAzureOpenAI
from azure.identity.aio import (
    AzureCliCredential,
    ManagedIdentityCredential,
    get_bearer_token_provider,
)

# ---------------------------------------------------------------------------
# Shared client singleton (set during app lifespan)
# ---------------------------------------------------------------------------

_client: AsyncAzureOpenAI | None = None


def get_client() -> AsyncAzureOpenAI:
    """Return the shared Azure OpenAI client. Raises if called before lifespan init."""
    assert _client is not None, "Azure OpenAI client not initialised — lifespan not started?"
    return _client


def set_client(client: AsyncAzureOpenAI | None) -> None:
    global _client
    _client = client


# ---------------------------------------------------------------------------
# Credential & client builder
# ---------------------------------------------------------------------------


def _get_credential():
    """Use Managed Identity in Azure App Service, fall back to CLI locally."""
    if os.getenv("WEBSITE_SITE_NAME"):
        return ManagedIdentityCredential()
    return AzureCliCredential()


def build_client() -> AsyncAzureOpenAI:
    return AsyncAzureOpenAI(
        azure_endpoint=os.environ["AZURE_ENDPOINT"],
        azure_ad_token_provider=get_bearer_token_provider(
            _get_credential(), "https://cognitiveservices.azure.com/.default"
        ),
        api_version="2025-03-01-preview",
        max_retries=int(os.getenv("AZURE_MAX_RETRIES", 10)),
    )


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

# For image to base64 conversion 
def encode_bytes(data: bytes, filename: str) -> str:
    """Return a base64 data-URI from raw bytes."""
    mime, _ = mimetypes.guess_type(filename)
    mime = mime or "image/png"
    b64 = base64.b64encode(data).decode()
    return f"data:{mime};base64,{b64}"

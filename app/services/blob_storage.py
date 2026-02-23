"""
Azure Blob Storage service for audit-log persistence.

Storage account : configured via BLOB_STORAGE_ACCOUNT (default: sharedblobstorageforai)
Container       : configured via BLOB_CONTAINER       (default: signature-agent-audits)

Uses DefaultAzureCredential (Managed Identity in Azure, CLI locally).
"""

import os

from azure.identity.aio import AzureCliCredential, ManagedIdentityCredential
from azure.storage.blob.aio import BlobServiceClient

from app.config import logger

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

STORAGE_ACCOUNT = os.getenv("BLOB_STORAGE_ACCOUNT", "sharedblobstorageforai")
CONTAINER_NAME = os.getenv("BLOB_CONTAINER", "signature-agent-audits")
ACCOUNT_URL = f"https://{STORAGE_ACCOUNT}.blob.core.windows.net"

# ---------------------------------------------------------------------------
# Singleton client
# ---------------------------------------------------------------------------

_blob_service_client: BlobServiceClient | None = None


def _get_credential():
    """Use Managed Identity in Azure App Service, fall back to CLI locally."""
    if os.getenv("WEBSITE_SITE_NAME"):
        return ManagedIdentityCredential()
    return AzureCliCredential()


def get_blob_service_client() -> BlobServiceClient:
    """Return (and lazily create) the shared BlobServiceClient."""
    global _blob_service_client
    if _blob_service_client is None:
        _blob_service_client = BlobServiceClient(
            account_url=ACCOUNT_URL,
            credential=_get_credential(),
        )
    return _blob_service_client


async def close_blob_client() -> None:
    """Gracefully close the blob client (call during shutdown)."""
    global _blob_service_client
    if _blob_service_client is not None:
        await _blob_service_client.close()
        _blob_service_client = None


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------


async def upload_log(request_id: str, json_text: str) -> None:
    """Upload a JSON audit log as a blob named `{request_id}.json`."""
    client = get_blob_service_client()
    blob = client.get_blob_client(container=CONTAINER_NAME, blob=f"{request_id}.json")
    await blob.upload_blob(
        json_text,
        overwrite=True,
        content_settings=_json_content_settings(),
    )
    logger.info("--- Blob uploaded: %s/%s.json ---", CONTAINER_NAME, request_id)


async def download_log(request_id: str) -> str | None:
    """Download and return a JSON audit log, or None if not found."""
    client = get_blob_service_client()
    blob = client.get_blob_client(container=CONTAINER_NAME, blob=f"{request_id}.json")
    try:
        stream = await blob.download_blob()
        data = await stream.readall()
        return data.decode("utf-8")
    except Exception:
        return None


async def check_blob_health() -> str:
    """Quick health check — verify we can reach the container."""
    client = get_blob_service_client()
    container = client.get_container_client(CONTAINER_NAME)
    props = await container.get_container_properties()
    _ = props.name  # force the awaited result to be used
    return "ok"


# ---------------------------------------------------------------------------
# Internal
# ---------------------------------------------------------------------------

def _json_content_settings():
    from azure.storage.blob import ContentSettings
    return ContentSettings(content_type="application/json")

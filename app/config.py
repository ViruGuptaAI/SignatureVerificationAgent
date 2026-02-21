import logging
from pathlib import Path
from typing import Literal

# ---------------------------------------------------------------------------
# Logger
# ---------------------------------------------------------------------------

logger = logging.getLogger("signature_agent")
logger.setLevel(logging.DEBUG)
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(asctime)s [%(name)s] %(message)s"))
    logger.addHandler(handler)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ALLOWED_TYPES: set[str] = {"image/png", "image/jpeg", "image/jpg", "image/webp", "image/gif"}
ALLOWED_MODELS = Literal["gpt-4.1", "gpt-5.2", "gpt-5-mini"]
MAX_IMAGE_SIZE: int = 1 * 1024 * 1024  # 1 MB

# ---------------------------------------------------------------------------
# Logs directory (project-root/logs)
# ---------------------------------------------------------------------------

LOGS_DIR = Path(__file__).resolve().parent.parent / "logs"
LOGS_DIR.mkdir(exist_ok=True)

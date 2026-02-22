from contextlib import asynccontextmanager
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.azure_client import build_client, set_client
from app.routes import compare, batch, health, logs

load_dotenv(override=True)

STATIC_DIR = Path(__file__).resolve().parent.parent / "static"


# ---------------------------------------------------------------------------
# Lifespan – create / tear down the shared OpenAI client
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    client = build_client()
    set_client(client)
    yield
    await client.close()
    set_client(None)


# ---------------------------------------------------------------------------
# FastAPI application
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Signature Matching Agent API",
    description="Upload two signature images to compare their visual similarity.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Register route modules ---
app.include_router(compare.router)
app.include_router(batch.router)
app.include_router(health.router)
app.include_router(logs.router)

# --- Serve frontend ---
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/")
async def root():
    return FileResponse(STATIC_DIR / "index.html")

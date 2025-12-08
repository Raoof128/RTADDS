"""Entry point for FastAPI application."""
from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from backend.api import audio, report, stream
from backend.utils.logger import get_logger

logger = get_logger(__name__)

app = FastAPI(title="Real-Time Audio Deepfake Detection System", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(audio.router)
app.include_router(stream.router)
app.include_router(report.router)

app.mount("/frontend", StaticFiles(directory="frontend"), name="frontend")


@app.get("/health")
async def health() -> dict:
    """Health check endpoint."""
    return {"status": "ok"}


@app.on_event("startup")
async def startup_event() -> None:
    logger.info("Starting Real-Time Audio Deepfake Detection System")

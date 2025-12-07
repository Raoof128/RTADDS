"""Simulated WebRTC streaming endpoints."""
from __future__ import annotations

from typing import Any, Dict

import numpy as np
from fastapi import APIRouter, HTTPException

from backend.api.schemas import (
    Metadata,
    StreamAnalysisResponse,
    StreamChunkRequest,
    StreamStartRequest,
    StreamStopRequest,
)
from backend.engines.classifier import predict_deepfake
from backend.engines.feature_extractor import FeatureExtractor
from backend.engines.fusion import fuse_signals
from backend.engines.jitter_detector import compute_jitter_metrics
from backend.engines.vocoder_detector import detect_vocoder_artifacts
from backend.utils.audio_utils import waveform_metadata
from backend.utils.logger import get_logger

router = APIRouter(prefix="/stream", tags=["stream"])
logger = get_logger(__name__)
feature_extractor = FeatureExtractor()

stream_state: Dict[str, Dict[str, Any]] = {}


@router.post("/start")
async def start_stream(request: StreamStartRequest) -> Dict[str, str]:
    """Initialize a simulated streaming session."""
    session_id = request.session_id
    if session_id in stream_state:
        raise HTTPException(status_code=400, detail="Session already exists")

    stream_state[session_id] = {"waveform": np.array([], dtype=np.float32)}
    logger.info("Started stream %s", session_id)
    return {"status": "started", "session_id": session_id}


@router.post("/chunk")
async def ingest_chunk(request: StreamChunkRequest) -> Dict[str, Any]:
    """Ingest a chunk of audio samples (already at 16kHz)."""

    session_id = request.session_id
    if session_id not in stream_state:
        raise HTTPException(status_code=404, detail="Unknown session")

    chunk = np.array(request.samples, dtype=np.float32)
    if chunk.size == 0:
        raise HTTPException(status_code=400, detail="Chunk contained no samples")

    current_waveform = stream_state[session_id]["waveform"]
    updated = np.concatenate([current_waveform, chunk])
    if updated.size > 128_000:  # cap at ~8 seconds buffered to avoid abuse
        logger.warning("Stream %s exceeded maximum buffered duration", session_id)
        raise HTTPException(status_code=413, detail="Stream too long; please restart")

    stream_state[session_id]["waveform"] = updated
    logger.debug("Stream %s received chunk of %d samples", session_id, len(chunk))
    return {"received": len(chunk)}


@router.post("/stop", response_model=StreamAnalysisResponse)
async def stop_stream(request: StreamStopRequest) -> StreamAnalysisResponse:
    """Finalize streaming session and run analysis."""
    session_id = request.session_id
    if session_id not in stream_state:
        raise HTTPException(status_code=404, detail="Unknown session")
    waveform = stream_state.pop(session_id)["waveform"]
    if waveform.size == 0:
        raise HTTPException(status_code=400, detail="No audio received")

    metadata = waveform_metadata(waveform, sample_rate=16_000)
    try:
        feature_vector, maps = feature_extractor.extract(waveform)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    vocoder_score, _ = detect_vocoder_artifacts(maps["mel_spectrogram"])
    jitter_score, _ = compute_jitter_metrics(waveform, 16_000)
    classifier_score, label = predict_deepfake(feature_vector)
    energy_flatness = float(maps["anomaly_map"].mean())

    final_score, verdict, details = fuse_signals(
        vocoder_score=vocoder_score,
        jitter_score=jitter_score,
        classifier_score=classifier_score,
        snr=metadata["snr"],
        energy_flatness=energy_flatness,
    )
    return StreamAnalysisResponse(
        metadata=Metadata(**metadata),
        label=label,
        verdict=verdict,
        final_score=final_score,
        scores=details,
    )

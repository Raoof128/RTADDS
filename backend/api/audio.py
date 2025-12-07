"""Audio ingestion and analysis endpoints."""
from __future__ import annotations

from fastapi import APIRouter, File, HTTPException, UploadFile

from backend.api.schemas import AnalysisResponse, AnalysisVisuals, Metadata
from backend.engines.classifier import predict_deepfake
from backend.engines.feature_extractor import FeatureExtractor
from backend.engines.fusion import fuse_signals
from backend.engines.jitter_detector import compute_jitter_metrics
from backend.engines.vocoder_detector import detect_vocoder_artifacts
from backend.utils import visuals
from backend.utils.audio_utils import AudioProcessingError, load_audio, waveform_metadata
from backend.utils.logger import get_logger

router = APIRouter(prefix="/analyze_audio", tags=["audio"])
logger = get_logger(__name__)
feature_extractor = FeatureExtractor()


@router.post("", response_model=AnalysisResponse)
async def analyze_audio(file: UploadFile = File(...)) -> AnalysisResponse:
    """Analyze uploaded audio file."""
    if file.content_type not in {"audio/wav", "audio/x-wav", "audio/mpeg", "application/octet-stream"}:
        raise HTTPException(status_code=400, detail="Unsupported file type")

    payload = await file.read()
    try:
        waveform, sample_rate = load_audio(payload)
    except AudioProcessingError as exc:  # pragma: no cover - exercised indirectly
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    metadata = waveform_metadata(waveform, sample_rate)

    try:
        feature_vector, maps = feature_extractor.extract(waveform)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    vocoder_score, artifact_map = detect_vocoder_artifacts(maps["mel_spectrogram"])
    jitter_score, jitter_metrics = compute_jitter_metrics(waveform, sample_rate)
    classifier_score, label = predict_deepfake(feature_vector)
    energy_flatness = float(maps["anomaly_map"].mean())

    final_score, verdict, details = fuse_signals(
        vocoder_score=vocoder_score,
        jitter_score=jitter_score,
        classifier_score=classifier_score,
        snr=metadata["snr"],
        energy_flatness=energy_flatness,
    )

    logger.info("Analyzed audio: verdict %s", verdict)
    return AnalysisResponse(
        metadata=Metadata(**metadata),
        vocoder_score=vocoder_score,
        jitter_score=jitter_score,
        jitter_metrics=jitter_metrics,
        classifier_score=classifier_score,
        label=label,
        final_score=final_score,
        verdict=verdict,
        visuals=AnalysisVisuals(
            waveform=visuals.waveform_plot(waveform, sample_rate),
            spectrogram=visuals.spectrogram_plot(waveform, sample_rate),
            jitter=visuals.jitter_plot(maps["jitter_series"]),
            artifact_map=visuals.artifact_heatmap(artifact_map),
        ),
        details=details,
    )

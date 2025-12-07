"""Pydantic models for API payloads and responses."""
from __future__ import annotations

from pydantic import BaseModel, Field, field_validator


class Metadata(BaseModel):
    """Metadata describing analyzed audio."""

    duration: float = Field(..., ge=0)
    snr: float
    rms: float = Field(..., ge=0)
    sample_rate: int = Field(..., gt=0)


class AnalysisVisuals(BaseModel):
    """Base64-encoded visualization assets."""

    waveform: str
    spectrogram: str
    jitter: str
    artifact_map: str


class AnalysisResponse(BaseModel):
    """Response payload for audio analysis."""

    metadata: Metadata
    vocoder_score: float = Field(..., ge=0, le=100)
    jitter_score: float = Field(..., ge=0, le=100)
    jitter_metrics: dict[str, float]
    classifier_score: float = Field(..., ge=0, le=100)
    label: str
    final_score: float = Field(..., ge=0, le=100)
    verdict: str
    visuals: AnalysisVisuals
    details: dict[str, float]


class StreamStartRequest(BaseModel):
    """Model for beginning a streaming session."""

    session_id: str = Field(..., min_length=1, max_length=64)


class StreamChunkRequest(BaseModel):
    """Model for ingesting a chunk of streaming audio."""

    session_id: str = Field(..., min_length=1, max_length=64)
    samples: list[float] = Field(..., min_length=1, max_length=64_000)

    @field_validator("samples")
    @classmethod
    def validate_samples(cls, samples: list[float]) -> list[float]:
        if any(not isinstance(x, (int, float)) for x in samples):
            raise ValueError("Samples must be numeric.")
        if any(not (float("-inf") < float(x) < float("inf")) for x in samples):
            raise ValueError("Samples must be finite numbers.")
        return samples


class StreamStopRequest(BaseModel):
    """Model for closing a streaming session."""

    session_id: str = Field(..., min_length=1, max_length=64)


class ReportRequest(BaseModel):
    """Model for generating a PDF report."""

    summary: dict[str, str]
    scores: dict[str, float]


class StreamAnalysisResponse(BaseModel):
    """Response payload for streaming analysis."""

    metadata: Metadata
    label: str
    verdict: str
    final_score: float
    scores: dict[str, float]

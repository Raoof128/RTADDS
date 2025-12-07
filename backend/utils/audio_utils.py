"""Audio utilities for loading, resampling, and segmentation."""
from __future__ import annotations

import io
from typing import Dict, List, Tuple

import librosa
import numpy as np
import soundfile as sf

from .logger import get_logger

logger = get_logger(__name__)

TARGET_SR = 16_000
CHUNK_DURATION = 0.4  # 400ms


class AudioProcessingError(RuntimeError):
    """Raised when audio bytes cannot be decoded or are invalid."""


def load_audio(file_bytes: bytes, sr: int = TARGET_SR) -> Tuple[np.ndarray, int]:
    """Load audio from bytes and resample to target sample rate.

    Args:
        file_bytes: Raw audio bytes.
        sr: Target sample rate.
    Returns:
        Tuple of waveform and sample rate.
    Raises:
        AudioProcessingError: If the payload cannot be parsed into audio.
    """
    if not file_bytes:
        raise AudioProcessingError("No audio content provided.")

    try:
        with io.BytesIO(file_bytes) as buffer:
            waveform, sample_rate = sf.read(buffer)
    except Exception as exc:  # noqa: BLE001 - need to catch decoding issues
        raise AudioProcessingError("Failed to decode audio payload.") from exc

    if waveform.size == 0:
        raise AudioProcessingError("Audio payload contained no samples.")

    if waveform.ndim > 1:
        waveform = np.mean(waveform, axis=1)
    if sample_rate != sr:
        waveform = librosa.resample(waveform, orig_sr=sample_rate, target_sr=sr)
        sample_rate = sr
    waveform = librosa.util.normalize(waveform)
    logger.info("Loaded audio duration %.2fs at %d Hz", len(waveform) / sample_rate, sample_rate)
    return waveform.astype(np.float32), sample_rate


def segment_audio(waveform: np.ndarray, sample_rate: int) -> List[np.ndarray]:
    """Segment audio into fixed-duration chunks.

    Args:
        waveform: 1-D numpy waveform array.
        sample_rate: Sample rate.
    Returns:
        List of waveform chunks.
    """
    chunk_size = int(CHUNK_DURATION * sample_rate)
    segments = []
    for start in range(0, len(waveform), chunk_size):
        end = start + chunk_size
        segments.append(waveform[start:end])
    logger.debug("Segmented audio into %d chunks", len(segments))
    return segments


def compute_snr(waveform: np.ndarray) -> float:
    """Compute a naive signal-to-noise ratio for diagnostics."""
    power_signal = np.mean(np.square(waveform))
    noise = waveform - librosa.effects.harmonic(waveform)
    power_noise = np.mean(np.square(noise)) + 1e-8
    snr = 10 * np.log10(power_signal / power_noise)
    return float(np.clip(snr, -20, 80))


def waveform_metadata(waveform: np.ndarray, sample_rate: int) -> Dict[str, float]:
    """Compute basic metadata for the waveform."""
    duration = len(waveform) / sample_rate
    snr = compute_snr(waveform)
    rms = float(np.sqrt(np.mean(np.square(waveform))))
    metadata = {"duration": duration, "snr": snr, "rms": rms, "sample_rate": sample_rate}
    logger.debug("Waveform metadata: %s", metadata)
    return metadata

"""Synthetic vocoder artifact detection logic."""
from __future__ import annotations

import numpy as np

from backend.utils.logger import get_logger

logger = get_logger(__name__)


def detect_vocoder_artifacts(mel_spectrogram: np.ndarray) -> tuple[float, np.ndarray]:
    """Detect vocoder-like artifacts using heuristic cues.

    Args:
        mel_spectrogram: Mel spectrogram matrix (dB).
    Returns:
        Tuple of artifact score (0-100) and heatmap.
    """
    # Check for over-smoothed harmonics in 3-7 kHz region (approx upper mel bins)
    high_bins = mel_spectrogram[30:60]
    temporal_flatness = np.mean(np.std(high_bins, axis=0))
    spectral_flatness = np.mean(np.std(high_bins, axis=1))

    stacking_score = float(np.clip(1 - (spectral_flatness / 20), 0, 1))
    coherence_score = float(np.clip(1 - (temporal_flatness / 15), 0, 1))

    artifact_map = np.tanh(np.abs(high_bins))
    combined = (stacking_score + coherence_score) / 2
    score = float(np.clip(combined * 100, 0, 100))
    logger.debug("Vocoder artifact score: %.2f", score)
    return score, artifact_map

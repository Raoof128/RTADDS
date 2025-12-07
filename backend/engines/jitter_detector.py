"""Micro-jitter detection approximations."""
from __future__ import annotations

from typing import Dict, List, Tuple

import librosa
import numpy as np

from backend.utils.logger import get_logger

logger = get_logger(__name__)


def compute_jitter_metrics(waveform: np.ndarray, sample_rate: int) -> Tuple[float, Dict[str, float]]:
    """Compute jitter and shimmer proxies.

    Args:
        waveform: Audio waveform.
        sample_rate: Sample rate.
    Returns:
        Jitter score (higher is more human) and metric details.
    """
    if waveform.size < sample_rate * 0.05:
        return 0.0, {"jitter_local": 0.0, "jitter_relative": 0.0, "shimmer": 0.0}

    f0 = librosa.yin(waveform, fmin=60, fmax=400, sr=sample_rate)
    f0 = f0[np.isfinite(f0)]
    if len(f0) == 0:
        return 0.0, {"jitter_local": 0.0, "jitter_relative": 0.0, "shimmer": 0.0}

    diffs = np.abs(np.diff(f0))
    jitter_local = float(np.mean(diffs) / (np.mean(f0) + 1e-6))
    jitter_relative = float(np.std(f0) / (np.mean(f0) + 1e-6))
    shimmer = float(np.std(librosa.feature.rms(y=waveform)))

    jitter_score = float(np.clip((0.5 - jitter_local) * 120, 0, 100))
    logger.debug("Jitter metrics computed: jitter_score=%.2f", jitter_score)
    return jitter_score, {
        "jitter_local": jitter_local,
        "jitter_relative": jitter_relative,
        "shimmer": shimmer,
    }

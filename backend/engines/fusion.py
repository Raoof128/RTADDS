"""Fusion engine combining multiple detection signals."""
from __future__ import annotations

import numpy as np

from backend.utils.logger import get_logger

logger = get_logger(__name__)


VERDICT_THRESHOLDS = {
    "SAFE": 30,
    "SUSPICIOUS": 60,
    "MALICIOUS": 85,
}


def fuse_signals(
    vocoder_score: float,
    jitter_score: float,
    classifier_score: float,
    snr: float,
    energy_flatness: float,
) -> tuple[float, str, dict[str, float]]:
    """Combine scores into a final risk assessment."""
    weights = {
        "vocoder": 0.25,
        "jitter": 0.2,
        "classifier": 0.35,
        "snr": 0.1,
        "flatness": 0.1,
    }
    normalized_snr = np.clip(1 - (snr / 50), 0, 1) * 100
    normalized_flatness = np.clip(energy_flatness * 100, 0, 100)

    final_score = (
        vocoder_score * weights["vocoder"]
        + (100 - jitter_score) * weights["jitter"]
        + classifier_score * weights["classifier"]
        + normalized_snr * weights["snr"]
        + normalized_flatness * weights["flatness"]
    )
    verdict = "SAFE"
    if final_score >= VERDICT_THRESHOLDS["MALICIOUS"]:
        verdict = "MALICIOUS"
    elif final_score >= VERDICT_THRESHOLDS["SUSPICIOUS"]:
        verdict = "SUSPICIOUS"
    logger.info("Final risk score %.2f verdict %s", final_score, verdict)
    details = {
        "vocoder_score": vocoder_score,
        "jitter_score": jitter_score,
        "classifier_score": classifier_score,
        "snr_component": normalized_snr,
        "flatness_component": normalized_flatness,
        "final_score": final_score,
    }
    return float(final_score), verdict, details

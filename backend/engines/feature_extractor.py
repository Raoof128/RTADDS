"""Feature extraction engine for audio analysis."""
from __future__ import annotations

from typing import Dict, Tuple

import librosa
import numpy as np

from backend.utils.logger import get_logger

logger = get_logger(__name__)


class FeatureExtractor:
    """Compute spectral and temporal features relevant to deepfake detection."""

    def __init__(self, sample_rate: int = 16_000) -> None:
        self.sample_rate = sample_rate

    def extract(self, waveform: np.ndarray) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        """Extract feature vector and auxiliary matrices.

        Args:
            waveform: Audio waveform.
        Returns:
            Tuple of flat feature vector and maps for visualization.
        Raises:
            ValueError: If the waveform is too short to analyze.
        """
        if waveform.size < int(self.sample_rate * 0.1):
            raise ValueError("Waveform too short for analysis; provide at least 100ms of audio.")

        padded = librosa.util.fix_length(waveform, size=max(len(waveform), self.sample_rate // 2))
        mel = librosa.feature.melspectrogram(y=padded, sr=self.sample_rate, n_mels=64)
        mel_db = librosa.power_to_db(mel, ref=np.max)
        mfcc = librosa.feature.mfcc(S=mel_db, n_mfcc=20)
        chroma = librosa.feature.chroma_cqt(y=padded, sr=self.sample_rate)
        centroid = librosa.feature.spectral_centroid(y=padded, sr=self.sample_rate)
        zcr = librosa.feature.zero_crossing_rate(y=padded)
        harmonic = librosa.effects.harmonic(padded)
        percussive = padded - harmonic
        hnr = (np.mean(np.abs(harmonic)) + 1e-6) / (np.mean(np.abs(percussive)) + 1e-6)

        jitter_proxy = np.std(librosa.yin(padded, fmin=50, fmax=400, sr=self.sample_rate))
        shimmer_proxy = float(np.std(librosa.feature.rms(y=padded)))

        feature_vector = np.concatenate(
            [
                mel_db.mean(axis=1),
                mel_db.std(axis=1),
                mfcc.mean(axis=1),
                mfcc.std(axis=1),
                chroma.mean(axis=1),
                centroid.flatten(),
                zcr.mean(axis=1),
                np.array([hnr, jitter_proxy, shimmer_proxy]),
            ]
        )

        anomaly_map = self._compute_anomaly_map(mel_db)
        logger.debug("Extracted feature vector length %d", len(feature_vector))
        return feature_vector.astype(np.float32), {
            "mel_spectrogram": mel_db,
            "anomaly_map": anomaly_map,
            "jitter_series": chroma.mean(axis=0),
        }

    def _compute_anomaly_map(self, mel_db: np.ndarray) -> np.ndarray:
        """Synthetic anomaly scoring for illustrative purposes."""
        high_band = mel_db[20:40]
        flatness = np.std(high_band, axis=0)
        artifact_map = 1 - (flatness / (np.max(flatness) + 1e-6))
        return artifact_map

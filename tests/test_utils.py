"""Unit tests for utility helpers."""
from __future__ import annotations

import numpy as np
import pytest

from backend.utils.audio_utils import AudioProcessingError, load_audio, segment_audio


def test_load_audio_rejects_empty_bytes() -> None:
    with pytest.raises(AudioProcessingError):
        load_audio(b"")


def test_segment_audio_splits_chunks() -> None:
    waveform = np.zeros(16_000, dtype=np.float32)
    segments = segment_audio(waveform, 16_000)
    assert len(segments) == 3  # 0.4s windows over 1 second audio

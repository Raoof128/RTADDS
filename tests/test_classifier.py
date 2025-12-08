"""Classifier behaviors and fallbacks."""
from __future__ import annotations

from typing import Any

import numpy as np

from backend.engines import classifier


def test_predict_deepfake_fallback(monkeypatch: Any) -> None:
    """Ensure heuristic path works when TensorFlow model cannot be loaded."""

    def _raise(_: int) -> None:
        raise RuntimeError("no tf")

    monkeypatch.setattr(classifier, "_get_model", _raise)
    score, label = classifier.predict_deepfake(np.ones(32, dtype=np.float32))
    assert 0 <= score <= 100
    assert label in {"REAL", "FAKE", "SUSPECT"}

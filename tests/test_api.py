"""Basic API tests using synthetic audio."""
from __future__ import annotations

import io
from typing import Any

import numpy as np
import soundfile as sf
from fastapi.testclient import TestClient

# Silence librosa warnings for synthetic short signals during tests.
import warnings

warnings.filterwarnings("ignore", category=UserWarning, module=r"librosa.*")

from backend.main import app

client = TestClient(app)


def _stub_classifier(_: np.ndarray) -> tuple[float, str]:
    """Deterministic classifier stub for tests."""
    return 12.5, "REAL"


def generate_sine_wave(duration: float = 1.0, sr: int = 16_000) -> bytes:
    """Generate a simple sine wave for testing."""
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    wave = 0.5 * np.sin(2 * np.pi * 220 * t)
    buffer = io.BytesIO()
    sf.write(buffer, wave, sr, format="WAV")
    buffer.seek(0)
    return buffer.read()


def test_health() -> None:
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_analyze_audio(monkeypatch: Any) -> None:
    monkeypatch.setattr("backend.api.audio.predict_deepfake", _stub_classifier)
    audio_bytes = generate_sine_wave(0.5)
    files = {"file": ("test.wav", audio_bytes, "audio/wav")}
    response = client.post("/analyze_audio", files=files)
    assert response.status_code == 200
    payload = response.json()
    assert "verdict" in payload
    assert "final_score" in payload
    assert payload["classifier_score"] == 12.5
    assert payload["label"] == "REAL"
    assert "visuals" in payload

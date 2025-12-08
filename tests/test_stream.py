"""Streaming endpoint coverage."""
from __future__ import annotations

import warnings
from typing import Any

import numpy as np
from fastapi.testclient import TestClient

from backend.main import app

# Silence librosa warnings for synthetic short signals during tests.
warnings.filterwarnings("ignore", category=UserWarning, module=r"librosa.*")

client = TestClient(app)


def _stub_classifier(_: np.ndarray) -> tuple[float, str]:
    return 8.0, "REAL"


def test_stream_flow(monkeypatch: Any) -> None:
    monkeypatch.setattr("backend.api.stream.predict_deepfake", _stub_classifier)

    start = client.post("/stream/start", json={"session_id": "demo"})
    assert start.status_code == 200

    chunk = client.post(
        "/stream/chunk",
        json={"session_id": "demo", "samples": [0.01] * 1600},
    )
    assert chunk.status_code == 200
    assert chunk.json()["received"] == 1600

    stop = client.post("/stream/stop", json={"session_id": "demo"})
    assert stop.status_code == 200
    body = stop.json()
    assert body["label"] == "REAL"
    assert body["final_score"] >= 0
    assert body["scores"]


def test_stream_unknown_session(monkeypatch: Any) -> None:
    monkeypatch.setattr("backend.api.stream.predict_deepfake", _stub_classifier)
    stop = client.post("/stream/stop", json={"session_id": "missing"})
    assert stop.status_code == 404
    assert stop.json()["detail"] == "Unknown session"


def test_stream_chunk_too_long(monkeypatch: Any) -> None:
    monkeypatch.setattr("backend.api.stream.predict_deepfake", _stub_classifier)
    client.post("/stream/start", json={"session_id": "long"})
    payload = {"session_id": "long", "samples": [0.0] * 64_000}
    first = client.post("/stream/chunk", json=payload)
    assert first.status_code == 200

    second = client.post("/stream/chunk", json=payload)
    assert second.status_code == 200

    overflow = client.post("/stream/chunk", json={"session_id": "long", "samples": [0.0] * 2_000})
    assert overflow.status_code == 413
    assert overflow.json()["detail"] == "Stream too long; please restart"

# API Reference

All endpoints are served from the FastAPI application (`backend/main.py`). JSON examples omit base64 visualization payloads for brevity.

## Health
- `GET /health` → `{ "status": "ok" }`

## Analyze Audio
- **POST** `/analyze_audio`
- **Request**: multipart/form-data with field `file` (WAV/MP3). Audio is resampled to 16 kHz, normalized, segmented, and features extracted.
- **Response** (`AnalysisResponse`):
  ```json
  {
    "metadata": {"duration": 1.0, "snr": 23.1, "rms": 0.12, "sample_rate": 16000},
    "vocoder_score": 18.2,
    "jitter_score": 64.4,
    "jitter_metrics": {"jitter_local": 0.03, "jitter_relative": 0.04, "shimmer": 0.01},
    "classifier_score": 12.5,
    "label": "REAL",
    "final_score": 28.4,
    "verdict": "SAFE",
    "visuals": {"waveform": "<b64>", "spectrogram": "<b64>", "jitter": "<b64>", "artifact_map": "<b64>"},
    "details": {"vocoder_score": 18.2, "jitter_score": 64.4, "classifier_score": 12.5, "snr_component": 12.3, "flatness_component": 4.2, "final_score": 28.4}
  }
  ```
- **Errors**: 400 for unsupported type, unreadable audio, or too-short samples.

## Streaming (Simulated WebRTC)
- **POST** `/stream/start`
  - Body: `{ "session_id": "demo" }`
- **POST** `/stream/chunk`
  - Body: `{ "session_id": "demo", "samples": [0.01, 0.02, ...] }`
  - Samples must already be 16 kHz mono floats.
- **POST** `/stream/stop`
  - Body: `{ "session_id": "demo" }`
  - **Response** (`StreamAnalysisResponse`):
    ```json
    {
      "metadata": {"duration": 0.4, "snr": 18.7, "rms": 0.05, "sample_rate": 16000},
      "label": "REAL",
      "verdict": "SAFE",
      "final_score": 22.1,
      "scores": {"vocoder_score": 12.0, "jitter_score": 70.0, "classifier_score": 8.0, "snr_component": 15.1, "flatness_component": 3.4, "final_score": 22.1}
    }
    ```
- **Errors**: 400 for empty chunks or too-short audio; 404 for unknown sessions.

## Reporting
- **POST** `/report`
- **Request** (`ReportRequest`):
  ```json
  { "summary": {"verdict": "SAFE", "notes": "synthetic test"}, "scores": {"risk": 21.5} }
  ```
- **Response**: `{ "report_path": "assets/report.pdf" }`
- Errors return `500` with a generic error detail.

## Frontend
Static dashboard available at `/frontend/index.html`, sourcing the same origin for API calls.

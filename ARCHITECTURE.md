# Architecture Overview

RTADDS follows a modular, layered design with clear separation between ingestion, feature extraction, detection heuristics, machine learning, fusion, and presentation layers.

## Layers
1. **API Layer (FastAPI)**
   - Routers: `/analyze_audio`, `/stream/*`, `/report`, `/health`.
   - Pydantic schemas ensure validated inputs and typed responses.
2. **Engines**
   - `FeatureExtractor`: Librosa-based spectral/temporal feature computation and anomaly maps.
   - `VocoderDetector`: Heuristics for over-smoothed harmonics and phase coherence (synthetic, safe).
   - `JitterDetector`: Micro-jitter/shimmer proxies and human-likeness scoring.
   - `Classifier`: TensorFlow MLP trained on synthetic real vs. fake features; cached to disk.
   - `Fusion`: Weighted aggregation of heuristics, classifier output, SNR, and energy flatness into risk/verdict.
3. **Utilities**
   - Audio loading/resampling/metadata, visualization (base64 PNGs), logging, and PDF export.
4. **Presentation**
   - Static dashboard rendering plots, scores, and guidance.

## Data Flow
1. Audio bytes or stream chunks enter the API.
2. Audio is resampled to 16 kHz, normalized, and optionally segmented.
3. Feature extraction produces mel spectrograms, MFCCs, chroma, centroid, ZCR, HNR, jitter/shimmer, and anomaly maps.
4. Vocoder and jitter detectors compute heuristic scores; classifier outputs deepfake probability.
5. Fusion combines signals into a final risk score and verdict with an explanation bundle.
6. Visuals and optional PDF reports are returned to the caller or rendered in the dashboard.

## Security & Privacy
- Only processes user-supplied or synthetic audio; no outbound calls or scraping.
- Input validation on all endpoints; defensive checks for empty/invalid audio.
- Logging via rotating file handlers with minimal PII (timestamps, verdicts only).
- TensorFlow model trained on synthetic data; no biometric or production voice content is stored.

## Extensibility
- Engines are modular: swap `Classifier` with a different backend (e.g., PyTorch) or extend `VocoderDetector` heuristics without touching API routes.
- Visualization utilities are decoupled from response schemas, enabling reuse in offline batch workflows.
- CI pipeline enforces linting and tests; add further checks (mypy, security scanners) via `.github/workflows/ci.yml`.

# Security Policy

## Reporting Vulnerabilities
- Please report security issues privately to the maintainers via email (security@localhost placeholder). Do not open public issues for vulnerabilities.
- Include a minimal reproduction, impact assessment, and any logs. We will acknowledge within 3 business days.

## Scope
- FastAPI endpoints (`/analyze_audio`, `/stream/*`, `/report`, `/health`).
- Audio ingestion, processing pipelines, classifier caching, and artifact generation (visuals, PDF).
- Static assets when rendered in browsers.

## Guidelines
- Never upload real biometric or live-call data; only synthetic or user-provided audio is permitted.
- Keep dependencies patched using `requirements*.txt`; run CI before releases.
- Validate inputs (length, type, content type) and prefer deterministic, explainable behavior for governance.
- Logs should avoid sensitive payloads; rotate via `backend/utils/logger.py`.

## Governance Alignment
- **NIST AI RMF**: Transparency (docs, API reference), monitoring (logs/CI), and human oversight (dashboard guidance).
- **ISO/IEC 42001**: Local processing, clear roles/responsibilities, and risk controls for synthetic data handling.
- **Australian Scamwatch**: Educational framing, user warnings, and actionable “do not trust” guidance for suspicious verdicts.

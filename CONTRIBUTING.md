# Contributing Guide

Thank you for considering a contribution to the Real-Time Audio Deepfake Detection System (RTADDS). Contributions that improve safety, robustness, documentation, and developer experience are highly valued.

## Development Workflow
1. Fork the repository and create a feature branch from `main` or `work`.
2. Install dependencies (or run `make dev`):
   ```bash
   pip install -r requirements.txt -r requirements-dev.txt
   ```
3. Run quality gates locally before opening a PR:
   ```bash
   make lint
   make test
   ```
4. Ensure code includes type hints, docstrings, and logging for I/O boundaries.
5. Open a pull request describing the change, rationale, and any safety or privacy considerations.

## Commit Standards
- Write clear, descriptive commit messages (imperative mood recommended).
- Include tests for new functionality or bug fixes.
- Avoid committing generated assets, secrets, or large binaries.
- Keep diffs focused; update documentation alongside code changes.

## Security & Safety
- Use only synthetic or user-provided audio for testing; never scrape or intercept live calls.
- Do not introduce network calls to third-party services for audio processing.
- Report vulnerabilities privately following the guidance in [SECURITY.md](SECURITY.md).

## Code of Conduct
Participation is governed by the [Code of Conduct](CODE_OF_CONDUCT.md). Thank you for helping keep the community welcoming and safe.

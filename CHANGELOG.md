# Changelog

## 1.1.0 — work-in-progress checkpoint

Partial code published for continued development. Known issues: STATUS.md.

# Changelog

## 1.1.0 — work-in-progress checkpoint

Tokenizer service implementation.

### Added
- Kronos Tokenizer Service (`backend/app/services/tokenizer.py`)
- Tokenizer API endpoints: `POST /api/v1/tokenizer/tokenize`, `POST /api/v1/tokenizer/detokenize`
- Tokenizer service tests (`backend/tests/test_tokenizer.py`)
- Input validation for window size, feature count, data types

## 1.0.0 — work-in-progress checkpoint

Partial code published for continued development. Known issues: STATUS.md.

All notable changes to this project will be documented in this file.

## [1.0.0] - 2026-10-22

### Added
- Project skeleton with FastAPI backend and React frontend
- Health check endpoints (`/health`, `/api/v1/health`)
- Docker Compose with backend, frontend, and Keycloak
- GitHub Actions CI workflow (backend tests, frontend build/test, Docker builds)
- Keycloak realm import with kronos-lab realm and demo user
- Basic frontend UI with health status display
- Typed API client with error handling
- Frontend unit tests with Vitest + Testing Library
- Backend unit tests with pytest + httpx

### Planned
- Prediction engine with configurable sampling
- LoRA fine-tuning workflow
- Validation dashboard with metrics visualization
- OIDC authentication integration

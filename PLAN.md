# Product development plan

Each round completes one feature or repair. Future items are planned, not implemented claims.

## Current round

project-skeleton

## Roadmap

### project-skeleton — Project Skeleton: FastAPI backend + React frontend with Docker

Status: done

Acceptance:
- FastAPI app runs on port 8000 with /health endpoint
- React + Vite + TypeScript app runs on port 5173
- Docker compose starts both services + Keycloak
- Backend tests pass (pytest)
- Frontend builds and tests pass (npm run build, npm test)

Implementation: `backend/app/main.py`, `backend/app/api/routes.py`, `backend/requirements.txt`, `backend/tests/test_health.py`, `frontend/package.json`, `frontend/src/main.tsx`, `frontend/src/App.tsx`, `frontend/src/api/client.ts`, `docker-compose.yaml`, `Dockerfile.backend`, `Dockerfile.frontend`

Tests: `backend/tests/test_health.py`, `frontend/src/App.test.tsx`

### tokenizer-service — Kronos Tokenizer Service: load tokenizer, encode/decode OHLCV

Status: planned

Acceptance:
- POST /api/v1/tokenize accepts OHLCV windows, returns token IDs
- POST /api/v1/detokenize accepts token IDs, returns OHLCV predictions
- Tokenizer loads from local HuggingFace cache or downloads on first use
- Input validation: window size, feature count, data types

Implementation: `backend/app/services/tokenizer.py`, `backend/app/api/tokenizer_routes.py`, `backend/tests/test_tokenizer.py`

Tests: `backend/tests/test_tokenizer.py`

### prediction-engine — Prediction Engine: Kronos model inference with sampling

Status: planned

Acceptance:
- POST /api/v1/predict accepts context window + horizon, returns sampled predictions
- Configurable temperature, top_p, sample_count
- Returns mean/std/quantiles across samples
- Handles variable context lengths (issue #92)

Implementation: `backend/app/services/predictor.py`, `backend/app/api/prediction_routes.py`, `backend/tests/test_predictor.py`

Tests: `backend/tests/test_predictor.py`

### finetuning-workflow — Fine-tuning Workflow: LoRA adapter training on custom data

Status: planned

Acceptance:
- POST /api/v1/finetune starts LoRA fine-tuning job
- GET /api/v1/finetune/{job_id} returns status/progress
- Accepts custom OHLCV dataset (CSV/Parquet)
- Saves adapter weights for later inference
- Supports market-specific fine-tuning (issue #68)

Implementation: `backend/app/services/finetuner.py`, `backend/app/api/finetune_routes.py`, `backend/tests/test_finetuner.py`

Tests: `backend/tests/test_finetuner.py`

### validation-dashboard — Validation Dashboard: backtest results + metrics visualization

Status: planned

Acceptance:
- Frontend page showing MAE, RMSE, MAPE, Correlation metrics
- Predicted vs actual price charts
- Load validation dataset (issue #114)
- Compare multiple model checkpoints

Implementation: `frontend/src/pages/ValidationDashboard.tsx`, `frontend/src/components/MetricsChart.tsx`, `frontend/src/components/PredictionChart.tsx`, `backend/app/api/validation_routes.py`

Tests: `frontend/src/pages/ValidationDashboard.test.tsx`

### auth-oidc — OIDC Authentication with Keycloak

Status: planned

Acceptance:
- Login redirects to Keycloak
- Callback exchanges code for tokens
- Protected API endpoints require valid access token
- Logout clears session

Implementation: `backend/app/core/auth.py`, `backend/app/api/auth_routes.py`, `frontend/src/auth/AuthProvider.tsx`, `frontend/src/components/LoginButton.tsx`

Tests: `backend/tests/test_auth.py`


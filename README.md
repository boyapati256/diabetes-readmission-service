# diabetes-readmission-service

A production-grade ML service that predicts whether a diabetic patient
will be readmitted to hospital within 30 days of discharge.

Built on the UCI Diabetes 130-US Hospitals dataset: https://archive.ics.uci.edu/ml/datasets/diabetes+130-us+hospitals+for+years+1999-2008

---

## Quick Start

Prerequisites: Python 3.11+, Docker (optional). Create/activate a virtualenv recommended.

Install Python dependencies:

```bash
pip install -r requirements.txt
```

Train a model (writes artifacts to `configs.train.yaml` paths):

```bash
python scripts/train.py --config configs/train.yaml
```

Serve the API locally (reads artifact path from `configs/train.yaml`):

```bash
# start with the provided script
python scripts/serve.py
# or directly with uvicorn
uvicorn api.main:app --host 0.0.0.0 --port 8000
```

Open interactive docs at: http://localhost:8000/docs

Run tests:

```bash
pytest -q
```

Docker (build and run):

```bash
# build local image
docker build -t diabetes-readmission-service:latest .

# run container (requires trained artifacts placed in ./models/)
docker run --rm -p 8000:8000 diabetes-readmission-service:latest

# Or use docker-compose (builds and runs the api service)
docker compose up --build

# Run tests in container (optional profile)
docker compose --profile test up --build
```

## Files of interest

- `configs/train.yaml` — training + data paths + model params
- `src/data/loader.py` — data loading and column validation
- `src/data/preprocess.py` — domain preprocessing (age, ICD9 grouping, imputation)
- `src/features/pipeline.py` — sklearn ColumnTransformer used for training & inference
- `src/training/train.py` — training pipeline (LightGBM, MLflow logging)
- `src/training/evaluate.py` — evaluation metrics (ROC-AUC, PR-AUC, Brier)
- `api/main.py` — FastAPI application
- `api/predictor.py` — loads `preprocessor.joblib` + `model.joblib` and serves predictions
- `scripts/train.py`, `scripts/serve.py` — convenience CLIs

## Notes / Troubleshooting

- The codebase uses a persisted preprocessor + model artifact approach. After training, artifacts are saved to the `paths.artifact_dir` configured in `configs/train.yaml` (default `models/v1/`).
- If `/health` returns degraded, ensure `models/v1/preprocessor.joblib` and `models/v1/model.joblib` exist (or re-run training).
- The configuration references `lightgbm` and `mlflow`; ensure those are installable in your environment (`requirements.txt` includes them).

If you'd like, I can:
- Run the test suite here (requires installing deps in this environment).
- Wire any remaining training/inference parity checks or CI config.# Diabetes Readmission Service

Minimal project skeleton for a diabetes readmission prediction service.

Structure:

- `src/` - package with data, features, training, and utils
- `api/` - FastAPI app and predictor
- `configs/` - YAML configs
- `scripts/` - simple CLI wrappers for training and evaluation
- `tests/` - pytest tests

See `configs/train.yaml` for default paths and hyperparameters.

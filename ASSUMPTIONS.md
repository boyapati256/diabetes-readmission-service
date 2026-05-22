# Assumption Log

## Data

- `weight` dropped: 97% missing — imputing would introduce
  more noise than signal.
- `payer_code` dropped: 40% missing, no clear clinical relevance
  to 30-day readmission.
- `medical_specialty` dropped: 50% missing.
- Only the first encounter per patient is kept. Later encounters
  could be influenced by prior readmission outcomes, causing
  data leakage if included.
- Age bucket midpoints used (e.g. [50-60) → 55) to make age
  ordinal-numeric without losing relative ordering.
- ICD-9 codes grouped into 9 clinical categories to handle the
  ~900 unique values and ensure robustness to unseen codes
  at inference time.

## Modelling

- Binary target: only `<30` is positive. `>30` and `NO` are
  both treated as negative — the clinical question is specifically
  about 30-day readmission risk.
- `class_weight='balanced'` used to handle ~11% positive rate
  without aggressive resampling.
- Decision threshold fixed at 0.5 for simplicity. In a real
  clinical deployment this would be tuned against a cost matrix
  where false negatives (missed readmissions) are more costly
  than false positives.

## Infrastructure

- Single uvicorn worker is sufficient for local/demo use and
  meets the p95 < 250ms target for single-record inference.
- No authentication implemented — required before any
  real-world deployment.
# Assumption Log

## Data

- `weight` dropped: 97% missing — imputing would introduce
  more noise than signal.
- `payer_code` dropped: 40% missing, no clear clinical relevance
  to 30-day readmission.
- `medical_specialty` dropped: 50% missing.
- Only the first encounter per patient is kept. Later encounters
  could be influenced by prior readmission outcomes, causing
  data leakage if included.
- Age bucket midpoints used (e.g. [50-60) → 55) to make age
  ordinal-numeric without losing relative ordering.
- ICD-9 codes grouped into 9 clinical categories to handle the
  ~900 unique values and ensure robustness to unseen codes
  at inference time.

## Modelling

- Binary target: only `<30` is positive. `>30` and `NO` are
  both treated as negative — the clinical question is specifically
  about 30-day readmission risk.
- `class_weight='balanced'` used to handle ~11% positive rate
  without aggressive resampling.
- Decision threshold fixed at 0.5 for simplicity. In a real
  clinical deployment this would be tuned against a cost matrix
  where false negatives (missed readmissions) are more costly
  than false positives.

## Infrastructure

- Single uvicorn worker is sufficient for local/demo use and
  meets the p95 < 250ms target for single-record inference.
- No authentication implemented — required before any
  real-world deployment.
# Assumptions

- Data is provided as CSV files with a single target column for training.
- Model artifacts are stored in `models/` and loaded via `joblib`.

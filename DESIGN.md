# Design Rationale

## Repository Structure

Modules are separated by responsibility: `src/data` owns all
data concerns, `src/features` owns transformation, `src/training`
owns the train/eval loop, and `api/` owns the serving layer.
This boundary means you can swap the model without touching the
API, and change the API without touching training logic.

## Artifact Strategy

The preprocessor and model are saved as separate joblib files
under a versioned directory (`models/v1/`). Keeping them separate
lets you inspect, test, and version each independently. A config
snapshot is saved alongside them so any artifact directory is
fully self-describing — you always know exactly what parameters
produced it.

The preprocessor is fitted on training data only, then applied
identically at inference via the same serialized object. This
is the only correct way to guarantee training/inference parity.
Any solution that re-implements transformation logic in the
serving layer will silently diverge over time.

## API Design

FastAPI was chosen for automatic OpenAPI docs, native Pydantic
integration, and async support. Pydantic schemas validate every
field at the API boundary — the model never receives malformed
data. The predictor is loaded once at startup, not per request,
keeping single-record latency well under the 250ms p95 target.

The `/health` endpoint is a first-class concern, not an
afterthought — it enables proper liveness/readiness checks in
any container orchestration environment.

## Trade-offs

- **LightGBM over deep learning**: interpretable, fast on CPU,
  handles mixed tabular data well, no GPU required.
- **OrdinalEncoder over OneHotEncoder**: avoids dimensionality
  explosion from high-cardinality categoricals like diagnosis
  codes. Unknown categories map to -1 rather than crashing.
- **Separate val split over cross-validation**: gives a stable
  early-stopping signal without the compute cost of k-fold,
  appropriate for the 6–8 hour constraint.
- **joblib over pickle**: safer for sklearn objects, handles
  numpy arrays more efficiently.
# Design Rationale

This repository provides a small, testable skeleton for a ML service.

Key decisions:

- Keep data loading and preprocessing separate from modeling code.
- Use small, dependency-first building blocks so tests can run quickly.
- Expose a `FastAPI` app to serve predictions.

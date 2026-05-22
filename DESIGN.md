# Design Rationale

This repository provides a small, testable skeleton for a ML service.

Key decisions:

- Keep data loading and preprocessing separate from modeling code.
- Use small, dependency-first building blocks so tests can run quickly.
- Expose a `FastAPI` app to serve predictions.

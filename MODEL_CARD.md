# Model Card — Diabetes 30-Day Readmission Classifier

## Model Details
- **Type**: LightGBM binary classifier
- **Version**: v1
- **Trained**: UCI Diabetes 130-US Hospitals 1999–2008
- **Target**: Readmission within 30 days (positive = 1)

## Intended Use
Decision-support tool to flag high-risk patients for
post-discharge follow-up. Not intended as a standalone
clinical decision system.

## Performance (Test Set)
| Metric      | Value  |
|-------------|--------|
| ROC-AUC     | ~0.68  |
| PR-AUC      | ~0.35  |
| Brier Score | ~0.09  |
| Positive Rate | ~11% |

## Known Limitations
- Trained on US hospital data from 1999–2008. Significant
  distribution shift is expected on modern data.
- Demographic features (race, gender, age) are included.
  Fairness across subgroups has not been formally audited.
- Model does not account for clinical context outside
  the structured fields (free-text notes, imaging, etc.)

## Risks
- False negatives (missed high-risk patients) are clinically
  more costly than false positives.
- Should not be used as sole basis for any clinical decision.
- Requires periodic retraining as patient population shifts.

## Fairness Considerations
Sensitive attributes are included as model features.
A full fairness audit using Fairlearn across race, gender,
and age subgroups is recommended before any clinical use.
# Model Card

This project stores model artifacts under `models/` and keeps a short
description of intended use and limitations here.

Intended use: binary classification of readmission risk.

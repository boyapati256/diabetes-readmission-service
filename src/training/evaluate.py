import numpy as np
import pandas as pd
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    brier_score_loss,
)
from src.utils.logger import get_logger

logger = get_logger(__name__)


def evaluate_model(model, X, y_true, split_name: str = "eval") -> dict:
    """
    Runs full evaluation suite on a given split.
    Returns a flat metrics dict — all values are MLflow-loggable.

    We report:
    - ROC-AUC       : overall discrimination ability
    - PR-AUC        : better than ROC on imbalanced classes
    - Brier Score   : calibration quality (lower = better)
    - Classification report at 0.5 threshold
    - Confusion matrix

    NOTE: In a real deployment you'd tune the threshold based on
    the cost trade-off between false positives and false negatives.
    For clinical use, missing a readmission (FN) is typically
    more costly than a false alarm (FP).
    """
    y_prob = model.predict_proba(X)[:, 1]
    y_pred = (y_prob >= 0.5).astype(int)

    metrics = {
        f"{split_name}_roc_auc": round(roc_auc_score(y_true, y_prob), 4),
        f"{split_name}_pr_auc": round(average_precision_score(y_true, y_prob), 4),
        f"{split_name}_brier_score": round(brier_score_loss(y_true, y_prob), 4),
        f"{split_name}_positive_rate": round(float(y_true.mean()), 4),
        f"{split_name}_n_samples": int(len(y_true)),
    }

    logger.info(f"--- Evaluation: {split_name} ---")
    for k, v in metrics.items():
        logger.info(f"  {k}: {v}")

    report = classification_report(y_true, y_pred, target_names=[
                                   "No Readmission", "Readmitted"])
    logger.info(f"\nClassification Report ({split_name}):\n{report}")

    cm = confusion_matrix(y_true, y_pred)
    logger.info(f"Confusion Matrix ({split_name}):\n{cm}")

    return metrics


def log_class_distribution(y: pd.Series, split_name: str):
    """Quick helper to log class balance for a given split."""
    counts = y.value_counts()
    rate = y.mean()
    logger.info(
        f"[{split_name}] Class distribution → "
        f"Negative: {counts.get(0, 0):,} | "
        f"Positive: {counts.get(1, 0):,} | "
        f"Positive rate: {rate:.2%}"
    )

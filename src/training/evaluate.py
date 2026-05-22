"""Evaluation helpers for models."""
from sklearn.metrics import precision_score, recall_score, f1_score


def evaluate_model(y_true, y_pred):
    """Return a dict with precision, recall and f1."""
    return {
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
    }


if __name__ == "__main__":
    print("evaluate utilities")

"""Core training logic (minimal example).

Exports `train_model` which trains a simple LogisticRegression
on provided features and target.
"""
from typing import Tuple
import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


def train_model(X: pd.DataFrame, y: pd.Series, model_path: str = None) -> Tuple[LogisticRegression, float]:
    """Train a LogisticRegression model and return (model, accuracy).

    If `model_path` is provided the trained model will be saved using joblib.
    """
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    preds = model.predict(X_val)
    acc = accuracy_score(y_val, preds)
    if model_path:
        joblib.dump(model, model_path)
    return model, acc


if __name__ == "__main__":
    print("Run via `scripts/train.py` with data inputs")

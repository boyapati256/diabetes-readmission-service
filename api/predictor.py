"""Predictor: loads artifacts and runs inference (minimal placeholder)."""
import typing as t
import joblib


class Predictor:
    def __init__(self, model_path: t.Optional[str] = None):
        self.model = None
        if model_path:
            self.load(model_path)

    def load(self, path: str):
        self.model = joblib.load(path)

    def predict(self, features: dict) -> t.Tuple[int, float]:
        if self.model is None:
            raise RuntimeError("Model not loaded")
        import numpy as np
        X = [features[k] for k in sorted(features.keys())]
        probs = self.model.predict_proba([X])[0]
        pred = int(self.model.predict([X])[0])
        return pred, float(max(probs))


predictor = Predictor()

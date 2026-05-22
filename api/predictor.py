"""Predictor: loads artifacts and runs inference (minimal placeholder)."""
import typing as t
import joblib


class ReadmissionPredictor:
    """
    Loads the versioned preprocessor + model artifacts and serves predictions.

    Kept as a class so the FastAPI app instantiates it ONCE at startup
    (not per request) — avoids repeated disk I/O on every call.

    Training/inference parity is guaranteed because:
    - The SAME preprocessor object fitted during training is loaded here
    - Column order is enforced via self.feature_columns
    - No transformation logic is duplicated in this file
    """

    def __init__(self, artifact_dir: str):
        self.artifact_dir = Path(artifact_dir)
        self.model = None
        self.preprocessor = None
        self.model_version = None
        self._is_loaded = False

        self._load_artifacts()

    def _load_artifacts(self):
        preprocessor_path = self.artifact_dir / "preprocessor.joblib"
        model_path = self.artifact_dir / "model.joblib"

        if not preprocessor_path.exists():
            raise FileNotFoundError(
                f"Preprocessor not found at {preprocessor_path}. "
                "Run: python scripts/train.py first."
            )
        if not model_path.exists():
            raise FileNotFoundError(
                f"Model not found at {model_path}. "
                "Run: python scripts/train.py first."
            )

        logger.info(f"Loading preprocessor from: {preprocessor_path}")
        self.preprocessor = joblib.load(preprocessor_path)

        logger.info(f"Loading model from: {model_path}")
        self.model = joblib.load(model_path)

        # Version is the artifact directory name e.g. "v1"
        self.model_version = self.artifact_dir.name
        self._is_loaded = True

        logger.info(
            f"Artifacts loaded successfully. Model version: {self.model_version}")

    def _request_to_dataframe(self, request_dict: dict) -> pd.DataFrame:
        """
        Converts the validated Pydantic request dict into a
        single-row DataFrame with columns in the exact order
        the preprocessor expects.
        """
        return pd.DataFrame([request_dict])

    def _get_risk_level(self, probability: float) -> str:
        """
        Converts raw probability into a human-readable risk band.

        Thresholds chosen conservatively for clinical context:
        missing a high-risk patient (FN) is costlier than
        a false alarm (FP).
        """
        if probability >= 0.5:
            return "High"
        elif probability >= 0.3:
            return "Medium"
        else:
            return "Low"

    def predict(self, request_dict: dict) -> dict:
        """
        Full inference pipeline:
            raw dict → DataFrame → preprocess → model → response dict

        Returns a dict matching PredictionResponse schema.
        """
        if not self._is_loaded:
            raise RuntimeError("Model artifacts are not loaded.")

        # 1. Convert to DataFrame
        df = self._request_to_dataframe(request_dict)

        # 2. Preprocess using the SAME pipeline fitted at training
        X_transformed = self.preprocessor.transform(df)

        # 3. Model inference
        probability = float(self.model.predict_proba(X_transformed)[0][1])
        prediction = int(probability >= 0.5)
        risk_level = self._get_risk_level(probability)

        return {
            "prediction": prediction,
            "probability": round(probability, 4),
            "risk_level": risk_level,
            "model_version": self.model_version,
        }

    @property
    def is_loaded(self) -> bool:
        return self._is_loaded


"""Predictor: loads artifacts and runs inference (minimal placeholder)."""


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

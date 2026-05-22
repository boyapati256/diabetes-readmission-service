import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from sklearn.model_selection import train_test_split
from lightgbm import LGBMClassifier

from src.features.pipeline import build_feature_pipeline, get_feature_names
from src.training.evaluate import evaluate_model, log_class_distribution
from src.utils.logger import get_logger

logger = get_logger(__name__)


def build_model(model_params: dict) -> LGBMClassifier:
    """Instantiates the classifier from config params."""
    logger.info(f"Building LGBMClassifier with params: {model_params}")
    return LGBMClassifier(**model_params)


def split_data(df: pd.DataFrame, target_col: str, test_size: float,
               val_size: float, random_seed: int):
    """
    Splits into train / val / test.
    Test set is held out entirely — never touches training or tuning.
    Val set is used for early stopping and threshold analysis.
    """
    X = df.drop(columns=[target_col])
    y = df[target_col]

    # First split off test set
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_seed, stratify=y
    )

    # Then split val from remaining
    relative_val_size = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp,
        test_size=relative_val_size,
        random_state=random_seed,
        stratify=y_temp,
    )

    logger.info(
        f"Split sizes → "
        f"Train: {len(X_train):,} | "
        f"Val: {len(X_val):,} | "
        f"Test: {len(X_test):,}"
    )

    log_class_distribution(y_train, "train")
    log_class_distribution(y_val,   "val")
    log_class_distribution(y_test,  "test")

    return X_train, X_val, X_test, y_train, y_val, y_test


def save_artifacts(pipeline, model, artifact_dir: str, config: dict):
    """
    Saves preprocessor + model as separate versioned artifacts.

    We save them SEPARATELY (not as one object) so that:
    - Preprocessing can be inspected/tested independently
    - Model can be swapped without reprocessing all data
    - Inference service loads each explicitly
    """
    artifact_path = Path(artifact_dir)
    artifact_path.mkdir(parents=True, exist_ok=True)

    preprocessor_path = artifact_path / "preprocessor.joblib"
    model_path = artifact_path / "model.joblib"
    config_path = artifact_path / "train_config_snapshot.yaml"

    joblib.dump(pipeline, preprocessor_path)
    joblib.dump(model,    model_path)

    # Snapshot the config used — critical for reproducibility
    import yaml
    with open(config_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False)

    logger.info(f"Artifacts saved to: {artifact_path}")
    logger.info(f"  preprocessor : {preprocessor_path}")
    logger.info(f"  model        : {model_path}")
    logger.info(f"  config snap  : {config_path}")

    return str(preprocessor_path), str(model_path)


def run_training(config: dict):
    """
    Master training function. Called by scripts/train.py.

    Flow:
        raw df → split → fit preprocessor on train only →
        transform all splits → fit model → evaluate → save artifacts
    """
    from src.data.loader import load_raw_data
    from src.data.preprocess import run_preprocessing

    # ── MLflow setup ────────────────────────────────────────────
    mlflow.set_tracking_uri(config["mlflow"]["tracking_uri"])
    mlflow.set_experiment(config["mlflow"]["experiment_name"])

    run_name = f"lgbm_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    with mlflow.start_run(run_name=run_name) as run:
        logger.info(f"MLflow run started: {run.info.run_id}")

        # ── Log config params ────────────────────────────────────
        mlflow.log_params(config["model"]["params"])
        mlflow.log_param("test_size",  config["data"]["test_size"])
        mlflow.log_param("val_size",   config["data"]["val_size"])
        mlflow.log_param("model_type", config["model"]["type"])

        # ── Load + preprocess ────────────────────────────────────
        df_raw = load_raw_data(config["paths"]["raw_data"])
        df = run_preprocessing(df_raw, config)

        # Save processed data for reference
        processed_path = Path(config["paths"]["processed_data"])
        processed_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(processed_path, index=False)
        logger.info(f"Processed data saved to: {processed_path}")

        # ── Split ────────────────────────────────────────────────
        X_train, X_val, X_test, y_train, y_val, y_test = split_data(
            df,
            target_col=config["data"]["target_column"],
            test_size=config["data"]["test_size"],
            val_size=config["data"]["val_size"],
            random_seed=config["project"]["random_seed"],
        )

        # ── Fit preprocessor on TRAIN only ──────────────────────
        # CRITICAL: fit only on train to prevent data leakage
        numeric_cols = config["data"]["numeric_columns"]
        categorical_cols = config["data"]["categorical_columns"]

        # age was converted to numeric in preprocessing, move it
        if "age" in categorical_cols:
            categorical_cols = [c for c in categorical_cols if c != "age"]
        if "age" in X_train.columns and "age" not in numeric_cols:
            numeric_cols = numeric_cols + ["age"]

        feature_pipeline = build_feature_pipeline(
            numeric_cols, categorical_cols)

        logger.info("Fitting feature pipeline on training data only...")
        X_train_t = feature_pipeline.fit_transform(X_train)
        X_val_t = feature_pipeline.transform(X_val)
        X_test_t = feature_pipeline.transform(X_test)

        feature_names = get_feature_names(feature_pipeline)
        mlflow.log_param("n_features", len(feature_names))
        logger.info(
            f"Feature count after transformation: {len(feature_names)}")

        # ── Train model ──────────────────────────────────────────
        model = build_model(config["model"]["params"])

        logger.info("Training model...")
        model.fit(
            X_train_t, y_train,
            eval_set=[(X_val_t, y_val)],
        )

        # ── Evaluate all splits ──────────────────────────────────
        train_metrics = evaluate_model(model, X_train_t, y_train, "train")
        val_metrics = evaluate_model(model, X_val_t,   y_val,   "val")
        test_metrics = evaluate_model(model, X_test_t,  y_test,  "test")

        all_metrics = {**train_metrics, **val_metrics, **test_metrics}
        mlflow.log_metrics(all_metrics)

        # ── Save artifacts ───────────────────────────────────────
        preprocessor_path, model_path = save_artifacts(
            feature_pipeline, model,
            artifact_dir=config["paths"]["artifact_dir"],
            config=config,
        )

        # Log artifacts to MLflow too
        mlflow.log_artifact(preprocessor_path)
        mlflow.log_artifact(model_path)
        mlflow.log_artifact(config["paths"]["processed_data"])

        logger.info(f"Training complete. MLflow run ID: {run.info.run_id}")
        logger.info(f"Test ROC-AUC: {test_metrics['test_roc_auc']}")

        return run.info.run_id, all_metrics

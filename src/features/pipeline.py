"""Feature engineering pipeline builder.

This module provides a compact, deterministic ColumnTransformer used by
training and inference to produce the final feature matrix.
"""
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.impute import SimpleImputer
from src.utils.logger import get_logger

logger = get_logger(__name__)


def build_feature_pipeline(numeric_cols: list, categorical_cols: list) -> ColumnTransformer:
    """
    Builds a ColumnTransformer that handles numeric and categorical
    columns separately. The object is serialized and loaded at
    inference time to guarantee parity between training and serving.

    Numeric pipeline:
        - Impute with median (safety net for runtime nulls)
        - StandardScaler

    Categorical pipeline:
        - Impute with 'Missing' string
        - OrdinalEncoder with handle_unknown='use_encoded_value' so
          unseen categories at inference do not crash the service
    """
    logger.info("Building feature pipeline")
    logger.info("  Numeric columns  : %d", len(numeric_cols))
    logger.info("  Categorical cols : %d", len(categorical_cols))

    numeric_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    categorical_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="constant", fill_value="Missing")),
        (
            "encoder",
            OrdinalEncoder(
                handle_unknown="use_encoded_value",
                unknown_value=-1,
                encoded_missing_value=-2,
            ),
        ),
    ])

    feature_pipeline = ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, numeric_cols),
            ("categorical", categorical_pipeline, categorical_cols),
        ],
        remainder="drop",
    )

    return feature_pipeline


def get_feature_names(pipeline: ColumnTransformer) -> list:
    """Return ordered list of feature names after transformation."""
    try:
        return list(pipeline.get_feature_names_out())
    except Exception:
        logger.warning("Could not extract feature names from pipeline.")
        return []


"""Feature engineering pipeline builder.

This module builds a fuller sklearn `Pipeline` that can:

- drop unnecessary columns
- optionally mask or drop sensitive fields
- impute and scale numeric features
- encode categorical features

import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.impute import SimpleImputer
from src.utils.logger import get_logger

logger = get_logger(__name__)


def build_feature_pipeline(numeric_cols: list, categorical_cols: list) -> ColumnTransformer:
    """
    Builds a ColumnTransformer that handles numeric and categorical
    columns separately — this entire object gets serialized and
    loaded at inference time to guarantee parity.

    Numeric pipeline:
        - Impute with median(safety net for any runtime nulls)
        - StandardScaler(helps LightGBM less, but good practice
          and required if you swap to logistic regression later)

    Categorical pipeline:
        - Impute with 'Missing' string
        - OrdinalEncoder with handle_unknown = 'use_encoded_value'
          so unseen categories at inference don't crash the service
    """
    logger.info(f"Building feature pipeline")
    logger.info(f"  Numeric columns  : {len(numeric_cols)}")
    logger.info(f"  Categorical cols : {len(categorical_cols)}")

    numeric_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    categorical_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="constant", fill_value="Missing")),
        (
            "encoder",
            OrdinalEncoder(
                handle_unknown="use_encoded_value",
                unknown_value=-1,         # unseen category at inference → -1
                encoded_missing_value=-2, # explicit NaN handling
            ),
        ),
    ])

    feature_pipeline = ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, numeric_cols),
            ("categorical", categorical_pipeline, categorical_cols),
        ],
        remainder="drop",   # drop any column not explicitly listed
        verbose_feature_names_out=False,
    )

    return feature_pipeline


def get_feature_names(pipeline: ColumnTransformer) -> list:
    """
    Returns ordered list of feature names after transformation.
    Useful for logging, model cards, and feature importance reports.
    """
    try:
        return list(pipeline.get_feature_names_out())
    except Exception:
        logger.warning("Could not extract feature names from pipeline.")
        return []
    - numeric_features: list of numeric column names to process

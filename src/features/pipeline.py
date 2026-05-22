"""Build a minimal sklearn pipeline / ColumnTransformer.

This file provides a tiny, dependency-light pipeline builder used in
training and tests. It avoids heavy preprocessing for the example.
"""
from typing import List
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer


def build_feature_pipeline(numeric_features: List[str]) -> ColumnTransformer:
    """Return a ColumnTransformer for numeric features.

    - Imputes missing values with mean
    - Standard scales numeric columns
    """
    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="mean")),
        ("scaler", StandardScaler()),
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
        ],
        remainder="drop",
    )
    return preprocessor


if __name__ == "__main__":
    print("feature pipeline builder")

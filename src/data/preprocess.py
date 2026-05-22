"""Data preprocessing utilities: cleaning and feature engineering."""
import pandas as pd


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """Minimal preprocessing pipeline.

    - Drops duplicate rows
    - Fills numeric NA with column mean
    - Returns a new DataFrame
    """
    df = df.copy()
    df = df.drop_duplicates()
    num_cols = df.select_dtypes(include=["number"]).columns
    for c in num_cols:
        df[c] = df[c].fillna(df[c].mean())
    return df


if __name__ == "__main__":
    print("preprocess module: call preprocess(df)")

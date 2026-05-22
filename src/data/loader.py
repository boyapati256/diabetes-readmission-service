"""Simple data loader utilities."""
from pathlib import Path
import pandas as pd


def load_raw_data(path: str) -> pd.DataFrame:
    """Load raw CSV data from `path` into a DataFrame.

    This is a minimal placeholder used by tests and scripts.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Data file not found: {path}")
    return pd.read_csv(p)


if __name__ == "__main__":
    print("loader.py module - provide a path to load data")

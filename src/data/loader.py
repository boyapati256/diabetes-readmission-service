import pandas as pd
from pathlib import Path
from src.utils.logger import get_logger

logger = get_logger(__name__)


def load_raw_data(path: str) -> pd.DataFrame:
    """
    Loads the raw UCI diabetes dataset from a local CSV path.

    The dataset must be manually downloaded from:
    https://archive.ics.uci.edu/ml/datasets/diabetes+130-us+hospitals+for+years+1999-2008

    Place the file at: data/raw/diabetic_data.csv
    """
    filepath = Path(path)

    if not filepath.exists():
        raise FileNotFoundError(
            f"Raw data not found at '{filepath}'.\n"
            "Download it from UCI and place it at data/raw/diabetic_data.csv\n"
            "See README.md → Data Access section for instructions."
        )

    logger.info(f"Loading raw data from: {filepath}")
    df = pd.read_csv(filepath, na_values=["?", "Unknown", "None", ""])
    logger.info(f"Loaded {len(df):,} rows × {df.shape[1]} columns")

    return df


def load_processed_data(path: str) -> pd.DataFrame:
    """Loads already-processed data (post preprocessing step)."""
    filepath = Path(path)

    if not filepath.exists():
        raise FileNotFoundError(
            f"Processed data not found at '{filepath}'.\n"
            "Run: python scripts/train.py to generate it first."
        )

    logger.info(f"Loading processed data from: {filepath}")
    df = pd.read_csv(filepath)
    logger.info(f"Loaded {len(df):,} rows × {df.shape[1]} columns")

    return df

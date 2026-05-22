import pandas as pd
import numpy as np
from src.utils.logger import get_logger

logger = get_logger(__name__)


def drop_columns(df: pd.DataFrame, columns: list) -> pd.DataFrame:
    """Drop columns that are uninformative or too sparse."""
    existing = [c for c in columns if c in df.columns]
    dropped = [c for c in columns if c not in df.columns]

    if dropped:
        logger.warning(f"Columns not found (skipping): {dropped}")

    logger.info(f"Dropping columns: {existing}")
    return df.drop(columns=existing)


def remove_duplicate_encounters(df: pd.DataFrame) -> pd.DataFrame:
    """
    Each patient may have multiple encounters.
    We keep only the FIRST encounter per patient to avoid data leakage
    (later encounters can be influenced by readmission outcomes).
    """
    before = len(df)

    # Sort by encounter_id ascending so first encounter comes first
    if "encounter_id" in df.columns:
        df = df.sort_values("encounter_id")

    if "patient_nbr" in df.columns:
        df = df.drop_duplicates(subset=["patient_nbr"], keep="first")

    after = len(df)
    logger.info(
        f"Removed duplicate patient encounters: {before:,} → {after:,} rows "
        f"({before - after:,} removed)"
    )
    return df


def encode_target(df: pd.DataFrame, target_col: str, positive_label: str) -> pd.DataFrame:
    """
    Converts 3-class readmission label to binary.
    '<30'  → 1  (readmitted within 30 days — positive class)
    '>30'  → 0
    'NO'   → 0
    """
    if target_col not in df.columns:
        raise ValueError(
            f"Target column '{target_col}' not found in dataframe.")

    original_dist = df[target_col].value_counts().to_dict()
    logger.info(f"Original target distribution: {original_dist}")

    df[target_col] = (df[target_col] == positive_label).astype(int)

    new_dist = df[target_col].value_counts().to_dict()
    positive_rate = df[target_col].mean()
    logger.info(
        f"Encoded target → binary. "
        f"Distribution: {new_dist} | Positive rate: {positive_rate:.2%}"
    )

    return df


def clean_age(df: pd.DataFrame) -> pd.DataFrame:
    """
    Age is stored as decade buckets e.g. '[0-10)', '[10-20)' etc.
    We convert to the bucket midpoint as an integer for the model.
    Keeps it ordinal and numeric without losing information.
    """
    if "age" not in df.columns:
        return df

    age_map = {
        "[0-10)": 5, "[10-20)": 15, "[20-30)": 25,
        "[30-40)": 35, "[40-50)": 45, "[50-60)": 55,
        "[60-70)": 65, "[70-80)": 75, "[80-90)": 85,
        "[90-100)": 95,
    }
    df["age"] = df["age"].map(age_map)
    logger.info("Converted age buckets to numeric midpoints.")
    return df


def clean_diagnosis_codes(df: pd.DataFrame) -> pd.DataFrame:
    """
    diag_1, diag_2, diag_3 are ICD-9 codes — very high cardinality.
    We group them into broad clinical categories to reduce noise
    and handle unseen codes at inference time.
    """
    diag_cols = ["diag_1", "diag_2", "diag_3"]

    def map_icd9(code) -> str:
        if pd.isna(code):
            return "Unknown"
        code = str(code)
        # Remove any V/E prefix codes → group as Other
        if code.startswith("V") or code.startswith("E"):
            return "External_or_Supplemental"
        try:
            num = float(code)
        except ValueError:
            return "Other"

        if 390 <= num <= 459 or num == 785:
            return "Circulatory"
        elif 460 <= num <= 519 or num == 786:
            return "Respiratory"
        elif 520 <= num <= 579 or num == 787:
            return "Digestive"
        elif num == 250:
            return "Diabetes"
        elif 800 <= num <= 999:
            return "Injury"
        elif 710 <= num <= 739:
            return "Musculoskeletal"
        elif 580 <= num <= 629 or num == 788:
            return "Genitourinary"
        elif 140 <= num <= 239:
            return "Neoplasms"
        else:
            return "Other"

    for col in diag_cols:
        if col in df.columns:
            df[col] = df[col].apply(map_icd9)

    logger.info("Grouped ICD-9 diagnosis codes into clinical categories.")
    return df


def handle_missing_values(df: pd.DataFrame, numeric_cols: list, categorical_cols: list) -> pd.DataFrame:
    """
    Numeric: fill with median (robust to outliers common in clinical data).
    Categorical: fill with the string 'Missing' (preserves missingness as signal).
    """
    for col in numeric_cols:
        if col in df.columns and df[col].isna().any():
            median_val = df[col].median()
            df[col] = df[col].fillna(median_val)
            logger.info(
                f"Filled numeric '{col}' NaNs with median={median_val:.2f}")

    for col in categorical_cols:
        if col in df.columns and df[col].isna().any():
            n_missing = df[col].isna().sum()
            df[col] = df[col].fillna("Missing")
            logger.info(
                f"Filled categorical '{col}' NaNs ({n_missing:,}) with 'Missing'")

    return df


def run_preprocessing(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """
    Master preprocessing function. Runs all steps in order.
    Called by the training script.
    """
    logger.info("=== Starting preprocessing pipeline ===")

    # 1. Remove duplicate patient encounters BEFORE dropping patient_nbr
    df = remove_duplicate_encounters(df)

    # 2. Encode target BEFORE dropping anything else
    df = encode_target(
        df,
        target_col=config["data"]["target_column"],
        positive_label=config["data"]["positive_label"],
    )

    # 3. Drop uninformative columns
    df = drop_columns(df, config["data"]["drop_columns"])

    # 4. Clean specific columns
    df = clean_age(df)
    df = clean_diagnosis_codes(df)

    # 5. Handle missing values
    df = handle_missing_values(
        df,
        numeric_cols=config["data"]["numeric_columns"],
        categorical_cols=config["data"]["categorical_columns"],
    )

    logger.info(f"=== Preprocessing complete. Final shape: {df.shape} ===")
    return df

"""
data_preprocessing.py - Data preprocessing and feature engineering pipeline.

Handles loading raw data, cleaning, feature engineering, encoding
categorical variables, and splitting into train/test sets.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from src.utils import DATA_PATH, save_artifact


# -----------------------------------------------
# Column Configuration
# -----------------------------------------------
COLUMNS_TO_DROP = ["id", "region", "model", "price_per_mile"]
TARGET = "price"

CATEGORICAL_FEATURES = [
    "manufacturer",
    "condition",
    "fuel",
    "title_status",
    "transmission",
    "drive",
    "type",
    "paint_color",
    "state",
]

NUMERICAL_FEATURES = ["year", "odometer", "car_age", "cylinders_num"]


# -----------------------------------------------
# Cylinder extraction
# -----------------------------------------------
CYLINDER_MAP = {
    "3 cylinders": 3,
    "4 cylinders": 4,
    "5 cylinders": 5,
    "6 cylinders": 6,
    "8 cylinders": 8,
    "10 cylinders": 10,
    "12 cylinders": 12,
    "other": 4,
    "Unknown": 4,
}


def _extract_cylinders(df):
    """Convert string cylinder values to a numeric column.

    Maps known cylinder strings to integers via CYLINDER_MAP. Unknown
    values default to 4 (most common). The original 'cylinders' column
    is dropped in place.

    Args:
        df: DataFrame with a 'cylinders' column.

    Returns:
        DataFrame with 'cylinders' replaced by 'cylinders_num'.
    """
    df["cylinders_num"] = df["cylinders"].map(CYLINDER_MAP).fillna(4).astype(int)
    df.drop(columns=["cylinders"], inplace=True)
    return df


# -----------------------------------------------
# Main preprocessing pipeline
# -----------------------------------------------
def load_and_preprocess(filepath=None, test_size=0.2, random_state=42):
    """Full preprocessing pipeline.

    Steps:
        1. Load CSV
        2. Drop irrelevant / data-leakage columns
        3. Extract numeric cylinder count
        4. Label-encode categorical features
        5. Train / test split (80/20)

    Args:
        filepath: Path to the CSV file. Defaults to DATA_PATH.
        test_size: Fraction of data reserved for testing.
        random_state: Seed for reproducibility.

    Returns:
        tuple: (X_train, X_test, y_train, y_test, encoders, feature_names)
            - encoders: dict mapping column name -> fitted LabelEncoder
            - feature_names: ordered list of feature column names
    """
    if filepath is None:
        filepath = DATA_PATH

    print("[INFO] Loading dataset...")
    df = pd.read_csv(filepath)
    print(f"  Loaded {df.shape[0]:,} rows x {df.shape[1]} columns")

    # --- Drop columns ---
    df.drop(columns=COLUMNS_TO_DROP, inplace=True, errors="ignore")
    print(f"  Dropped columns: {COLUMNS_TO_DROP}")

    # --- Extract cylinders ---
    df = _extract_cylinders(df)
    print("  Extracted numeric cylinder count")

    # --- Label Encoding ---
    # 'Unknown' values are treated as a distinct category.
    encoders = {}
    for col in CATEGORICAL_FEATURES:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        encoders[col] = le
    print(f"  Label-encoded {len(CATEGORICAL_FEATURES)} categorical features")

    # --- Feature / target split ---
    feature_cols = NUMERICAL_FEATURES + CATEGORICAL_FEATURES
    X = df[feature_cols]
    y = df[TARGET]

    # --- Train / test split ---
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    print(f"  Train set: {X_train.shape[0]:,} | Test set: {X_test.shape[0]:,}")

    # --- Save encoders & feature list ---
    save_artifact(encoders, "label_encoders.pkl")
    save_artifact(feature_cols, "feature_columns.pkl")

    return X_train, X_test, y_train, y_test, encoders, feature_cols


# -----------------------------------------------
# Standalone entry point
# -----------------------------------------------
if __name__ == "__main__":
    X_train, X_test, y_train, y_test, encoders, features = load_and_preprocess()
    print("\n[DONE] Preprocessing complete.")
    print(f"  Features ({len(features)}): {features}")
    print(f"  X_train shape: {X_train.shape}")
    print(f"  y_train range: ${y_train.min():,} - ${y_train.max():,}")

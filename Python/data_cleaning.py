"""
data_cleaning.py
================
Module 2 of the Banking Transaction Anomaly & Customer Behaviour EDA project.

Responsibility
--------------
Take the raw dataset and:
  1. audit data quality (missing values, duplicates),
  2. apply cleaning (de-duplication, correct data types, parsed dates),
  3. perform feature engineering / data transformation,
  4. persist ``Dataset/cleaned_dataset.csv`` for the analysis stage.

Author : Muzamil banu
Course : AIML - Anudip Foundation
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

from data_loading import RAW_PATH, DATASET_DIR, load_raw_data, download_raw_dataset

CLEANED_PATH = DATASET_DIR / "cleaned_dataset.csv"

# Columns that must be numeric / datetime for analysis.
NUMERIC_COLS = [
    "TransactionAmount", "CustomerAge", "TransactionDuration",
    "LoginAttempts", "AccountBalance",
]
DATE_COLS = ["TransactionDate", "PreviousTransactionDate"]


# --------------------------------------------------------------------------- #
# Quality audit
# --------------------------------------------------------------------------- #
def audit_quality(df: pd.DataFrame) -> dict:
    """Report the data-quality issues that the cleaning step will address."""
    report = {
        "rows": len(df),
        "columns": df.shape[1],
        "missing_values": int(df.isna().sum().sum()),
        "duplicate_rows": int(df.duplicated().sum()),
        "duplicate_transaction_ids": int(df["TransactionID"].duplicated().sum()),
    }
    print("\n" + "=" * 70)
    print("DATA QUALITY AUDIT")
    print("=" * 70)
    for key, value in report.items():
        print(f"{key:28}: {value}")

    print("\nPer-column missing values:")
    missing = df.isna().sum()
    print(missing[missing > 0].to_string() if missing.sum() else "  none")

    negative_amounts = int((pd.to_numeric(df["TransactionAmount"], errors="coerce") < 0).sum())
    print(f"{'negative_amounts':28}: {negative_amounts}")
    report["negative_amounts"] = negative_amounts
    return report


# --------------------------------------------------------------------------- #
# Cleaning
# --------------------------------------------------------------------------- #
def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Return a cleaned copy of the dataframe."""
    df = df.copy()

    # 1. Remove exact duplicate records.
    before = len(df)
    df = df.drop_duplicates()
    print(f"[data_cleaning] Dropped {before - len(df)} duplicate row(s)")

    # 2. Enforce data types.
    for col in NUMERIC_COLS:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    for col in DATE_COLS:
        df[col] = pd.to_datetime(df[col], errors="coerce")

    # 3. Trim whitespace from categorical/text columns (robust across pandas
    #    string dtypes: anything that is neither numeric nor datetime).
    text_cols = [
        c for c in df.columns
        if not pd.api.types.is_numeric_dtype(df[c])
        and not pd.api.types.is_datetime64_any_dtype(df[c])
    ]
    for col in text_cols:
        df[col] = df[col].astype(str).str.strip()
    print(f"[data_cleaning] Normalised text columns: {text_cols}")

    # 4. Sorting gives a deterministic, account-and-time ordered dataset.
    df = df.sort_values(["AccountID", "TransactionDate"]).reset_index(drop=True)
    return df


# --------------------------------------------------------------------------- #
# Feature engineering / transformation
# --------------------------------------------------------------------------- #
def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add the derived features used by the analysis and visualisation stages."""
    df = df.copy()

    # --- calendar features from the transaction timestamp ------------------ #
    df["TransactionYear"] = df["TransactionDate"].dt.year
    df["TransactionMonth"] = df["TransactionDate"].dt.month
    df["TransactionMonthName"] = df["TransactionDate"].dt.strftime("%b")
    df["TransactionDayOfWeek"] = df["TransactionDate"].dt.dayofweek
    df["TransactionDayName"] = df["TransactionDate"].dt.strftime("%a")
    df["TransactionHour"] = df["TransactionDate"].dt.hour
    df["IsWeekend"] = df["TransactionDayOfWeek"].isin([5, 6])

    # --- transaction velocity: gap to the account's previous transaction ---- #
    gap_hours = (
        df.groupby("AccountID")["TransactionDate"]
          .diff()
          .dt.total_seconds() / 3600.0
    )
    df["TimeSinceLastTransactionHours"] = gap_hours.round(3)

    # --- balance relationship ---------------------------------------------- #
    df["AmountToBalanceRatio"] = (
        df["TransactionAmount"] / df["AccountBalance"].replace(0, np.nan)
    ).round(5)

    # --- signed amount: debits reduce the balance, credits increase it ------ #
    df["SignedAmount"] = np.where(
        df["TransactionType"].str.lower() == "debit",
        -df["TransactionAmount"],
        df["TransactionAmount"],
    )

    print(f"[data_cleaning] Engineered {len([
        'TransactionYear', 'TransactionMonth', 'TransactionMonthName',
        'TransactionDayOfWeek', 'TransactionDayName', 'TransactionHour',
        'IsWeekend', 'TimeSinceLastTransactionHours', 'AmountToBalanceRatio',
        'SignedAmount']) } derived features")
    return df


# --------------------------------------------------------------------------- #
# Persistence
# --------------------------------------------------------------------------- #
def save_cleaned(df: pd.DataFrame, path: Path = CLEANED_PATH) -> Path:
    """Write the cleaned dataset to disk."""
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    print(f"[data_cleaning] Saved cleaned dataset -> {path}")
    return path


def build_cleaned_dataset() -> pd.DataFrame:
    """Full cleaning pipeline used by the notebook and the CLI entry point."""
    download_raw_dataset()  # no-op if the raw file already exists
    raw = load_raw_data()
    report = audit_quality(raw)
    cleaned = clean_data(raw)
    cleaned = engineer_features(cleaned)
    save_cleaned(cleaned)

    print("\n--- Cleaned dataset at a glance ---")
    print(f"Shape : {cleaned.shape[0]} rows x {cleaned.shape[1]} columns")
    print(f"Columns added : "
          f"{[c for c in cleaned.columns if c not in raw.columns]}")
    print(f"Remaining missing values : {int(cleaned.isna().sum().sum())}")
    print(f"Negative amounts : {report['negative_amounts']}")
    return cleaned


def main() -> None:
    build_cleaned_dataset()


if __name__ == "__main__":
    sys.exit(main())

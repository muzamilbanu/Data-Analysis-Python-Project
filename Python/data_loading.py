"""
data_loading.py
===============
Module 1 of the Banking Transaction Anomaly & Customer Behaviour EDA project.

Responsibility
--------------
Download (if required) and load the raw banking transactions dataset, then
report its basic structure so every downstream step starts from a known state.

Dataset
-------
Bank Transaction Dataset for Fraud Detection
Source : https://www.kaggle.com/datasets/valakhorasani/bank-transaction-dataset-for-fraud-detection
File   : bank_transactions_data_2.csv  (2,512 rows x 16 columns)

Author : Muzamil banu
Course : AIML - Anudip Foundation
"""

from __future__ import annotations

import sys
import urllib.request
from pathlib import Path

import pandas as pd

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
BASE_DIR = Path(__file__).resolve().parent.parent
DATASET_DIR = BASE_DIR / "Dataset"
RAW_PATH = DATASET_DIR / "raw_dataset.csv"

# Public no-authentication mirror of the original Kaggle file.
DATASET_URL = (
    "https://raw.githubusercontent.com/RifqiMuafa20/"
    "Clustering-and-Classification-Bank-Transaction-Dataset/main/"
    "bank_transactions_data_2.csv"
)


# --------------------------------------------------------------------------- #
# Functions
# --------------------------------------------------------------------------- #
def download_raw_dataset(url: str = DATASET_URL,
                         dest: Path = RAW_PATH,
                         force: bool = False) -> Path:
    """Download the raw CSV to ``dest`` unless it already exists.

    Parameters
    ----------
    url : str
        Location of the raw dataset.
    dest : pathlib.Path
        Where the CSV should be stored.
    force : bool
        Re-download even if ``dest`` already exists.

    Returns
    -------
    pathlib.Path
        Path of the local raw dataset.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)

    if dest.exists() and not force:
        print(f"[data_loading] Raw dataset already present -> {dest}")
        return dest

    print(f"[data_loading] Downloading dataset from:\n    {url}")
    urllib.request.urlretrieve(url, dest)
    print(f"[data_loading] Saved raw dataset -> {dest}")
    return dest


def load_raw_data(path: Path = RAW_PATH) -> pd.DataFrame:
    """Load the raw dataset as a ``pandas.DataFrame``.

    No cleaning is applied here on purpose - this function returns the data
    exactly as stored on disk so that the cleaning step is fully auditable.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"Raw dataset not found at {path}. "
            "Run download_raw_dataset() first."
        )
    df = pd.read_csv(path)
    print(f"[data_loading] Loaded raw dataset: {df.shape[0]} rows x {df.shape[1]} columns")
    return df


def summarise(df: pd.DataFrame) -> None:
    """Print a compact structural summary of the dataframe."""
    print("\n" + "=" * 70)
    print("RAW DATASET SUMMARY")
    print("=" * 70)
    print(f"Shape            : {df.shape[0]} rows x {df.shape[1]} columns")
    print(f"Columns          : {list(df.columns)}")

    print("\n--- Data types & missing values ---")
    info = pd.DataFrame({
        "dtype": df.dtypes.astype(str),
        "non_null": df.notna().sum(),
        "missing": df.isna().sum(),
        "missing_%": (df.isna().mean() * 100).round(2),
        "unique": df.nunique(),
    })
    print(info.to_string())

    print("\n--- First 5 rows ---")
    print(df.head().to_string())

    print("\n--- Numeric description ---")
    print(df.describe().T.to_string())


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #
def main() -> None:
    download_raw_dataset()
    df = load_raw_data()
    summarise(df)


if __name__ == "__main__":
    sys.exit(main())

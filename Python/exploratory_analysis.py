"""
exploratory_analysis.py
=======================
Module 3 of the Banking Transaction Anomaly & Customer Behaviour EDA project.

Responsibility
--------------
Run the exploratory analyses that the project was designed around and print
their results. Every number reported in the README / report is produced here.

Analyses
--------
  1. Distribution analysis      - shape of key numeric variables
  2. Category-wise analysis     - behaviour by transaction type / channel / occupation
  3. Trend analysis             - monthly transaction volume and value
  4. Time-based analysis        - hour-of-day, day-of-week, weekend patterns
  5. Comparison analysis        - credit vs debit, channel comparisons
  6. Relationship analysis      - relationships between numeric variables
  7. Correlation analysis       - correlation matrix of numeric features
  8. Anomaly analysis           - statistical outlier flags (IQR, z-score, login behaviour)

Author : [Your Name]
Course : AIML - Anudip Foundation
"""

from __future__ import annotations

import sys

import numpy as np
import pandas as pd

from data_cleaning import CLEANED_PATH

pd.set_option("display.width", 140)


def _banner(title: str) -> None:
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


# --------------------------------------------------------------------------- #
# 0. Overview
# --------------------------------------------------------------------------- #
def dataset_overview(df: pd.DataFrame) -> None:
    _banner("0. DATASET OVERVIEW")
    print(f"Transactions            : {len(df):,}")
    print(f"Unique accounts         : {df['AccountID'].nunique():,}")
    print(f"Unique merchants        : {df['MerchantID'].nunique():,}")
    print(f"Unique locations        : {df['Location'].nunique():,}")
    print(f"Unique devices          : {df['DeviceID'].nunique():,}")
    print(f"Period                  : {df['TransactionDate'].min()} -> {df['TransactionDate'].max()}")
    per_account = df.groupby("AccountID").size()
    print(f"Transactions per account: min={per_account.min()}, "
          f"median={per_account.median():.0f}, max={per_account.max()}")


# --------------------------------------------------------------------------- #
# 1. Distribution
# --------------------------------------------------------------------------- #
def distribution_analysis(df: pd.DataFrame) -> None:
    _banner("1. DISTRIBUTION ANALYSIS")
    cols = ["TransactionAmount", "AccountBalance", "CustomerAge", "TransactionDuration"]
    stats = df[cols].describe().T
    stats["skew"] = df[cols].skew()
    stats["median"] = df[cols].median()
    print(stats.to_string())

    amount = df["TransactionAmount"]
    q1, q2, q3 = amount.quantile([0.25, 0.5, 0.75])
    print(f"\nTransactionAmount quartiles: Q1={q1:.2f}  median={q2:.2f}  Q3={q3:.2f}")
    print(f"TransactionAmount mean={amount.mean():.2f}  skew={amount.skew():.2f} "
          f"({'right-skewed' if amount.skew() > 0 else 'left-skewed'})")
    for p in (0.90, 0.95, 0.99):
        print(f"  {int(p*100)}th percentile : {amount.quantile(p):.2f}")


# --------------------------------------------------------------------------- #
# 2. Category-wise
# --------------------------------------------------------------------------- #
def category_analysis(df: pd.DataFrame) -> None:
    _banner("2. CATEGORY-WISE ANALYSIS")
    for col in ["TransactionType", "Channel", "CustomerOccupation"]:
        agg = (df.groupby(col)
                 .agg(transactions=("TransactionID", "count"),
                      total_amount=("TransactionAmount", "sum"),
                      avg_amount=("TransactionAmount", "mean"),
                      avg_balance=("AccountBalance", "mean"))
                 .round(2)
                 .sort_values("transactions", ascending=False))
        print(f"\n--- by {col} ---")
        print(agg.to_string())


# --------------------------------------------------------------------------- #
# 3. Trend
# --------------------------------------------------------------------------- #
def trend_analysis(df: pd.DataFrame) -> None:
    _banner("3. TREND ANALYSIS (monthly)")
    monthly = (df.set_index("TransactionDate")
                 .resample("MS")
                 .agg(transactions=("TransactionID", "count"),
                      total_amount=("TransactionAmount", "sum"),
                      avg_amount=("TransactionAmount", "mean"))
                 .round(2))
    print(monthly.to_string())
    peak = monthly["total_amount"].idxmax()
    trough = monthly["total_amount"].idxmin()
    print(f"\nPeak month by value      : {peak:%b %Y} ({monthly.loc[peak, 'total_amount']:.2f})")
    print(f"Lowest month by value    : {trough:%b %Y} ({monthly.loc[trough, 'total_amount']:.2f})")
    print(f"Monthly average value    : {monthly['total_amount'].mean():.2f}")


# --------------------------------------------------------------------------- #
# 4. Time-based
# --------------------------------------------------------------------------- #
def time_based_analysis(df: pd.DataFrame) -> None:
    _banner("4. TIME-BASED ANALYSIS")
    print("--- by hour of day ---")
    hourly = df.groupby("TransactionHour").agg(
        transactions=("TransactionID", "count"),
        total_amount=("TransactionAmount", "sum")).round(2)
    print(hourly.to_string())
    print(f"Busiest hour : {hourly['transactions'].idxmax()}:00 "
          f"({hourly['transactions'].max()} transactions)")
    print(f"NOTE: transactions only occur between "
          f"{int(hourly.index.min()):02d}:00 and {int(hourly.index.max()):02d}:59.")

    print("\n--- by day of week ---")
    day_order = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    daily = (df.groupby("TransactionDayName")
               .agg(transactions=("TransactionID", "count"),
                    total_amount=("TransactionAmount", "sum"),
                    avg_amount=("TransactionAmount", "mean"))
               .reindex(day_order))
    # Days with no activity legitimately become 0 transactions / 0 value.
    daily[["transactions", "total_amount"]] = daily[["transactions", "total_amount"]].fillna(0)
    daily[[c for c in daily.columns if c != "transactions"]] = daily[
        [c for c in daily.columns if c != "transactions"]].round(2)
    print(daily.to_string())

    we = (df.groupby("IsWeekend")
            .agg(transactions=("TransactionID", "count"),
                 total_amount=("TransactionAmount", "sum"),
                 avg_amount=("TransactionAmount", "mean"))
            .reindex([False, True]))
    we[["transactions", "total_amount"]] = we[["transactions", "total_amount"]].fillna(0)
    we = we.round(2)
    we.index = ["Weekday", "Weekend"]
    print("\n--- weekend vs weekday ---")
    print(we.to_string())
    if we.loc["Weekend", "transactions"] == 0:
        print("\nNOTE: the dataset contains no Saturday/Sunday transactions;")
        print("      all activity falls on Monday-Friday.")


# --------------------------------------------------------------------------- #
# 5. Comparison
# --------------------------------------------------------------------------- #
def comparison_analysis(df: pd.DataFrame) -> None:
    _banner("5. COMPARISON ANALYSIS")
    print("--- Credit vs Debit ---")
    cd = df.groupby("TransactionType").agg(
        transactions=("TransactionID", "count"),
        total_amount=("TransactionAmount", "sum"),
        avg_amount=("TransactionAmount", "mean"),
        median_amount=("TransactionAmount", "median"),
        avg_duration=("TransactionDuration", "mean")).round(2)
    print(cd.to_string())

    print("\n--- Channel x TransactionType (average amount) ---")
    pivot = df.pivot_table(index="Channel", columns="TransactionType",
                           values="TransactionAmount", aggfunc="mean").round(2)
    print(pivot.to_string())

    print("\n--- Channel totals ---")
    ch = df.groupby("Channel").agg(
        transactions=("TransactionID", "count"),
        total_amount=("TransactionAmount", "sum"),
        share_pct=("TransactionAmount", lambda s: 100 * s.sum() / df["TransactionAmount"].sum())
    ).round(2)
    print(ch.to_string())


# --------------------------------------------------------------------------- #
# 6. Relationship
# --------------------------------------------------------------------------- #
def relationship_analysis(df: pd.DataFrame) -> None:
    _banner("6. RELATIONSHIP ANALYSIS")
    pairs = [
        ("TransactionAmount", "AccountBalance"),
        ("TransactionAmount", "TransactionDuration"),
        ("TransactionAmount", "CustomerAge"),
        ("TransactionAmount", "LoginAttempts"),
        ("AccountBalance", "CustomerAge"),
        ("AccountBalance", "TransactionDuration"),
    ]
    print("Pearson correlation between key pairs:")
    for a, b in pairs:
        r = df[[a, b]].corr().iloc[0, 1]
        print(f"  {a:20} vs {b:20} : r = {r: .3f}")

    print("\nAverage transaction amount by login attempts:")
    la = df.groupby("LoginAttempts").agg(
        transactions=("TransactionID", "count"),
        avg_amount=("TransactionAmount", "mean"),
        avg_balance=("AccountBalance", "mean")).round(2)
    print(la.to_string())


# --------------------------------------------------------------------------- #
# 7. Correlation
# --------------------------------------------------------------------------- #
def correlation_analysis(df: pd.DataFrame) -> pd.DataFrame:
    _banner("7. CORRELATION ANALYSIS")
    cols = ["TransactionAmount", "AccountBalance", "CustomerAge",
            "TransactionDuration", "LoginAttempts", "TransactionHour",
            "TimeSinceLastTransactionHours", "AmountToBalanceRatio"]
    corr = df[cols].corr().round(3)
    print(corr.to_string())

    # strongly correlated pairs (|r| >= 0.4, excluding the diagonal)
    pairs = []
    for i, a in enumerate(cols):
        for b in cols[i + 1:]:
            r = corr.loc[a, b]
            if abs(r) >= 0.4:
                pairs.append((a, b, r))
    print("\nNotable pairs (|r| >= 0.4):")
    print("  " + ("\n  ".join(f"{a} ~ {b}: {r}" for a, b, r in pairs) if pairs else "none"))
    return corr


# --------------------------------------------------------------------------- #
# 8. Anomaly
# --------------------------------------------------------------------------- #
def anomaly_analysis(df: pd.DataFrame) -> None:
    _banner("8. ANOMALY ANALYSIS (statistical, no ground-truth label)")
    amount = df["TransactionAmount"]

    # --- IQR rule ---------------------------------------------------------- #
    q1, q3 = amount.quantile([0.25, 0.75])
    iqr = q3 - q1
    upper_iqr = q3 + 1.5 * iqr
    df["IsAmountOutlierIQR"] = amount > upper_iqr

    # --- z-score rule ------------------------------------------------------ #
    z = (amount - amount.mean()) / amount.std(ddof=0)
    df["AmountZScore"] = z.round(3)
    df["IsAmountOutlierZ"] = z.abs() > 3

    # --- behavioural flags ------------------------------------------------- #
    df["HasMultipleLoginAttempts"] = df["LoginAttempts"] > 1
    df["HasHighLoginAttempts"] = df["LoginAttempts"] >= 3
    df["IsRapidRepeat"] = df["TimeSinceLastTransactionHours"].fillna(1e9) < 1.0
    df["HasHighBalanceUsage"] = df["AmountToBalanceRatio"] > 0.5

    flags = ["IsAmountOutlierIQR", "IsAmountOutlierZ", "HasHighLoginAttempts",
             "IsRapidRepeat", "HasHighBalanceUsage"]
    print("IQR upper bound for TransactionAmount :", round(upper_iqr, 2))
    print("\nAnomaly flag counts:")
    for f in flags:
        print(f"  {f:28}: {int(df[f].sum()):4}  ({100 * df[f].mean():.2f}%)")

    # composite risk flag: statistical outlier OR suspicious login behaviour
    df["IsAnomaly"] = df["IsAmountOutlierIQR"] | df["HasHighLoginAttempts"]
    print(f"\nComposite 'IsAnomaly' (amount outlier OR >=3 login attempts): "
          f"{int(df['IsAnomaly'].sum())} ({100 * df['IsAnomaly'].mean():.2f}%)")

    print("\nProfile of anomalous vs normal transactions:")
    profile = df.groupby("IsAnomaly").agg(
        transactions=("TransactionID", "count"),
        avg_amount=("TransactionAmount", "mean"),
        median_amount=("TransactionAmount", "median"),
        avg_balance=("AccountBalance", "mean"),
        avg_duration=("TransactionDuration", "mean"),
        avg_login=("LoginAttempts", "mean")).round(2)
    print(profile.to_string())

    print("\nTop 5 locations of anomalous transactions:")
    top_loc = (df[df["IsAnomaly"]]["Location"].value_counts().head(5))
    print(top_loc.to_string())

    print("\nAnomalous share by channel:")
    by_ch = (df.groupby("Channel")["IsAnomaly"].mean() * 100).round(2)
    print(by_ch.to_string())

    # persist the flags alongside the cleaned dataset
    df.to_csv(CLEANED_PATH, index=False)
    print(f"\n[analysis] Re-saved cleaned dataset with anomaly flags -> {CLEANED_PATH}")


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #
def load_cleaned() -> pd.DataFrame:
    df = pd.read_csv(CLEANED_PATH, parse_dates=["TransactionDate", "PreviousTransactionDate"])
    return df


def main() -> None:
    df = load_cleaned()
    dataset_overview(df)
    distribution_analysis(df)
    category_analysis(df)
    trend_analysis(df)
    time_based_analysis(df)
    comparison_analysis(df)
    relationship_analysis(df)
    correlation_analysis(df)
    anomaly_analysis(df)


if __name__ == "__main__":
    sys.exit(main())

"""
data_visualization.py
=====================
Module 4 of the Banking Transaction Anomaly & Customer Behaviour EDA project.

Responsibility
--------------
Render the charts that back the exploratory analysis and save them as PNGs in
the ``Visualizations/`` folder:

  * distribution_analysis.png  - distributions of key numeric variables
  * trend_analysis.png         - monthly transaction volume and value
  * category_analysis.png      - category-wise behaviour
  * correlation_analysis.png   - correlation heatmap + balance-usage scatter
  * anomaly_analysis.png       - statistical anomaly flags

Author : Muzamil banu
Course : AIML - Anudip Foundation
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # headless rendering

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from data_cleaning import CLEANED_PATH
from exploratory_analysis import load_cleaned, anomaly_analysis

BASE_DIR = Path(__file__).resolve().parent.parent
VIZ_DIR = BASE_DIR / "Visualizations"
VIZ_DIR.mkdir(parents=True, exist_ok=True)

sns.set_theme(style="whitegrid", palette="deep")
plt.rcParams.update({"figure.dpi": 110, "savefig.dpi": 150,
                     "axes.titleweight": "bold", "axes.titlesize": 12})


def _save(fig, name: str) -> Path:
    path = VIZ_DIR / name
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"[data_visualization] Saved -> {path}")
    return path


# --------------------------------------------------------------------------- #
# 1. Distribution
# --------------------------------------------------------------------------- #
def plot_distribution(df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    specs = [
        ("TransactionAmount", "Transaction Amount", "skyblue"),
        ("AccountBalance", "Account Balance", "salmon"),
        ("CustomerAge", "Customer Age", "mediumseagreen"),
        ("TransactionDuration", "Transaction Duration (seconds)", "mediumpurple"),
    ]
    for ax, (col, label, color) in zip(axes.ravel(), specs):
        sns.histplot(df[col], bins=30, kde=True, ax=ax, color=color)
        ax.axvline(df[col].mean(), color="black", ls="--", lw=1,
                   label=f"mean {df[col].mean():.1f}")
        ax.axvline(df[col].median(), color="red", ls=":", lw=1,
                   label=f"median {df[col].median():.1f}")
        ax.set_title(f"Distribution of {label}")
        ax.set_xlabel(label)
        ax.legend(fontsize=8)

    fig.suptitle("Distribution Analysis - Banking Transactions (n=2,512)",
                 fontsize=15, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    _save(fig, "distribution_analysis.png")


# --------------------------------------------------------------------------- #
# 2. Trend
# --------------------------------------------------------------------------- #
def plot_trend(df: pd.DataFrame) -> None:
    monthly = (df.set_index("TransactionDate").resample("MS")
                 .agg(transactions=("TransactionID", "count"),
                      total_amount=("TransactionAmount", "sum")))
    monthly = monthly[monthly["transactions"] > 0]
    labels = [d.strftime("%b\n%Y") for d in monthly.index]
    x = range(len(monthly))

    fig, axes = plt.subplots(2, 1, figsize=(13, 9), sharex=True)

    axes[0].plot(x, monthly["transactions"], marker="o", color="steelblue", lw=2)
    axes[0].fill_between(x, monthly["transactions"], alpha=0.15, color="steelblue")
    axes[0].set_title("Monthly Transaction Volume")
    axes[0].set_ylabel("Number of transactions")
    axes[0].grid(axis="y", alpha=0.4)

    axes[1].plot(x, monthly["total_amount"], marker="s", color="darkorange", lw=2)
    axes[1].fill_between(x, monthly["total_amount"], alpha=0.15, color="darkorange")
    axes[1].set_title("Monthly Transaction Value")
    axes[1].set_ylabel("Total amount")
    axes[1].set_xticks(list(x))
    axes[1].set_xticklabels(labels, fontsize=9)
    axes[1].grid(axis="y", alpha=0.4)

    peak = monthly["total_amount"].idxmax()
    axes[1].annotate(f"peak {peak:%b %Y}",
                     xy=(list(monthly.index).index(peak), monthly["total_amount"].max()),
                     xytext=(-10, 18), textcoords="offset points",
                     fontsize=9, color="darkred",
                     arrowprops=dict(arrowstyle="->", color="darkred"))

    # The final month is partial (the dataset ends on 1 Jan 2024).
    last = monthly.index[-1]
    axes[1].annotate(f"{last:%b %Y} partial\n(data ends {df['TransactionDate'].max():%d %b %Y})",
                     xy=(len(monthly) - 1, monthly["total_amount"].iloc[-1]),
                     xytext=(-95, 45), textcoords="offset points",
                     fontsize=8, color="dimgray",
                     arrowprops=dict(arrowstyle="->", color="dimgray", lw=0.8))

    fig.suptitle("Trend Analysis - Monthly Transaction Activity",
                 fontsize=15, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    _save(fig, "trend_analysis.png")


# --------------------------------------------------------------------------- #
# 3. Category-wise
# --------------------------------------------------------------------------- #
def plot_category(df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    sns.countplot(data=df, x="TransactionType", ax=axes[0, 0], color="steelblue")
    axes[0, 0].set_title("Transactions by Type")
    axes[0, 0].bar_label(axes[0, 0].containers[0])

    sns.countplot(data=df, x="Channel", ax=axes[0, 1], color="darkorange")
    axes[0, 1].set_title("Transactions by Channel")
    axes[0, 1].bar_label(axes[0, 1].containers[0])

    sns.countplot(data=df, y="CustomerOccupation", ax=axes[1, 0], color="mediumseagreen",
                  order=df["CustomerOccupation"].value_counts().index)
    axes[1, 0].set_title("Transactions by Customer Occupation")
    axes[1, 0].bar_label(axes[1, 0].containers[0])

    occ = (df.groupby("CustomerOccupation")["AccountBalance"].mean()
             .sort_values(ascending=False).reset_index())
    sns.barplot(data=occ, x="AccountBalance", y="CustomerOccupation",
                ax=axes[1, 1], color="mediumpurple")
    axes[1, 1].set_title("Average Account Balance by Occupation")
    axes[1, 1].set_xlabel("Average balance")
    axes[1, 1].bar_label(axes[1, 1].containers[0], fmt="%.0f", fontsize=8)

    fig.suptitle("Category-wise Analysis - Customer & Transaction Behaviour",
                 fontsize=15, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    _save(fig, "category_analysis.png")


# --------------------------------------------------------------------------- #
# 4. Correlation
# --------------------------------------------------------------------------- #
def plot_correlation(df: pd.DataFrame) -> None:
    cols = ["TransactionAmount", "AccountBalance", "CustomerAge",
            "TransactionDuration", "LoginAttempts", "TransactionHour",
            "TimeSinceLastTransactionHours", "AmountToBalanceRatio"]
    corr = df[cols].corr()

    fig, axes = plt.subplots(1, 2, figsize=(16, 7),
                             gridspec_kw={"width_ratios": [1.15, 1]})

    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0,
                square=True, linewidths=0.5, cbar_kws={"shrink": 0.8},
                ax=axes[0])
    axes[0].set_title("Correlation Heatmap of Numeric Features")

    sns.scatterplot(data=df, x="AccountBalance", y="TransactionAmount",
                    hue="TransactionType", alpha=0.55, s=22, ax=axes[1])
    axes[1].set_title("Transaction Amount vs Account Balance")
    axes[1].set_xlabel("Account balance")
    axes[1].set_ylabel("Transaction amount")

    fig.suptitle("Correlation & Relationship Analysis",
                 fontsize=15, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    _save(fig, "correlation_analysis.png")


# --------------------------------------------------------------------------- #
# 5. Anomaly
# --------------------------------------------------------------------------- #
def plot_anomaly(df: pd.DataFrame) -> None:
    q1, q3 = df["TransactionAmount"].quantile([0.25, 0.75])
    upper = q3 + 1.5 * (q3 - q1)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # amount distribution with IQR threshold
    sns.histplot(df["TransactionAmount"], bins=40, ax=axes[0, 0], color="skyblue")
    axes[0, 0].axvline(upper, color="red", ls="--", lw=1.4,
                       label=f"IQR upper bound = {upper:.0f}")
    axes[0, 0].set_title("Transaction Amount with Outlier Threshold")
    axes[0, 0].set_xlabel("Transaction amount")
    axes[0, 0].legend(fontsize=9)

    # login attempts
    sns.countplot(data=df, x="LoginAttempts", ax=axes[0, 1], color="salmon")
    axes[0, 1].set_title("Login Attempts Before Transaction")
    axes[0, 1].bar_label(axes[0, 1].containers[0], fontsize=8)

    # anomaly share by channel
    by_ch = (df.groupby("Channel")["IsAnomaly"].mean() * 100).reset_index()
    sns.barplot(data=by_ch, x="Channel", y="IsAnomaly", ax=axes[1, 0], color="mediumpurple")
    axes[1, 0].set_title("Anomalous Transaction Share by Channel")
    axes[1, 0].set_ylabel("% of transactions flagged")
    axes[1, 0].bar_label(axes[1, 0].containers[0], fmt="%.2f", fontsize=8)

    # amount: anomalous vs normal
    sns.boxplot(data=df, x="IsAnomaly", y="TransactionAmount",
                ax=axes[1, 1], hue="IsAnomaly", legend=False, palette="Set2")
    axes[1, 1].set_title("Transaction Amount - Normal vs Anomalous")
    axes[1, 1].set_xlabel("Is anomalous (amount outlier OR >=3 login attempts)")
    axes[1, 1].set_ylabel("Transaction amount")

    fig.suptitle("Anomaly Analysis - Statistical Outlier & Behaviour Flags",
                 fontsize=15, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    _save(fig, "anomaly_analysis.png")


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #
def main() -> None:
    df = load_cleaned()
    if "IsAnomaly" not in df.columns:  # ensure flags exist
        anomaly_analysis(df)

    plot_distribution(df)
    plot_trend(df)
    plot_category(df)
    plot_correlation(df)
    plot_anomaly(df)
    print(f"\n[data_visualization] All figures written to {VIZ_DIR}")


if __name__ == "__main__":
    sys.exit(main())

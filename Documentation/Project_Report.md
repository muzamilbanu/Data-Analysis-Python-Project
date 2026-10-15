# Project Report
## Banking Transaction Anomaly & Customer Behaviour EDA

| | |
|---|---|
| **Industry** | IDFC FIRST Bank — Banking & Financial Services (BFSI) |
| **Organization** | Anudip Foundation |
| **Course** | AIML |
| **Author** | Muzamil banu |
| **Student ID** | [Your Student ID] |
| **Batch Code** | [Your Batch Code] |
| **Tools** | Python, Jupyter Notebook, NumPy, Pandas, Matplotlib, Seaborn, uv |

---

## 1. Executive Summary

This project performs an end-to-end exploratory data analysis (EDA) of **2,512
retail-banking transactions** spanning **January 2023 to January 2024** across
**495 accounts**. The work has two goals: describe **customer behaviour**
(across channels, transaction types, occupations and time) and identify
**statistical anomalies** (unusually large amounts and repeated-login
behaviour) that merit fraud-risk attention.

The analysis shows that transaction amounts are strongly right-skewed, that
Debits outnumber Credits ~3.4:1, that channels are used almost equally, and that
occupation is strongly tied to account balance. Using IQR and z-score rules plus
login-behaviour flags, **205 transactions (8.16%)** are flagged as anomalous; on
average these are far larger and involve roughly 2.4× the login attempts of
normal transactions. All findings are derived from the actual dataset by the
code in `Python/` and the executed notebook.

> The dataset is **synthetic** and has **no ground-truth fraud label**. Anomalies
> reported here are statistical/behavioural flags, not confirmed fraud.

---

## 2. Problem Statement

Retail banks face two persistent BFSI challenges:

1. **Anomaly / fraud-risk monitoring** — a small share of transactions are much
   larger than normal or occur after suspicious repeated login attempts, and are
   difficult to isolate in high-volume data.
2. **Customer-behaviour understanding** — spending differs by channel
   (ATM/Online/Branch), transaction type (Credit/Debit) and occupation, which
   matters for segmentation and risk policy.

## 3. Objectives

- Profile the distributions of amount, balance, age and duration.
- Quantify activity and value by type, channel and occupation.
- Examine monthly, day-of-week and hour-of-day patterns.
- Detect and profile statistical/behavioural anomalies.
- Measure relationships among numeric features.
- Translate results into practical recommendations.

## 4. Dataset

| Item | Detail |
|---|---|
| Name | Bank Transaction Dataset for Fraud Detection |
| Source | Kaggle — `valakhorasani/bank-transaction-dataset-for-fraud-detection` |
| Original file | `bank_transactions_data_2.csv` |
| Size | 2,512 rows × 16 columns |
| Period | 2 Jan 2023 → 1 Jan 2024 |
| Accounts / Merchants / Locations / Devices | 495 / 100 / 43 / 681 |
| Currency | not specified by the dataset (values reported unit-less) |

**Columns:** TransactionID, AccountID, TransactionAmount, TransactionDate,
TransactionType, Location, DeviceID, IP Address, MerchantID, Channel,
CustomerAge, CustomerOccupation, TransactionDuration, LoginAttempts,
AccountBalance, PreviousTransactionDate.

**Quality:** 0 missing, 0 duplicates, 0 negative amounts.

## 5. Methodology

### 5.1 Workflow

Industry Selection → Problem Identification → Dataset Collection →
Data Cleaning → Data Transformation → Data Analysis → Data Visualization →
Insights → Recommendations.

### 5.2 Data Cleaning (`data_cleaning.py`)

- Checked and confirmed no missing values and no duplicate rows.
- Enforced numeric types (amount, age, duration, login attempts, balance) and
  parsed `TransactionDate` / `PreviousTransactionDate` as datetimes.
- Trimmed whitespace from text columns; sorted by account and time.
- Currency/reference note: `PreviousTransactionDate` is a dataset snapshot
  timestamp (all values cluster around 2024-11-04), not a true per-account
  previous transaction, so it was **not** used as a transaction-gap measure.

### 5.3 Data Transformation (engineered features)

`TransactionYear`, `TransactionMonth`, `TransactionMonthName`,
`TransactionDayOfWeek`, `TransactionDayName`, `TransactionHour`, `IsWeekend`,
`TimeSinceLastTransactionHours` (per-account transaction velocity),
`AmountToBalanceRatio`, and `SignedAmount` (negative for Debits, positive for
Credits). The analysis stage adds the anomaly flags.

## 6. Analysis & Results

### 6.1 Distribution Analysis

| Variable | Mean | Median | Min | Max | Std | Skew |
|---|---|---|---|---|---|---|
| TransactionAmount | 297.59 | 211.14 | 0.26 | 1,919.11 | 291.95 | +1.74 |
| AccountBalance | 5,114.30 | 4,735.51 | 101.25 | 14,977.99 | 3,900.94 | +0.60 |
| CustomerAge | 44.67 | 45.00 | 18 | 80 | 17.79 | +0.15 |
| TransactionDuration (s) | 119.64 | 112.50 | 10 | 300 | 69.96 | +0.60 |

Transaction amount quartiles: Q1 = 81.89, Q3 = 414.53. Percentiles: 90th =
701.31, 95th = 878.18, 99th = 1,360.60 — confirming a long right tail.

### 6.2 Category-wise Analysis

**By transaction type**

| Type | Transactions | Total value | Avg value |
|---|---|---|---|
| Debit | 1,944 (77.4%) | 573,463.00 | 294.99 |
| Credit | 568 (22.6%) | 174,092.57 | 306.50 |

**By channel**

| Channel | Transactions | Total value | Share of value |
|---|---|---|---|
| Branch | 868 | 250,183.00 | 33.47% |
| ATM | 833 | 256,331.43 | 34.29% |
| Online | 811 | 241,041.14 | 32.24% |

**By occupation**

| Occupation | Transactions | Avg amount | Avg balance |
|---|---|---|---|
| Student | 657 | 313.22 | 1,570.21 |
| Doctor | 631 | 292.70 | 8,978.99 |
| Engineer | 625 | 289.04 | 5,486.41 |
| Retired | 599 | 294.53 | 4,542.16 |

### 6.3 Trend Analysis (monthly)

Peak month by value: **Sep 2023 — 72,832.25**. Weakest full month: **Apr 2023 —
41,003.84**. Monthly average value: **57,504.27**. January 2024 is partial (13
transactions; data ends 1 Jan 2024).

### 6.4 Time-based Analysis

- **Hour of day:** activity occurs only between **16:00 and 18:59**;
  16:00 = 1,316 transactions (52.4%), 17:00 = 819, 18:00 = 377.
- **Day of week:** Monday = 1,070 (42.6%), Fri = 373, Thu = 368, Tue = 360,
  Wed = 341; **Sat = Sun = 0** (no weekend activity).

### 6.5 Comparison Analysis

Credit transactions have a slightly higher average (306.50 vs 294.99) and median
(223.36 vs 205.14) than Debits. Channel × type averages range narrowly from
276.63 (Branch Debits) to 316.75 (Branch Credits).

### 6.6 Relationship & Correlation Analysis

| Pair | Pearson r |
|---|---|
| TransactionAmount ~ AccountBalance | −0.025 |
| TransactionAmount ~ TransactionDuration | 0.004 |
| TransactionAmount ~ CustomerAge | −0.026 |
| TransactionAmount ~ LoginAttempts | −0.008 |
| AccountBalance ~ CustomerAge | 0.320 |
| AmountToBalanceRatio ~ TransactionAmount | 0.390 |
| AmountToBalanceRatio ~ AccountBalance | −0.363 |

No pair reaches |r| ≥ 0.4; the notable (moderate) links involve the derived
amount-to-balance ratio and the balance↔age relationship.

### 6.7 Anomaly Analysis

IQR upper bound for TransactionAmount = **913.49**.

| Flag | Count | Share |
|---|---|---|
| Amount outlier (IQR) | 113 | 4.50% |
| Amount outlier (z-score > 3) | 48 | 1.91% |
| ≥ 3 login attempts | 95 | 3.78% |
| Rapid repeat (< 1 h gap) | 16 | 0.64% |
| High balance usage (ratio > 0.5) | 228 | 9.08% |
| **Composite (large amount OR ≥3 logins)** | **205** | **8.16%** |

Anomalous vs normal:

| Group | Avg amount | Median amount | Avg logins | Avg balance |
|---|---|---|---|---|
| Normal | 255.73 | 199.02 | 1.01 | 5,124.05 |
| Anomalous | 768.76 | 959.48 | 2.40 | 5,004.63 |

Anomalous share by channel: **Online 9.12%**, ATM 8.40%, Branch 7.03%.

## 7. Key Insights

1. Transaction amounts are right-skewed (median 211.14, mean 297.59, skew +1.74).
2. Debits dominate frequency (77.4%) but not ticket size.
3. Channels are near-equally used (each ≈ one-third of value).
4. Occupation strongly separates average balance (Doctor 8,978.99 → Student
   1,570.21); Students spend the largest average amount from the smallest
   balances.
5. Activity peaks in September 2023; April 2023 is the weakest full month.
6. All activity sits in the 16:00–18:59 window and on weekdays, with Mondays
   dominant (42.6%).
7. Numeric features are largely uncorrelated (no |r| ≥ 0.4).
8. A clear anomaly tail exists: 8.16% of transactions are flagged, dominated by
   large amounts and repeated login attempts.
9. Flagged transactions are much larger (768.76 vs 255.73) and involve ~2.4×
   the login attempts of normal ones.
10. Online has the highest anomalous share (9.12%).

## 8. Recommendations

1. Apply amount-based review around the 913.49 IQR threshold (4.5% of volume).
2. Use ≥3 login attempts as a strong risk signal; step-up authentication from the
   2nd failed attempt.
3. Concentrate monitoring on the Online channel (9.12% anomalous share), crossed
   with the amount and login rules.
4. Design occupation/balance-band risk segments; monitor high-balance-usage
   cohorts (Students).
5. Treat the restricted time window (16:00–18:59, weekdays) as a data-collection
   gap, not as proof that off-hours activity is absent.
6. Add a ground-truth fraud label (or device/location reputation data) so the
   statistical flags can become a trainable risk model.

## 9. Limitations

- Synthetic data with no confirmed fraud labels → anomalies are statistical only.
- Transaction timestamps cover only 16:00–18:59 on weekdays, limiting
  time-of-day / weekend analysis.
- `PreviousTransactionDate` is a snapshot timestamp and could not be used as a
  per-account previous-transaction reference.

## 10. Conclusion

The project delivers a reproducible, documented EDA pipeline for retail-banking
transactions. It characterises customer behaviour across segment, channel and
time dimensions, and produces a transparent, rule-based anomaly shortlist
(8.16% of transactions) that is immediately useful as a triage layer — and
ready to be extended into a supervised risk model once labels are available.

## Appendix — Repository Map

| Path | Purpose |
|---|---|
| `Python/data_loading.py` | Download + load + summary |
| `Python/data_cleaning.py` | Cleaning + feature engineering |
| `Python/exploratory_analysis.py` | The eight analyses |
| `Python/data_visualization.py` | The five figures |
| `Notebook/Data_Analysis_EDA.ipynb` | Executed narrative notebook |
| `Dataset/raw_dataset.csv` | Original data |
| `Dataset/cleaned_dataset.csv` | Cleaned + features + anomaly flags |
| `Visualizations/*.png` | Analysis figures |

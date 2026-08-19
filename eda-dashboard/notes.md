# Matplotlib EDA — Customer Churn Dataset

## Industry Context

Exploratory Data Analysis with Matplotlib is a baseline expectation for entry-level
ML Engineer roles. Before any modeling work, engineers are expected to understand
class balance, categorical relationships, and distribution shapes in the data —
skipping this step is a common way weak candidates get filtered out early.
A first EDA project on a real-world tabular dataset (handling missing values,
producing multiple visualizations, and pulling out actionable insights) is a
standard first portfolio milestone for this role.

## Dataset

Telco Customer Churn (Kaggle, blastchar). 7043 rows, 21 columns.
`TotalCharges` is stored as a string in the raw CSV — 11 rows contain blank
values (all corresponding to customers with `tenure == 0`, i.e. brand-new
accounts with no billing history yet). These rows were dropped after
converting the column with `pd.to_numeric(errors="coerce")`, since partial
billing data isn't usable for the charges-related analysis and the count is
small enough (0.16% of rows) that dropping doesn't bias the dataset.

Post-cleaning shape: 7032 rows.

## Core Concepts Practiced

- **Categorical distribution with percentage annotation**: `value_counts()` +
  `value_counts(normalize=True)` combined to show both raw counts and
  proportions on the same chart.
- **Crosstab for grouped comparison**: `pd.crosstab()` to compare a categorical
  target (`Churn`) across another categorical feature (`Contract`), with
  explicit `reindex()` to enforce a logical category order instead of
  alphabetical default.
- **Density-normalized histograms**: `density=True` used instead of raw counts
  when comparing two groups of very different sizes (5163 vs 1869 customers).
  Without normalization, the larger group visually dominates and the shape
  comparison becomes misleading.

## Task

Answer three EDA questions using Matplotlib, each saved as a separate figure:

- **Q1 — Overall churn rate**: What proportion of customers churn?
- **Q2 — Churn by contract type**: Does contract length affect churn?
- **Q3 — Tenure distribution by churn status**: Do churned customers leave earlier
  in their lifecycle than retained customers?

## Output & Findings

All figures saved to `visuals/matplotlib/` at 300 DPI.

| File | Finding |
|---|---|
| `q1_churn_rate.png` | 26.6% of customers have churned — confirms the dataset's known class imbalance (~3:1 retained-to-churned). |
| `q2_churn_by_contract.png` | Month-to-month contracts show near-parity between churned and retained customers, while One year and Two year contracts show sharply lower churn — contract length is a strong retention signal. |
| `q3_tenure_distribution.png` | Churned customers are heavily concentrated in the first ~10 months of tenure; retained customers skew toward long tenure (60+ months). Tenure is one of the strongest early-warning signals for churn in this dataset. |

## Notes

- Chart color/style kept at Matplotlib defaults — no manual palette customization,
  since the goal here was distribution/comparison clarity, not presentation polish.
- `plt.close(fig)` called after each save to avoid memory buildup when generating
  multiple figures in one script run.
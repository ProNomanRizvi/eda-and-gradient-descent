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

<br>

# Seaborn EDA — Customer Churn Dataset

## Industry Context

Seaborn extends Matplotlib for statistical visualization — its core value is
built-in aggregation and grouping with less code (e.g. `hue=` for automatic
group comparison). Two Seaborn plot types are near-mandatory in any real-world
EDA project: correlation heatmaps for numeric feature relationships, and
boxplot/violinplot for distribution spread and outlier detection. A first EDA
project is expected to demonstrate all three types of variable analysis —
univariate, bivariate, and multivariate — not just bar charts.

## Core Concepts Practiced

- **`hue=` for grouped statistical plots**: `sns.histplot()` and
  `sns.countplot()` both accept `hue="Churn"`, letting one call replace what
  would otherwise need manual grouping and multiple `ax.bar()`/`ax.hist()`
  calls in Matplotlib.
- **Density normalization across unequal group sizes**: initial `sns.histplot()`
  used raw counts, which made the minority class (Churn: Yes, 1869 customers)
  visually disappear next to the majority class (5163 customers). Fixed with
  `stat="density", common_norm=False` — each group is normalized to its own
  area, making the shape comparison fair regardless of group size. This is
  the same principle applied in the Matplotlib tenure histogram (Q3), just
  surfaced as an explicit bug here since Seaborn's default (`stat="count"`)
  doesn't normalize automatically.
- **Correlation heatmap**: `.corr()` + `sns.heatmap(annot=True)` for a
  numeric feature correlation matrix — standard first step in any EDA to
  catch multicollinearity or confirm expected relationships.
- **Boxplot for outlier detection**: `sns.boxplot()` shows median, IQR, and
  flags points beyond 1.5×IQR as individual outlier markers. Added as a bonus
  task after an industry-gap check flagged its absence from the original
  question set — outlier/spread analysis is a standard expectation in
  Seaborn-based EDA that the KDE and countplots alone don't cover.

## Task

Answer four EDA questions using Seaborn, plus one bonus outlier-detection task:

- **Q4 — Monthly charges distribution by churn**: Do churned customers pay more?
- **Q5 — Correlation heatmap**: How do tenure, MonthlyCharges, and TotalCharges relate?
- **Q6 — Internet service type vs churn**: Does service type affect churn?
- **Q7 — Payment method vs churn**: Does payment method affect churn?
- **Bonus — Monthly charges boxplot by churn**: Are there outliers or spread
  differences in charges between churned and retained customers?

## Output & Findings

All figures saved to `visuals/seaborn/` at 300 DPI.

| File | Finding |
|---|---|
| `q4_monthly_charges_by_churn.png` | Churned customers are concentrated in the $70–100 range; retained customers peak near $20 (likely no-internet-service accounts). Higher monthly charges are associated with higher churn. |
| `q5_correlation_heatmap.png` | tenure–TotalCharges: 0.83 (strong — expected, since charges accumulate over time). tenure–MonthlyCharges: 0.25 (weak — longer-tenured customers don't necessarily pay more per month, they just accumulate more total billing). |
| `q6_internet_service_vs_churn.png` | Fiber optic customers have the highest churn count of the three service types; customers with no internet service churn the least. |
| `q7_payment_method_vs_churn.png` | Electronic check has by far the highest churn count among payment methods — the other three (mailed check, bank transfer, credit card) show much lower churn. |
| `q_bonus_monthlycharges_boxplot.png` | No IQR-based outliers in MonthlyCharges for either group (data is naturally bounded, ~$18–$119). The real signal is a median shift: Churn=Yes median (~$79) sits well above Churn=No median (~$64), confirming the Q4 finding through a different lens. |

## Notes

- The `TotalCharges` cleaning step (convert to numeric, drop 11 rows) is
  duplicated from `01_eda_matplotlib.py` rather than shared, since each
  script in this project is kept standalone/self-contained. Acceptable at
  this scale; would move to a shared `utils.py` if the project grew further.
- The Q4 density-normalization bug is a good example of why group-size
  imbalance needs to be checked explicitly for every new plot type — the
  same fix from Matplotlib's Q3 didn't carry over automatically because
  Seaborn's default statistic is different from what was used there.
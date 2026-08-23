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

# Plotly EDA — Customer Churn Dataset

## Industry Context

Plotly's core value over Matplotlib/Seaborn is interactivity — hover tooltips,
zoom, and pan — which matters for dashboards and stakeholder-facing work where
static screenshots aren't enough. For a portfolio project, including a Plotly
section signals the ability to build interactive dashboards, not just static
reports, which is an increasingly expected skill for entry-level data roles.

## Core Concepts Practiced

- **Interactive hover context**: `hover_data=["customerID", "Contract"]` in
  the Q8 scatter lets a viewer inspect individual customers without adding
  visual clutter to the chart itself — something a static plot can't offer.
- **Rate over raw count for imbalanced groups**: Q9 and Q10 both calculate
  churn *rate* (`groupby().apply(lambda x: (x == "Yes").mean() * 100)`)
  instead of raw churn counts. This matters because groups being compared
  (e.g. senior vs non-senior citizens) are very different sizes — comparing
  raw counts would be misleading, rate comparison isn't.
- **Reshaping wide to long format for combined comparison**: Q10 needed to
  compare two separate binary columns (`Partner`, `Dependents`) against churn
  in a single grouped bar chart. Solved by building two long-format
  sub-tables (one per column, each tagged with a `Relationship` label) and
  concatenating them, so both comparisons render side-by-side in one figure
  instead of two separate charts.
- **Overplotting correction**: the Q8 scatter (7032 points) initially had no
  transparency, making dense regions an indistinguishable solid mass and
  hiding the actual point density. Fixed with `opacity=0.5`.
- **Dual export**: every figure is saved as both a static PNG
  (`write_image()`, for READMEs and quick viewing) and an interactive HTML
  file (`write_html()`, for demonstrating the actual interactivity).

## Task

Answer three EDA questions using Plotly Express:

- **Q8 — Tenure vs Monthly Charges**: Is there a visible relationship between
  how long a customer has stayed and what they pay, split by churn status?
- **Q9 — Senior citizen churn rate**: Do senior citizens churn at a different
  rate than non-seniors?
- **Q10 — Partner/Dependents vs churn**: Does having a partner or dependents
  affect churn rate?

## Output & Findings

All figures saved to `visuals/plotly/` as both PNG (300 DPI equivalent via
`scale=2`) and interactive HTML.

| File | Finding |
|---|---|
| `q8_tenure_vs_monthlycharges.png/.html` | Churned customers (red) are heavily concentrated in the low-tenure region; retained customers (blue) dominate the high-tenure region. No strong linear relationship between tenure and monthly charges themselves — churn risk is driven more by tenure than by charge level directly. |
| `q9_senior_citizen_churn_rate.png/.html` | Senior citizens churn at 41.7%, nearly double the non-senior rate of 23.7% — one of the strongest single-feature churn signals in the dataset. |
| `q10_partner_dependents_churn.png/.html` | Customers without a partner churn at 33.0% vs 19.7% for those with one. Customers without dependents churn at 31.3% vs 15.5% for those with dependents. Both relationship factors are protective against churn, with dependents showing the larger effect. |

## Notes

- The `TotalCharges` cleaning step is duplicated from the Matplotlib/Seaborn
  scripts, consistent with the standalone-script pattern used throughout
  this project.
- Overplotting in Q8 is a good example of a visualization bug that isn't a
  statistical error — the underlying data and aggregation were correct, but
  the chart's readability was compromised until transparency was added. Both
  types of issues (statistical and visual) need separate checks during review.
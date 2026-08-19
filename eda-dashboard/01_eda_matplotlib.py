import pandas as pd
import matplotlib.pyplot as plt


# ---------------------------------------------------------
# 1. Load and clean data
# ---------------------------------------------------------

DATA_PATH = "data/telco_customer_churn.csv"
OUTPUT_DIR = "visuals/matplotlib"

df = pd.read_csv(DATA_PATH)

# TotalCharges is stored as text in the raw dataset.
# Convert it to numeric values. Invalid/blank values become NaN.
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

nan_count = df["TotalCharges"].isna().sum()
print(f"Rows with missing TotalCharges: {nan_count}")

# Decision:
# Drop rows where TotalCharges could not be converted.
# These records contain incomplete billing information, so we
# avoid keeping partially invalid customer records in the EDA.
df = df.dropna(subset=["TotalCharges"]).copy()

print(f"Dataset shape after cleaning: {df.shape}")


# ---------------------------------------------------------
# Q1: Overall Churn Rate
# ---------------------------------------------------------

churn_counts = df["Churn"].value_counts()

churn_percentages = (
    df["Churn"]
    .value_counts(normalize=True)
    .mul(100)
)

fig, ax = plt.subplots(figsize=(8, 6))

bars = ax.bar(
    churn_counts.index,
    churn_counts.values
)

ax.set_title("Overall Customer Churn Rate")
ax.set_xlabel("Churn Status")
ax.set_ylabel("Number of Customers")

# Add percentage labels above each bar
for bar, status in zip(bars, churn_counts.index):
    percentage = churn_percentages[status]

    ax.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height(),
        f"{percentage:.1f}%",
        ha="center",
        va="bottom",
        fontsize=11
    )

ax.set_ylim(0, max(churn_counts.values) * 1.12)

fig.tight_layout()
fig.savefig(
    f"{OUTPUT_DIR}/q1_churn_rate.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close(fig)


# ---------------------------------------------------------
# Q2: Churn by Contract Type
# ---------------------------------------------------------

contract_churn = pd.crosstab(
    df["Contract"],
    df["Churn"]
)

# Keep contract types in a logical order
contract_order = [
    "Month-to-month",
    "One year",
    "Two year"
]

contract_churn = contract_churn.reindex(contract_order)

fig, ax = plt.subplots(figsize=(9, 6))

contract_churn.plot(
    kind="bar",
    ax=ax
)

ax.set_title("Customer Churn by Contract Type")
ax.set_xlabel("Contract Type")
ax.set_ylabel("Number of Customers")

ax.legend(
    title="Churn",
    labels=["No", "Yes"]
)

plt.xticks(rotation=0)

fig.tight_layout()
fig.savefig(
    f"{OUTPUT_DIR}/q2_churn_by_contract.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close(fig)


# ---------------------------------------------------------
# Q3: Tenure Distribution — Churned vs Non-Churned
# ---------------------------------------------------------

churned_tenure = df.loc[
    df["Churn"] == "Yes",
    "tenure"
]

non_churned_tenure = df.loc[
    df["Churn"] == "No",
    "tenure"
]

fig, ax = plt.subplots(figsize=(9, 6))

ax.hist(
    non_churned_tenure,
    bins=30,
    alpha=0.6,
    density=True,
    label="Churn: No"
)

ax.hist(
    churned_tenure,
    bins=30,
    alpha=0.6,
    density=True,
    label="Churn: Yes"
)

ax.set_title("Tenure Distribution by Customer Churn Status")
ax.set_xlabel("Tenure (Months)")
ax.set_ylabel("Density")

ax.legend(title="Customer Status")

fig.tight_layout()
fig.savefig(
    f"{OUTPUT_DIR}/q3_tenure_distribution.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close(fig)


print("\nAll Matplotlib visualizations saved successfully.")
import os

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

DATA_PATH = "data/telco_customer_churn.csv"
OUTPUT_DIR = "visuals/seaborn"

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ---------------------------------------------------------
# Load and clean data
# ---------------------------------------------------------

df = pd.read_csv(DATA_PATH)

# TotalCharges is stored as text in the raw dataset.
# Convert it to numeric. Invalid/blank values become NaN.
df["TotalCharges"] = pd.to_numeric(
    df["TotalCharges"],
    errors="coerce"
)

nan_count = df["TotalCharges"].isna().sum()

print(f"Rows with missing TotalCharges: {nan_count}")

# Decision:
# Drop rows where TotalCharges could not be converted.
# These records have incomplete billing information, so we
# remove them instead of using unreliable values.
df = df.dropna(subset=["TotalCharges"]).copy()

print(f"Dataset shape after cleaning: {df.shape}")


# ---------------------------------------------------------
# Q4: Monthly Charges Distribution by Churn
# ---------------------------------------------------------

fig, ax = plt.subplots(figsize=(9, 6))

sns.histplot(
    data=df,
    x="MonthlyCharges",
    hue="Churn",
    kde=True,
    bins=30,
    alpha=0.45,
    stat="density",      
    common_norm=False,
    ax=ax
)

ax.set_title("Monthly Charges Distribution by Churn")
ax.set_xlabel("Monthly Charges")
ax.set_ylabel("Number of Customers")

fig.tight_layout()

fig.savefig(
    f"{OUTPUT_DIR}/q4_monthly_charges_by_churn.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close(fig)


# ---------------------------------------------------------
# Q5: Correlation Heatmap
# ---------------------------------------------------------

correlation = df[
    ["tenure", "MonthlyCharges", "TotalCharges"]
].corr()

fig, ax = plt.subplots(figsize=(7, 6))

sns.heatmap(
    correlation,
    annot=True,
    cmap="coolwarm",
    fmt=".2f",
    linewidths=0.5,
    ax=ax
)

ax.set_title("Correlation Between Customer Charges and Tenure")

fig.tight_layout()

fig.savefig(
    f"{OUTPUT_DIR}/q5_correlation_heatmap.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close(fig)


# ---------------------------------------------------------
# Q6: Internet Service vs Churn
# ---------------------------------------------------------

fig, ax = plt.subplots(figsize=(9, 6))

sns.countplot(
    data=df,
    x="InternetService",
    hue="Churn",
    ax=ax
)

ax.set_title("Customer Churn by Internet Service Type")
ax.set_xlabel("Internet Service")
ax.set_ylabel("Number of Customers")

ax.legend(
    title="Churn",
    labels=["No", "Yes"]
)

fig.tight_layout()

fig.savefig(
    f"{OUTPUT_DIR}/q6_internet_service_vs_churn.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close(fig)


# ---------------------------------------------------------
# Q7: Payment Method vs Churn
# ---------------------------------------------------------

fig, ax = plt.subplots(figsize=(11, 6))

sns.countplot(
    data=df,
    x="PaymentMethod",
    hue="Churn",
    ax=ax
)

ax.set_title("Customer Churn by Payment Method")
ax.set_xlabel("Payment Method")
ax.set_ylabel("Number of Customers")

# Payment method names are long, so rotate them for readability.
plt.xticks(
    rotation=30,
    ha="right"
)

ax.legend(
    title="Churn",
    labels=["No", "Yes"]
)

fig.tight_layout()

fig.savefig(
    f"{OUTPUT_DIR}/q7_payment_method_vs_churn.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close(fig)


print("\nAll Seaborn visualizations saved successfully.")
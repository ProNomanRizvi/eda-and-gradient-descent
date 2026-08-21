import os

import pandas as pd
import plotly.express as px


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

DATA_PATH = "data/telco_customer_churn.csv"
OUTPUT_DIR = "visuals/plotly"

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
# Drop rows where TotalCharges could not be converted because
# these records contain incomplete billing information.
df = df.dropna(subset=["TotalCharges"]).copy()

print(f"Dataset shape after cleaning: {df.shape}")


# ---------------------------------------------------------
# Q8: Tenure vs Monthly Charges
# ---------------------------------------------------------

fig = px.scatter(
    df,
    x="tenure",
    y="MonthlyCharges",
    color="Churn",
    opacity=0.5,
    hover_data=["customerID", "Contract"],
    title="Tenure vs Monthly Charges by Churn Status",
    labels={
        "tenure": "Tenure (Months)",
        "MonthlyCharges": "Monthly Charges",
        "Churn": "Churn Status"
    }
)

fig.write_image(
    f"{OUTPUT_DIR}/q8_tenure_vs_monthlycharges.png",
    width=1200,
    height=700,
    scale=2
)

# Optional interactive HTML version
fig.write_html(
    f"{OUTPUT_DIR}/q8_tenure_vs_monthlycharges.html"
)


# ---------------------------------------------------------
# Q9: Senior Citizen Churn Rate
# ---------------------------------------------------------

# Convert binary SeniorCitizen values into readable labels.
df["SeniorCitizenLabel"] = df["SeniorCitizen"].map({
    0: "No",
    1: "Yes"
})

# Calculate churn rate for each SeniorCitizen group.
senior_churn_rate = (
    df.groupby("SeniorCitizenLabel")["Churn"]
    .apply(lambda x: (x == "Yes").mean() * 100)
    .reset_index(name="ChurnRate")
)

# Keep a logical order.
senior_churn_rate["SeniorCitizenLabel"] = pd.Categorical(
    senior_churn_rate["SeniorCitizenLabel"],
    categories=["No", "Yes"],
    ordered=True
)

senior_churn_rate = senior_churn_rate.sort_values(
    "SeniorCitizenLabel"
)

fig = px.bar(
    senior_churn_rate,
    x="SeniorCitizenLabel",
    y="ChurnRate",
    text="ChurnRate",
    title="Churn Rate by Senior Citizen Status",
    labels={
        "SeniorCitizenLabel": "Senior Citizen",
        "ChurnRate": "Churn Rate (%)"
    }
)

fig.update_traces(
    texttemplate="%{text:.1f}%",
    textposition="outside"
)

fig.update_yaxes(
    range=[
        0,
        senior_churn_rate["ChurnRate"].max() * 1.15
    ]
)

fig.write_image(
    f"{OUTPUT_DIR}/q9_senior_citizen_churn_rate.png",
    width=1000,
    height=650,
    scale=2
)

fig.write_html(
    f"{OUTPUT_DIR}/q9_senior_citizen_churn_rate.html"
)


# ---------------------------------------------------------
# Q10: Partner / Dependents vs Churn
# ---------------------------------------------------------

# Convert Partner and Dependents into one long-format table.
partner_data = df[["Churn", "Partner"]].copy()
partner_data["Relationship"] = "Partner"
partner_data.rename(
    columns={"Partner": "Status"},
    inplace=True
)

dependents_data = df[["Churn", "Dependents"]].copy()
dependents_data["Relationship"] = "Dependents"
dependents_data.rename(
    columns={"Dependents": "Status"},
    inplace=True
)

relationship_df = pd.concat(
    [partner_data, dependents_data],
    ignore_index=True
)

# Calculate churn rate for each relationship/status combination.
relationship_churn_rate = (
    relationship_df
    .groupby(["Relationship", "Status"])["Churn"]
    .apply(lambda x: (x == "Yes").mean() * 100)
    .reset_index(name="ChurnRate")
)

fig = px.bar(
    relationship_churn_rate,
    x="Relationship",
    y="ChurnRate",
    color="Status",
    barmode="group",
    text="ChurnRate",
    title="Churn Rate by Partner and Dependent Status",
    labels={
        "Relationship": "Relationship Type",
        "ChurnRate": "Churn Rate (%)",
        "Status": "Status"
    },
    category_orders={
        "Relationship": ["Partner", "Dependents"],
        "Status": ["No", "Yes"]
    }
)

fig.update_traces(
    texttemplate="%{text:.1f}%",
    textposition="outside"
)

fig.update_yaxes(
    range=[
        0,
        relationship_churn_rate["ChurnRate"].max() * 1.15
    ]
)

fig.write_image(
    f"{OUTPUT_DIR}/q10_partner_dependents_churn.png",
    width=1100,
    height=700,
    scale=2
)

fig.write_html(
    f"{OUTPUT_DIR}/q10_partner_dependents_churn.html"
)


print("\nAll Plotly visualizations saved successfully.")
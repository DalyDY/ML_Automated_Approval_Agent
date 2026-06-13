import pandas as pd

# ==========================
# Load Dataset
# ==========================
df = pd.read_csv("prosperLoanData.csv")

print(f"Original Shape: {df.shape}")

# ==========================
# Remove Duplicate Rows
# ==========================
duplicates = df.duplicated().sum()
print(f"Duplicates Found: {duplicates}")

df = df.drop_duplicates()

# ==========================
# Create Target
# ==========================
def create_decision(rating):

    if rating in ["AA", "A"]:
        return "Approved"

    elif rating in ["B", "C"]:
        return "Review"

    elif rating in ["D", "E", "HR"]:
        return "Denied"

    return None


df["decision"] = df["ProsperRating (Alpha)"].apply(
    create_decision
)

# Remove rows with no target
df = df.dropna(subset=["decision"])

# ==========================
# Remove Original Rating
# ==========================
df = df.drop(columns=["ProsperRating (Alpha)"])

# ==========================
# Remove ID Columns
# ==========================
drop_cols = [
    "ListingKey",
    "LoanKey",
    "MemberKey",
    "ListingNumber",
    "LoanNumber"
]

df = df.drop(columns=drop_cols, errors="ignore")

# ==========================
# Remove Columns With Too Many Missing Values
# ==========================
missing_percent = df.isnull().mean() * 100

cols_to_drop = missing_percent[
    missing_percent > 50
].index

print(
    f"Dropping {len(cols_to_drop)} columns with >50% missing values"
)

df = df.drop(columns=cols_to_drop)

# ==========================
# Fill Numeric Missing Values
# ==========================
numeric_cols = df.select_dtypes(
    include=["int64", "float64"]
).columns

for col in numeric_cols:
    df[col] = df[col].fillna(
        df[col].median()
    )

# ==========================
# Fill Categorical Missing Values
# ==========================
categorical_cols = df.select_dtypes(
    include=["object", "string"]
).columns

for col in categorical_cols:

    if col == "decision":
        continue

    mode_values = df[col].mode()

    if len(mode_values) > 0:
        df[col] = df[col].fillna(
            mode_values[0]
        )
    else:
        df[col] = df[col].fillna(
            "Unknown"
        )

date_cols = [
    "ListingCreationDate",
    "ClosedDate",
    "DateCreditPulled",
    "FirstRecordedCreditLine",
    "LoanOriginationDate"
]

df = df.drop(columns=date_cols, errors="ignore")

# Make sure target is last column
cols = [col for col in df.columns if col != "decision"]
cols.append("decision")

df = df[cols]

leakage_cols = [
    "ProsperRating (numeric)",
    "ProsperScore",
    "EstimatedEffectiveYield",
    "EstimatedLoss",
    "EstimatedReturn",
    "BorrowerAPR",
    "BorrowerRate",
    "LenderYield",
    "LP_CustomerPayments",
    "LP_CustomerPrincipalPayments",
    "LP_InterestandFees",
    "LP_ServiceFees",
    "LP_CollectionFees",
    "LP_GrossPrincipalLoss",
    "LP_NetPrincipalLoss",
    "LP_NonPrincipalRecoverypayments",
    "LoanStatus"
]

df = df.drop(
    columns=leakage_cols,
    errors="ignore"
)

# ==========================
# Save Clean Dataset
# ==========================
df.to_csv(
    "final_data.csv",
    index=False
)

# ==========================
# Summary
# ==========================
print(f"\nFinal Shape: {df.shape}")

print("\nDecision Class Distribution:")
print(df["decision"].value_counts())

print("\nDecision Class Percentage:")
print(
    df["decision"].value_counts(
        normalize=True
    ) * 100
)

print("\nSaved as final_data.csv")
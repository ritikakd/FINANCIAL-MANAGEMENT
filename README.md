# FINANCIAL-MANAGEMENT
CODE FOR FINANCIAL MANAGEMENT PROJECT


import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

# ------------------------------------------------------------
# 1. LOAD DATA
# ------------------------------------------------------------
file_path = r"C:\Users\LENOVO\Downloads\Financial_Management_Dataset.csv"
df = pd.read_csv(file_path)

print("\nOriginal Columns:")
print(df.columns.tolist())

df.columns = df.columns.str.lower().str.strip()


# ------------------------------------------------------------
# 2. AUTOMATIC COLUMN DETECTION
# ------------------------------------------------------------
def detect(col_keywords):
    """Return first matching column or None."""
    for col in df.columns:
        for key in col_keywords:
            if key in col:
                return col
    return None


possible_date      = detect(["date"])
possible_amount    = detect(["amount"])
possible_dept      = detect(["department", "dept"])
possible_cat       = detect(["category"])
possible_account   = detect(["account"])
possible_approver  = detect(["approved", "approver"])
possible_dc        = detect(["debit", "credit", "transaction type", "trans type", "type"])
possible_currency  = detect(["currency"])

print("\nDetected Columns:")
print("date:", possible_date)
print("amount:", possible_amount)
print("department:", possible_dept)
print("category:", possible_cat)
print("account:", possible_account)
print("approver:", possible_approver)
print("debit/credit:", possible_dc)
print("currency:", possible_currency)


# ------------------------------------------------------------
# 3. RENAME USING ONLY FOUND COLUMNS (prevents errors!)
# ------------------------------------------------------------
rename_map = {}

if possible_date: rename_map[possible_date] = "date"
if possible_amount: rename_map[possible_amount] = "amount"
if possible_dept: rename_map[possible_dept] = "department"
if possible_cat: rename_map[possible_cat] = "category"
if possible_account: rename_map[possible_account] = "account"
if possible_approver: rename_map[possible_approver] = "approver"
if possible_dc: rename_map[possible_dc] = "debit_credit"
if possible_currency: rename_map[possible_currency] = "currency"

df = df.rename(columns=rename_map)

print("\nRenamed Columns:")
print(df.columns.tolist())


# ------------------------------------------------------------
# 4. CLEAN & STANDARDIZE
# ------------------------------------------------------------
df["date"] = pd.to_datetime(df["date"], errors="coerce")

if "debit_credit" in df.columns:
    df["debit_credit"] = df["debit_credit"].astype(str).str.lower()
else:
    print("\n⚠ No debit/credit column detected. Some analyses will skip this.")

df["amount"] = pd.to_numeric(df["amount"], errors="coerce")

df = df.dropna(subset=["date", "amount"])

df["year_month"] = df["date"].dt.to_period("M")


# ------------------------------------------------------------
# Q1 — Monthly spending & income trends
# ------------------------------------------------------------
q1 = df.groupby(["year_month", "department"])["amount"].sum().reset_index()


# ------------------------------------------------------------
# Q2 — Departments overshooting their budgets
# (Assuming debits = spending)
# ------------------------------------------------------------
if "debit_credit" in df.columns:
    debits = df[df["debit_credit"] == "debit"]
    q2 = debits.groupby(["department", "category"])["amount"].sum().reset_index()
else:
    q2 = pd.DataFrame()


# ------------------------------------------------------------
# Q3 — Average monthly expenditure per account
# ------------------------------------------------------------
q3 = df.groupby(["year_month", "account"])["amount"].sum().groupby("account").mean().reset_index()


# ------------------------------------------------------------
# Q4 — Net cash flow by month/department/account
# ------------------------------------------------------------
if "debit_credit" in df.columns:
    df["signed_amount"] = np.where(df["debit_credit"] == "credit",
                                   df["amount"],
                                   -df["amount"])
else:
    df["signed_amount"] = df["amount"]  # fallback

q4 = df.groupby(["year_month", "department", "account"])["signed_amount"].sum().reset_index()


# ------------------------------------------------------------
# Q5 — Which categories contribute most to inflow/outflow
# ------------------------------------------------------------
q5 = df.groupby(["category"])["amount"].sum().reset_index()


# ------------------------------------------------------------
# Q6 — Detect large withdrawals/deposits (liquidity risk)
# ------------------------------------------------------------
threshold = df["amount"].mean() + 2 * df["amount"].std()
q6 = df[df["amount"] > threshold]


# ------------------------------------------------------------
# Q7 — Trends in expense categories
# ------------------------------------------------------------
q7 = df.groupby(["year_month", "category"])["amount"].sum().reset_index()


# ------------------------------------------------------------
# Q8 — Top approvers by volume & total spend
# ------------------------------------------------------------
if "approver" in df.columns:
    q8_volume = df["approver"].value_counts().reset_index()
    q8_amount = df.groupby("approver")["amount"].sum().reset_index()
else:
    q8_volume = pd.DataFrame()
    q8_amount = pd.DataFrame()


# ------------------------------------------------------------
# Q9 — % spent on non-operational
# ------------------------------------------------------------
non_ops = ["maintenance", "travel", "ad-hoc", "adhoc", "repairs"]

df["non_operational"] = df["category"].str.lower().isin(non_ops)

q9_percent = (df[df["non_operational"]]["amount"].sum() /
              df["amount"].sum()) * 100


# ------------------------------------------------------------
# Q10 — Departments with unusual frequencies or high amounts
# ------------------------------------------------------------
q10_freq = df["department"].value_counts().reset_index()
q10_high = df[df["amount"] > df["amount"].mean() + 2 * df["amount"].std()]


# ------------------------------------------------------------
# Q11 — Transactions >2 SD above mean
# ------------------------------------------------------------
q11 = df[df["amount"] > df["amount"].mean() + 2 * df["amount"].std()]


# ------------------------------------------------------------
# Q12 — Simple forecast using monthly totals (last 12 months)
# ------------------------------------------------------------
monthly_totals = df.groupby("year_month")["amount"].sum()
q12_forecast = monthly_totals.tail(12).mean()


# ------------------------------------------------------------
# Q13 — Credit vs debit over time
# ------------------------------------------------------------
if "debit_credit" in df.columns:
    q13 = df.groupby(["year_month", "debit_credit"])["amount"].sum().reset_index()
else:
    q13 = pd.DataFrame()


# ------------------------------------------------------------
# Q14 — Approvers approving high-amount transactions
# ------------------------------------------------------------
if "approver" in df.columns:
    high_thresh = df["amount"].mean() + df["amount"].std()
    q14 = df[df["amount"] > high_thresh].groupby(["approver", "department"])["amount"].sum().reset_index()
else:
    q14 = pd.DataFrame()


# ------------------------------------------------------------
# Q15 — Distribution of Credit/Debit by category
# ------------------------------------------------------------
if "debit_credit" in df.columns:
    q15 = df.groupby(["category", "debit_credit"])["amount"].sum().reset_index()
else:
    q15 = pd.DataFrame()

print("\nAll analyses complete.")
print("Non-operational spend %:", round(q9_percent, 2))

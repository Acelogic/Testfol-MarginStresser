
import pandas as pd
import numpy as np

# Simulate Data
dates = pd.date_range("2024-01-01", periods=1)
equity = 100.0
loan = 100.0
port = equity + loan # 200.0
maint = 0.25

# Create Series
port_series = pd.Series([port], index=dates)
loan_series = pd.Series([loan], index=dates)
final_adj_series = pd.Series([equity], index=dates) # Net Equity

# 1. Metric Formula (Simulate Margin)
# Usage = Loan / (Port * (1 - M))
metric_usage = loan / (port * (1 - maint))

# 2. Old Chart Formula (Buying Power)
# Usage = Loan / (Equity * (1-M)/M)
old_chart_usage = loan / (equity * (1 - maint) / maint)

# 3. New Chart Formula (Maintenance Usage)
# Max Loan = Port * (1 - M)
# Usage = Loan / Max Loan
max_loan_series = port_series * (1 - maint)
new_chart_usage = loan_series / max_loan_series

print(f"Scenario: Equity={equity}, Loan={loan}, Port={port}, Maint={maint}")
print(f"Metric Usage (Target): {metric_usage:.4f}")
print(f"Old Chart Usage:       {old_chart_usage:.4f}")
print(f"New Chart Usage:       {new_chart_usage.iloc[0]:.4f}")

if abs(metric_usage - new_chart_usage.iloc[0]) < 0.0001:
    print("SUCCESS: New Chart Formula matches Metric Formula.")
else:
    print("FAILURE: Mismatch!")

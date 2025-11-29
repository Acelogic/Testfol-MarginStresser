
import pandas as pd
import numpy as np

def calculate_cagr(series):
    if series.empty: return 0.0
    start_val = series.iloc[0]
    end_val = series.iloc[-1]
    years = (series.index[-1] - series.index[0]).days / 365.25
    if years <= 0: return 0.0
    cagr = ((end_val / start_val) ** (1 / years) - 1) * 100
    return cagr

# Simulation of the Fix Logic
# Scenario: Pay Down Margin
# Assets grow at 10% (no additions).
# Loan decreases by $1000/yr (plus interest, simplified here to just principal reduction for clarity).
# Net Equity = Assets - Loan.

dates = pd.date_range(start="2020-01-01", periods=5, freq="YE")

# Assets (Portfolio Value) - Same as before (lower growth)
vals_assets = [10000, 11000, 12100, 13310, 14641]
series_assets = pd.Series(vals_assets, index=dates)

# Loan (Starting at $5000, paying down $1000/yr)
# Year 0: 5000
# Year 1: 4000
# Year 2: 3000
# Year 3: 2000
# Year 4: 1000
vals_loan = [5000, 4000, 3000, 2000, 1000]
series_loan = pd.Series(vals_loan, index=dates)

# Net Equity
series_equity = series_assets - series_loan
# Year 0: 10000 - 5000 = 5000
# Year 1: 11000 - 4000 = 7000
# Year 2: 12100 - 3000 = 9100
# Year 3: 13310 - 2000 = 11310
# Year 4: 14641 - 1000 = 13641

# Calculate CAGR on Assets (Old Behavior)
cagr_assets = calculate_cagr(series_assets)
print(f"Old Metric (Assets): Start=${vals_assets[0]}, End=${vals_assets[-1]}, CAGR={cagr_assets:.2f}%")

# Calculate CAGR on Equity (New Behavior)
cagr_equity = calculate_cagr(series_equity)
print(f"New Metric (Equity): Start=${series_equity.iloc[0]}, End=${series_equity.iloc[-1]}, CAGR={cagr_equity:.2f}%")

# Expected: Equity CAGR should be significantly higher because it captures the paydown "return" (de-leveraging gain).
# 5000 -> 13641 over 4 years is ~28% CAGR!


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

# Scenario 1: Standard DCA (Cashflow buys Assets)
# Start: $10,000. Return 10%/yr. Cashflow $1,000/yr.
dates = pd.date_range(start="2020-01-01", periods=5, freq="YE")
# Year 0: 10000
# Year 1: 10000*1.1 + 1000 = 12000
# Year 2: 12000*1.1 + 1000 = 14200
# Year 3: 14200*1.1 + 1000 = 16620
# Year 4: 16620*1.1 + 1000 = 19282
vals_dca = [10000, 12000, 14200, 16620, 19282]
series_dca = pd.Series(vals_dca, index=dates)

cagr_dca = calculate_cagr(series_dca)
print(f"Scenario 1 (DCA to Assets): End Val=${vals_dca[-1]}, CAGR={cagr_dca:.2f}%")

# Scenario 2: Pay Down Margin (Cashflow reduces Loan, Assets grow purely by return)
# Start: $10,000. Return 10%/yr. Cashflow $0 to Assets (goes to loan).
# Year 0: 10000
# Year 1: 10000*1.1 = 11000
# Year 2: 11000*1.1 = 12100
# Year 3: 12100*1.1 = 13310
# Year 4: 13310*1.1 = 14641
vals_paydown = [10000, 11000, 12100, 13310, 14641]
series_paydown = pd.Series(vals_paydown, index=dates)

cagr_paydown = calculate_cagr(series_paydown)
print(f"Scenario 2 (Pay Down Margin - Assets Only): End Val=${vals_paydown[-1]}, CAGR={cagr_paydown:.2f}%")

# Expected: CAGR_DCA > CAGR_Paydown because End Val is higher due to contributions.
# But true "Performance" (TWR) is 10% for both.
# The naive CAGR calc conflates contributions with growth.

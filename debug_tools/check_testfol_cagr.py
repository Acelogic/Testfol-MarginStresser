
import testfol_api as api
import pandas as pd
import numpy as np
from datetime import date

def calculate_naive_cagr(series):
    if series.empty: return 0.0
    start_val = series.iloc[0]
    end_val = series.iloc[-1]
    years = (series.index[-1] - series.index[0]).days / 365.25
    if years <= 0: return 0.0
    cagr = ((end_val / start_val) ** (1 / years) - 1) * 100
    return cagr

# Parameters
start_val = 10000
cashflow = 1000 # Monthly
start_date = date(2020, 1, 1)
end_date = date(2023, 1, 1)
allocation = {"SPY": 100}

print("Fetching backtest from Testfol API...")
try:
    port_series, stats, extra = api.fetch_backtest(
        start_date=str(start_date),
        end_date=str(end_date),
        start_val=start_val,
        cashflow=cashflow,
        cashfreq="Monthly",
        rolling=60,
        invest_div=True,
        rebalance="Yearly",
        allocation=allocation
    )
    
    api_cagr = stats.get('cagr')
    local_naive_cagr = calculate_naive_cagr(port_series)
    
    print(f"API Reported CAGR: {api_cagr:.2f}%")
    print(f"Local Naive CAGR:  {local_naive_cagr:.2f}%")
    
    if abs(api_cagr - local_naive_cagr) < 0.1:
        print("CONCLUSION: Testfol uses the Naive (Inflated) CAGR calculation.")
    else:
        print("CONCLUSION: Testfol uses a different method (likely TWR/MWR) that differs from Naive CAGR.")
        
except Exception as e:
    print(f"Error: {e}")

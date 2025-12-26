
import sys
import os
import pandas as pd
from datetime import datetime

# Ensure project root is in path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.services.testfol_api import fetch_backtest

# --- Configuration ---
TICKER = "QQQSIM?L=2"
START_DATE = datetime(2000, 7, 3)
END_DATE = datetime(2025, 12, 24)
START_VAL = 20000.0
CASHFLOW = 0.0

print(f"--- Verifying Testfol API for {TICKER} ---")
print(f"Start Date: {START_DATE.date()}")
print(f"Start Val: ${START_VAL:,.2f}")

try:
    port_series, stats, _ = fetch_backtest(
        start_date=START_DATE,
        end_date=END_DATE,
        start_val=START_VAL,
        cashflow=CASHFLOW,
        cashfreq="Monthly",
        rolling=60,
        invest_div=True,
        rebalance="Yearly",
        allocation={TICKER: 100},
        return_raw=False
    )
    
    print("\n--- API Returned Stats ---")
    print(f"CAGR: {stats.get('cagr', 0)*100:.2f}%")
    print(f"Max Drawdown: {stats.get('max_drawdown', 0):.2f}%")
    print(f"Sharpe: {stats.get('sharpe', 0):.2f}")
    
    if not port_series.empty:
        print(f"\nFirst Date: {port_series.index[0].date()}")
        print(f"First Value: ${port_series.iloc[0]:,.2f}")
        print(f"Last Date: {port_series.index[-1].date()}")
        print(f"Last Value: ${port_series.iloc[-1]:,.2f}")
    
except Exception as e:
    print(f"API Error: {e}")

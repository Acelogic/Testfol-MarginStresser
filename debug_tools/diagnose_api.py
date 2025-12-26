
import sys
import os
import pandas as pd
from datetime import datetime
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from app.services.testfol_api import fetch_backtest

TICKER = "QQQSIM?L=2"
START_DATE = datetime(2000, 7, 3)
END_DATE = datetime(2025, 12, 24)
START_VAL = 20000.0

scenarios = [
    {"name": "Baseline (Script)", "div": True, "offset": 0},
    {"name": "No Dividends", "div": False, "offset": 0},
    {"name": "Offset 1 Day", "div": True, "offset": 1},
    {"name": "Offset 100 Days", "div": True, "offset": 100},
    {"name": "Start June 30", "div": True, "offset": 0, "start_dt": datetime(2000, 6, 30)},
]

print(f"--- Diagnosing QQQSIM Discrepancy ---")
print(f"Goal: Find scenario ending near $82,007 (vs $101,895)")

for s in scenarios:
    s_date = s.get("start_dt", START_DATE)
    try:
        series, stats, _ = fetch_backtest(
            start_date=s_date,
            end_date=END_DATE,
            start_val=START_VAL,
            cashflow=0,
            cashfreq="Monthly",
            rolling=60,
            invest_div=s["div"],
            rebalance="Yearly",
            rebalance_offset=s["offset"],
            allocation={TICKER: 100},
            return_raw=False
        )
        end_val = series.iloc[-1] if not series.empty else 0
        print(f"Scenario '{s['name']}': End Val=${end_val:,.2f}")
    except Exception as e:
        print(f"Scenario '{s['name']}' Failed: {e}")

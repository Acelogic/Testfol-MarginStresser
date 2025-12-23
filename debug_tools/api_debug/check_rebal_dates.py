import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
import app.services.testfol_api as api
import pandas as pd
import json

def check_dates(offset_val, label):
    print(f"\n--- Checking Offset: {offset_val} ({label}) ---")
    try:
        # Use a short recent period to see daily dates clearly
        resp = api.fetch_backtest(
            start_date="2020-01-01",
            end_date="2023-12-31",
            start_val=10000,
            cashflow=0,
            cashfreq="None",
            rolling=60,
            invest_div=True,
            rebalance="Yearly",
            allocation={"SPY": 50, "TLT": 50},
            return_raw=False,
            include_raw=True, # Need this to get rebalancing_events
            rebalance_offset=offset_val
        )
        
        # Unpack result
        _, _, extra = resp
        events = extra.get('rebalancing_events', [])
        
        if not events:
            print("No rebalancing events found.")
            return

        # Structure is list of portfolios -> 'events' key -> list of [Date, ...]
        portfolio_events = events[0].get('events', [])
        
        print(f"Total Events: {len(portfolio_events)}")
        # Print first 3 dates
        for i, e in enumerate(portfolio_events[:3]):
            print(f"Event {i+1}: {e[0]}")
            
    except Exception as e:
        print(f"Error: {e}")

# 1. Default (Offset 0) - Expect Dec 31-ish
check_dates(0, "Default")

# 2. Small Offset (Offset 10) - Expect ~2 weeks before Dec 31
check_dates(10, "Small Offset (10)")

# 3. Half Year (Offset 126) - Expect ~June
check_dates(126, "Half Year (126)")

# 4. Large Offset (Offset 200) - Expect ~Feb/March
check_dates(200, "Large Offset (200)")

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
import app.services.testfol_api as api
import pandas as pd
import requests
import json
import time

# Helper to send raw request
def send_payload(extra_params={}, backtest_params={}):
    url = "https://testfol.io/api/backtest"
    
    payload = {
        "start_date": "2010-01-01",
        "end_date": "2020-01-01",
        "start_val": 10000,
        "adj_inflation": False,
        "cashflow": 0,
        "cashflow_freq": "None",
        "rolling_window": 60,
        "backtests": [{
            "invest_dividends": True,
            "rebalance_freq": "Yearly",
            "allocation": {"SPY": 50, "TLT": 50},
            "drag": 0,
            "absolute_dev": 0,
            "relative_dev": 0
        }]
    }
    
    # Merge extras
    payload.update(extra_params)
    payload["backtests"][0].update(backtest_params)
    
    try:
        r = requests.post(url, json=payload, timeout=30)
        time.sleep(1.5) # Courtesy delay
        if r.status_code == 200:
            resp = r.json()
            # Get final value
            charts = resp.get("charts", {}).get("history", [])
            if charts:
                final_val = charts[1][-1]
                return final_val
    except Exception as e:
        print(f"Error: {e}")
        
    return None

print("--- Testing API Offset Parameters ---")

# 1. Baseline
base_val = send_payload()
print(f"Baseline (Yearly, Jan 1): {base_val}")

# 2. Test rebalance_offset (assume it takes integer days, e.g. 30 for ~1 month)
offset_val = send_payload(backtest_params={"rebalance_offset": 60})
print(f"With rebalance_offset=60:  {offset_val}")

if base_val != offset_val and offset_val is not None:
    print(">> MATCH! 'rebalance_offset' is working!")
else:
    print(">> No effect.")

# 3. Test backtest_offset?
offset_val2 = send_payload(extra_params={"rebalance_offset": 60}) # Try at top level?
print(f"With top-level rebal_offset: {offset_val2}")

# 4. Test cashflow offset
base_cf = send_payload(extra_params={"cashflow": 100, "cashflow_freq": "Monthly"})
print(f"Baseline (Cashflow):      {base_cf}")

cf_offset = send_payload(extra_params={"cashflow": 100, "cashflow_freq": "Monthly", "cashflow_offset": 15})
print(f"With cashflow_offset=15:  {cf_offset}")

if base_cf != cf_offset and cf_offset is not None:
    print(">> MATCH! 'cashflow_offset' is working!")
else:
    print(">> No effect.")

# 5. Test 'offset' generic
generic_offset = send_payload(backtest_params={"offset": 60})
print(f"With 'offset'=60:         {generic_offset}")

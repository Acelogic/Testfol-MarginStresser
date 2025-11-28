import sys
sys.path.append("..")
import testfol_api as api
import pandas as pd
import datetime as dt
import json

# Configuration
start_date = dt.date(2012, 1, 1)
end_date = dt.date.today()
start_val = 10000.0
cashflow = 0.0
cashfreq = "Monthly"
invest_div = True
rebalance = "Yearly"

alloc_list = [
    {"Ticker":"AAPL?L=2","Weight %":7.5,"Maint %":50},
    {"Ticker":"MSFT?L=2","Weight %":7.5,"Maint %":50},
    {"Ticker":"AVGO?L=2","Weight %":7.5,"Maint %":50},
    {"Ticker":"AMZN?L=2","Weight %":7.5,"Maint %":50},
    {"Ticker":"META?L=2","Weight %":7.5,"Maint %":50},
    {"Ticker":"NVDA?L=2","Weight %":7.5,"Maint %":50},
    {"Ticker":"GOOGL?L=2","Weight %":7.5,"Maint %":50},
    {"Ticker":"TSLA?L=2","Weight %":7.5,"Maint %":50},
    {"Ticker":"GLD","Weight %":20,"Maint %":25},
    {"Ticker":"VXUS","Weight %":15,"Maint %":25},
    {"Ticker":"TQQQ","Weight %":5,"Maint %":75},
]

alloc_preview = {item["Ticker"]: item["Weight %"] for item in alloc_list}

print("Fetching backtest...")
try:
    # Pass return_raw=True to get the full JSON
    resp = api.fetch_backtest(
        start_date, end_date, start_val,
        cashflow, cashfreq, 60,
        invest_div, rebalance, alloc_preview,
        return_raw=True
    )
    
    print("Response Keys:", list(resp.keys()))
    
    if "assets" in resp:
        print("\n'assets' key found. Inspecting first item...")
        assets = resp["assets"]
        if isinstance(assets, list) and len(assets) > 0:
            print(assets[0])
        elif isinstance(assets, dict):
            print(list(assets.keys()))
            
    if "charts" in resp:
        print("\n'charts' keys:", list(resp["charts"].keys()))
        
    # Check for any other lists that might contain time series
    for k, v in resp.items():
        if isinstance(v, list) and len(v) > 0:
            print(f"\nKey '{k}' is a list of length {len(v)}")
            if isinstance(v[0], dict):
                print(f"  Sample item keys: {list(v[0].keys())}")
            else:
                print(f"  Sample item: {v[0]}")

except Exception as e:
    print(f"Error: {e}")

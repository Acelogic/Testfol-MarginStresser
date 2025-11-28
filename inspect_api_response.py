import testfol_api as api
import datetime as dt
import json

start_date = dt.date(2022, 1, 1)
end_date = dt.date(2023, 1, 1)
start_val = 10000
cashflow = 0
cashfreq = "Yearly"
rolling = 60
invest_div = True
rebalance = "Quarterly"
allocation = {"AAPL": 50, "MSFT": 50}

print("Fetching backtest data...")
try:
    resp = api.fetch_backtest(
        start_date, end_date, start_val,
        cashflow, cashfreq, rolling,
        invest_div, rebalance, allocation,
        return_raw=True
    )
    
    with open("api_dump.json", "w") as f:
        json.dump(resp, f, indent=2)
    print("Dumped to api_dump.json")

except Exception as e:
    print(f"Error: {e}")

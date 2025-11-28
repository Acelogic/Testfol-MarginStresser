import sys
import os
import pandas as pd
import datetime as dt

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

import testfol_api as api

def check_stats():
    print("Fetching backtest stats...")
    
    # Dummy params
    start_date = dt.date(2020, 1, 1)
    end_date = dt.date(2023, 1, 1)
    start_val = 10000
    cashflow = 0
    cashfreq = "Yearly"
    rolling = 60
    invest_div = True
    rebalance = "Yearly"
    allocation = {"SPY": 100}
    
    try:
        port_series, stats, extra = api.fetch_backtest(
            start_date, end_date, start_val, cashflow, cashfreq, rolling,
            invest_div, rebalance, allocation
        )
        
        print("\nStats Object:")
        print(stats)
        
        cagr = stats.get("cagr")
        print(f"\nCAGR Value: {cagr}")
        
        if cagr > 1:
            print("CAGR appears to be a Percentage (e.g. 10.5 for 10.5%)")
        else:
            print("CAGR appears to be a Decimal (e.g. 0.105 for 10.5%)")
            
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    check_stats()

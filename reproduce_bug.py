import pandas as pd
import numpy as np
from datetime import date
from app.core.shadow_backtest import run_shadow_backtest

def reproduce():
    # 1. Create synthetic data (2000 to 2024)
    dates = pd.date_range(start="2000-01-01", end="2024-01-01", freq="B")
    data = np.cumprod(1 + np.random.normal(0, 0.01, size=(len(dates), 2)), axis=0)
    prices_df = pd.DataFrame(data, index=dates, columns=["SPY", "TLT"])
    
    # 2. Define Backtest Params
    allocation = {"SPY": 60, "TLT": 40}
    start_date = date(2020, 1, 1) # User wants to start in 2020
    
    print(f"Running backtest with requested start_date: {start_date}")
    print(f"Data available from: {prices_df.index[0].date()} to {prices_df.index[-1].date()}")
    
    # 3. Run Backtest
    trades, pl, comp, unrealized, logs, port_series = run_shadow_backtest(
        allocation=allocation,
        start_val=10000,
        start_date=start_date,
        end_date=date(2024, 1, 1),
        prices_df=prices_df,
        rebalance_freq="Yearly"
    )
    
    # 4. Verify Start Date
    if port_series.empty:
        print("FAIL: Portfolio series is empty.")
        return

    actual_start = port_series.index[0].date()
    print(f"Actual simulation start: {actual_start}")
    
    if actual_start.year == 2000:
        print("FAIL: Simulation started at beginning of data (2000) instead of requested 2020.")
    elif actual_start.year >= 2020:
        print("PASS: Simulation respected the start date.")
    else:
        print(f"FAIL: Simulation started at unexpected date: {actual_start}")

if __name__ == "__main__":
    reproduce()

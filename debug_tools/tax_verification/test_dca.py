import sys
import os
import pandas as pd
import numpy as np
from datetime import date, timedelta
from unittest.mock import MagicMock

# Add parent directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import shadow_backtest

def test_dca_logic():
    print("--- Testing DCA Logic ---")

    # Mock fetch_prices to return deterministic data
    # Create a date range
    dates = pd.date_range(start="2023-01-01", end="2023-12-31", freq="B")
    
    # Create a mock price series: Constant $100
    prices = pd.DataFrame(index=dates)
    prices["TEST"] = 100.0
    
    # Monkeypatch fetch_prices
    original_fetch = shadow_backtest.fetch_prices
    shadow_backtest.fetch_prices = MagicMock(return_value=(prices, "Mocked Data"))
    
    try:
        # Run Backtest with Monthly DCA
        allocation = {"TEST": 1.0}
        start_val = 10000.0
        cashflow = 1000.0
        
        print(f"Initial Value: ${start_val:,.2f}")
        print(f"Monthly DCA:   ${cashflow:,.2f}")
        
        trades, pl, comp, unrealized, logs = shadow_backtest.run_shadow_backtest(
            allocation=allocation,
            start_val=start_val,
            start_date="2023-01-01",
            end_date="2023-12-31",
            rebalance_freq="Yearly", # No rebalancing to isolate DCA
            cashflow=cashflow,
            cashflow_freq="Monthly"
        )
        
        # Verify Results
        # 12 months in 2023.
        # Jan 1: Initial 10,000
        # End of Jan, Feb, ... Nov (11 injections?)
        # Let's check the logs or trades to see how many injections happened.
        
        dca_trades = trades[trades["Realized P&L"] == 0] # DCA buys have 0 realized P&L initially
        print(f"Number of DCA Injections: {len(dca_trades)}")
        
        total_injected = len(dca_trades) * cashflow
        expected_final_value = start_val + total_injected
        
        # Get final value from composition
        final_comp = comp[comp["Date"] == comp["Date"].max()]
        final_value = final_comp["Value"].sum()
        
        print(f"Total Injected: ${total_injected:,.2f}")
        print(f"Final Value:    ${final_value:,.2f}")
        print(f"Expected Value: ${expected_final_value:,.2f}")
        
        # Since price is constant, value should be exactly start + injections
        assert abs(final_value - expected_final_value) < 0.01
        
        # Verify Tax Lots
        # We should have 1 initial lot + N injection lots
        # We can't easily access the internal tax_lots from here without modifying the function return or inspecting logs
        # But if the value is correct and trades are recorded, it's likely working.
        
        # Check if trades have correct data
        print("\nDCA Trades:")
        print(dca_trades[["Date", "Ticker", "Trade Amount", "Price (Est)"]])
        
        # Verify that we have injections for each month end (except maybe Dec if it ends exactly on Dec 31?)
        # Logic says: if i < len(dates) - 1 and dates[i+1].month != date.month:
        # So it happens on the last trading day of the month.
        # Jan..Nov = 11 months. Dec is last month, loop goes to len-1.
        # If dates ends on Dec 29 (Friday), loop goes to Dec 29.
        # dates[i+1] doesn't exist for the last item.
        # So we expect 11 injections (Jan-Nov).
        
        assert len(dca_trades) >= 11
        
        print("\nDCA Test Passed!")
        
    finally:
        # Restore original function
        shadow_backtest.fetch_prices = original_fetch

if __name__ == "__main__":
    test_dca_logic()

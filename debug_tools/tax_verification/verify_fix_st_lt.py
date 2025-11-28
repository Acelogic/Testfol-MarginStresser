import sys
import os
import pandas as pd
import numpy as np

# Add parent directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import shadow_backtest
import tax_library

def verify_st_lt_generation():
    print("--- Verifying Short-Term vs Long-Term Gain Generation ---")
    
    # Setup a scenario that SHOULD generate Short-Term gains:
    # - Volatile asset (TQQQ) mixed with stable asset (BIL)
    # - Monthly rebalancing
    # - 1 Year duration
    
    allocation = {
        "TQQQ": 50.0,
        "BIL": 50.0
    }
    
    start_val = 100000.0
    start_date = "2021-01-01"
    end_date = "2021-12-31"
    
    print(f"Running Shadow Backtest (Monthly Rebalancing)...")
    print(f"Allocation: {allocation}")
    
    trades_df, pl_by_year, composition_df, unrealized_pl_df, logs = shadow_backtest.run_shadow_backtest(
        allocation,
        start_val,
        start_date,
        end_date,
        rebalance_freq="Monthly"
    )
    
    if pl_by_year.empty:
        print("❌ No trades generated! Cannot verify.")
        return
        
    print("\n--- P&L Results by Year ---")
    print(pl_by_year)
    
    # Check for ST Gains
    total_st_gain = pl_by_year["Realized ST P&L"].sum()
    total_lt_gain = pl_by_year["Realized LT P&L"].sum()
    
    print(f"\nTotal Short-Term P&L: ${total_st_gain:,.2f}")
    print(f"Total Long-Term P&L:  ${total_lt_gain:,.2f}")
    
    if total_st_gain != 0:
        print("\n✅ SUCCESS: Short-Term gains/losses were generated!")
        print("This confirms that the system now distinguishes between ST and LT holdings.")
    else:
        print("\n❌ FAILURE: No Short-Term gains generated despite monthly rebalancing.")
        
    # Verify Tax Calculation on these gains
    print("\n--- Verifying Tax Calculation ---")
    # Assume we have some ST gain
    if total_st_gain > 0:
        # Calculate tax
        tax = tax_library.calculate_tax_on_realized_gains(
            short_term_gain=total_st_gain,
            long_term_gain=0,
            other_income=100000,
            year=2021,
            filing_status="Single"
        )
        # ST Gain should be taxed at marginal rate (24% for 100k+ in 2021)
        # 24% of gain
        expected_tax_approx = total_st_gain * 0.24
        print(f"Tax on ${total_st_gain:,.2f} ST Gain (with $100k income): ${tax:,.2f}")
        print(f"Expected approx (~24%): ${expected_tax_approx:,.2f}")
        
        if tax > 0:
             print("✅ Tax calculated on ST gain.")
        else:
             print("❌ No tax calculated on ST gain.")

if __name__ == "__main__":
    verify_st_lt_generation()

#!/usr/bin/env python3
"""
debug_tools/trace_cagr_calculation.py

This script compares local leveraged simulation vs direct Testfol API results.
It uses the ACTUAL logic from shadow_backtest.py by calling the entry point.
"""
import sys
sys.path.insert(0, '/Users/mcruz/Developer/Testfol-MarginStresser')

import pandas as pd
from app.core import calculations, shadow_backtest
from app.services import testfol_api as api

# Match user's parameters
START_DATE = "2000-09-15"
END_DATE = "2025-12-23"
START_VAL = 10000.0
CASHFLOW = 1000.0

# Allocations
ALLOC_QQQ = {"QQQ": 100}
ALLOC_LEV = {"QQQSIM?L=2": 100}

def main():
    print("=" * 60)
    print("CAGR Calculation Trace: QQQ vs QQQSIM?L=2")
    print("=" * 60)
    
    # 1. Fetch Standard QQQ
    print("\n1️⃣  Fetching Standard QQQ (API)...")
    try:
        r_port, r_stats, _ = api.fetch_backtest(
            start_date=START_DATE, end_date=END_DATE, start_val=START_VAL,
            cashflow=CASHFLOW, cashfreq="Monthly", rolling=60, invest_div=True,
            rebalance="Yearly", allocation=ALLOC_QQQ, include_raw=True
        )
    except Exception as e:
        print(f"❌ API Error (QQQ): {e}")
        return
    
    # 2. Fetch QQQSIM?L=2 Directly
    print("\n2️⃣  Fetching QQQSIM?L=2 (API)...")
    try:
        l_port, l_stats, _ = api.fetch_backtest(
            start_date=START_DATE, end_date=END_DATE, start_val=START_VAL,
            cashflow=CASHFLOW, cashfreq="Monthly", rolling=60, invest_div=True,
            rebalance="Yearly", allocation=ALLOC_LEV, include_raw=True
        )
    except Exception as e:
        print(f"❌ API Error (LEV): {e}")
        return
    
    # 3. Run Shadow Backtest on the LEVERAGED ticker to see our local result
    # We call run_shadow_backtest which now handles leverage logic at the asset level.
    print("\n3️⃣  Running Local Shadow Backtest for QQQSIM?L=2...")
    try:
        # We need the TWR series to calculate CAGR correctly
        _, _, _, _, _, _, l_twr_series = shadow_backtest.run_shadow_backtest(
            allocation=ALLOC_LEV,
            start_val=START_VAL,
            start_date=START_DATE,
            end_date=END_DATE,
            cashflow=CASHFLOW,
            cashflow_freq="Monthly",
            rebalance_freq="Yearly"
        )
    except Exception as e:
        print(f"❌ Local Shadow Backtest Error: {e}")
        import traceback
        st.code(traceback.format_exc())
        return
        
    local_lev_stats = calculations.generate_stats(l_twr_series)

    # 4. Results
    print("\n" + "=" * 60)
    print("📊 COMPARISON SUMMARY")
    print("=" * 60)
    print(f"Standard QQQ CAGR (API):       {r_stats.get('cagr', 0.0):.2f}%")
    print(f"Leveraged API CAGR (?L=2):      {l_stats.get('cagr', 0.0):.2f}%")
    print(f"Local Shadow CAGR (TWR-based): {local_lev_stats.get('cagr', 0.0):.2f}%")
    print("=" * 60)
    
    diff = abs(l_stats.get('cagr', 0.0) - local_lev_stats.get('cagr', 0.0))
    if diff < 1.0:
        print("✅ SUCCESS: Local simulation matches API within 1%!")
    else:
        print(f"❌ DISCREPANCY: Difference of {diff:.2f}% remains.")

if __name__ == "__main__":
    main()

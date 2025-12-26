#!/usr/bin/env python3
"""
debug_tools/test_qld_cagr.py

Test CAGR for QLD (2x leveraged QQQ ETF)
"""
import requests
import pandas as pd
import json

API_URL = "https://testfol.io/api/backtest"

# QLD inception: 2006-06-21, but we'll test from 2000-09-15 to match chart
START_DATE = "2000-01-01"
END_DATE = "2025-12-23"
START_VAL = 10000.0
CASHFLOW = 1000.0

# Allocation: QQQSIM?L=2
ALLOCATION = {"QQQSIM?L=2": 100}

def main():
    print("=" * 60)
    print("Testfol API - QLD (2x QQQ) CAGR Verification")
    print("=" * 60)
    
    start_str = pd.Timestamp(START_DATE).strftime('%Y-%m-%d')
    end_str = pd.Timestamp(END_DATE).strftime('%Y-%m-%d')
    
    payload = {
        "start_date": start_str,
        "end_date": end_str,
        "start_val": START_VAL,
        "adj_inflation": False,
        "cashflow": CASHFLOW,
        "cashflow_freq": "Monthly",
        "cashflow_offset": 0,
        "rolling_window": 60,
        "backtests": [{
            "invest_dividends": True,
            "rebalance_freq": "Yearly",
            "rebalance_offset": 0,
            "allocation": ALLOCATION,
            "drag": 0,
            "absolute_dev": 0,
            "relative_dev": 0
        }]
    }
    
    print(f"\n📤 Request: QLD 100%, {START_DATE} to {END_DATE}")
    
    try:
        r = requests.post(API_URL, json=payload, timeout=30)
        r.raise_for_status()
        resp = r.json()
    except Exception as e:
        print(f"❌ API Error: {e}")
        return
    
    stats = resp.get("stats", {})
    if isinstance(stats, list):
        stats = stats[0] if stats else {}
    
    print("\n" + "=" * 60)
    print("📊 QLD STATS FROM API")
    print("=" * 60)
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    print(f"\n📈 CAGR: {stats.get('cagr', 'N/A'):.2f}%")
    print(f"📉 Max Drawdown: {stats.get('max_drawdown', 'N/A'):.2f}%")
    print("=" * 60)

if __name__ == "__main__":
    main()

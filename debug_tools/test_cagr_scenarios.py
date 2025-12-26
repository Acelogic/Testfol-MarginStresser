#!/usr/bin/env python3
"""
debug_tools/test_cagr_scenarios.py

Tests 4 scenarios to solve the "4.7% vs 5.56% vs 7.18%" mystery.
"""
import requests
import pandas as pd
import json

API_URL = "https://testfol.io/api/backtest"
END_DATE = "2025-12-23"
START_VAL = 10000.0
CASHFLOW = 1000.0

SCENARIOS = [
    {"ticker": "QQQ?L=2", "start": "2000-01-01", "desc": "Real QQQ (L=2) @ Jan 1"},
    {"ticker": "QQQ?L=2", "start": "2000-09-15", "desc": "Real QQQ (L=2) @ Sep 15"},
    {"ticker": "QQQSIM?L=2", "start": "2000-01-01", "desc": "Sim QQQ (L=2) @ Jan 1"},
    {"ticker": "QQQSIM?L=2", "start": "2000-09-15", "desc": "Sim QQQ (L=2) @ Sep 15"},
]

def run_test(scenario):
    ticker = scenario["ticker"]
    start_date = scenario["start"]
    
    payload = {
        "start_date": start_date,
        "end_date": END_DATE,
        "start_val": START_VAL,
        "adj_inflation": False,
        "cashflow": CASHFLOW,
        "cashflow_freq": "Monthly",
        "cashflow_offset": 0,
        "rolling_window": 60,
        "backtests": [{
            "invest_dividends": True,
            "rebalance_freq": "Yearly",
            "allocation": {ticker: 100},
            "drag": 0,
            "absolute_dev": 0,
            "relative_dev": 0
        }]
    }
    
    try:
        r = requests.post(API_URL, json=payload, timeout=30)
        r.raise_for_status()
        stats = r.json().get("stats", {})
        if isinstance(stats, list): stats = stats[0]
        
        cagr = stats.get("cagr", 0.0)
        end_bal = stats.get("end_val", 0.0)
        
        print(f"✅ {scenario['desc']:<25} | CAGR: {cagr:6.2f}% | End Bal: ${end_bal:,.0f}")
        
    except Exception as e:
        print(f"❌ {scenario['desc']:<25} | Error: {e}")

def main():
    print("="*65)
    print(f"{'SCENARIO':<27} | {'CAGR':<9} | {'END BALANCE'}")
    print("="*65)
    
    for s in SCENARIOS:
        run_test(s)
    print("="*65)

if __name__ == "__main__":
    main()

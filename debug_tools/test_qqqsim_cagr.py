#!/usr/bin/env python3
"""
debug_tools/test_qqqsim_cagr.py

This script makes a direct call to the Testfol API to verify the CAGR
for a QQQ or similar portfolio with specific parameters.
"""
import requests
import pandas as pd
import json

# --- Configuration ---
API_URL = "https://testfol.io/api/backtest"

# Match user's parameters (from the chart)
START_DATE = "2000-09-15"
END_DATE = "2025-12-23"
START_VAL = 10000.0
CASHFLOW = 1000.0  # Monthly DCA
CASHFREQ = "Monthly"

# Allocation: QQQSIM?L=2 (Leveraged Simulation)
ALLOCATION = {"QQQSIM?L=2": 100}

def main():
    print("=" * 60)
    print("Testfol API Direct Call - CAGR Verification")
    print("=" * 60)
    
    # Format dates exactly like testfol_api.py
    start_str = pd.Timestamp(START_DATE).strftime('%Y-%m-%d')
    end_str = pd.Timestamp(END_DATE).strftime('%Y-%m-%d')
    
    # Build Payload (matching testfol_api.py exactly)
    payload = {
        "start_date": start_str,
        "end_date": end_str,
        "start_val": START_VAL,
        "adj_inflation": False,
        "cashflow": CASHFLOW,
        "cashflow_freq": CASHFREQ,
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
    
    print(f"\n📤 Request Payload:")
    print(json.dumps(payload, indent=2))
    
    # Make API Call
    print(f"\n⏳ Calling Testfol API...")
    try:
        r = requests.post(API_URL, json=payload, timeout=30)
        r.raise_for_status()
        resp = r.json()
    except Exception as e:
        print(f"❌ API Error: {e}")
        return
    
    # Extract Stats (matching testfol_api.py exactly)
    stats = resp.get("stats", {})
    if isinstance(stats, list):
        stats = stats[0] if stats else {}
    
    # Extract Chart Data
    charts = resp.get("charts", {})
    history = charts.get("history", [[], []])
    ts, vals = history
    
    if not ts or not vals:
        print("❌ No chart data returned!")
        return
    
    dates = pd.to_datetime(ts, unit="s")
    series = pd.Series(vals, index=dates, name="Portfolio")
    
    # --- Print ALL Stats from API ---
    print("\n" + "=" * 60)
    print("📊 RAW STATS FROM API (stats object)")
    print("=" * 60)
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    print("\n" + "=" * 60)
    print("📈 CHART DATA SUMMARY")
    print("=" * 60)
    print(f"Start Date:     {series.index[0].date()}")
    print(f"End Date:       {series.index[-1].date()}")
    print(f"Start Value:    ${series.iloc[0]:,.2f}")
    print(f"End Value:      ${series.iloc[-1]:,.2f}")
    
    # Calculate years
    years = (series.index[-1] - series.index[0]).days / 365.25
    print(f"Years:          {years:.2f}")
    
    # API CAGR
    api_cagr = stats.get("cagr", 0.0)
    print(f"\n📈 CAGR (from stats): {api_cagr}%")
    
    # Also show other key stats
    print(f"📉 Max Drawdown: {stats.get('max_drawdown', 'N/A')}%")
    print(f"📊 Sharpe:       {stats.get('sharpe', 'N/A')}")
    print(f"📊 Volatility:   {stats.get('volatility', 'N/A')}")
    
    print("=" * 60)
    
    # Save raw response for debugging
    with open("debug_tools/qqqsim_api_response.json", "w") as f:
        json.dump(resp, f, indent=2)
    print(f"\n💾 Full API response saved to: debug_tools/qqqsim_api_response.json")

if __name__ == "__main__":
    main()

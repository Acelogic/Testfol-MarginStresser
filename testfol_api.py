import requests
import pandas as pd
import numpy as np

API_URL = "https://testfol.io/api/backtest"

def table_to_dicts(df: pd.DataFrame):
    """
    Converts the allocation dataframe to dictionaries for allocation and maintenance.
    """
    df = df.dropna(subset=["Ticker"]).copy()
    alloc = {r["Ticker"].strip(): float(r["Weight %"]) for _,r in df.iterrows()}
    maint = {r["Ticker"].split("?")[0].strip(): float(r["Maint %"]) for _,r in df.iterrows()}
    return alloc, maint

def fetch_backtest(start_date, end_date, start_val, cashflow, cashfreq, rolling,
                   invest_div, rebalance, allocation, return_raw=False, include_raw=False):
    """
    Fetches backtest data from testfol.io API.
    """
    payload = {
        "start_date": str(start_date),
        "end_date":   str(end_date),
        "start_val":  start_val,
        "adj_inflation": False,
        "cashflow": cashflow,
        "cashflow_freq": cashfreq,
        "rolling_window": rolling,
        "backtests": [{
            "invest_dividends": invest_div,
            "rebalance_freq":   rebalance,
            "allocation":       allocation,
            "drag": 0,
            "absolute_dev": 0,
            "relative_dev": 0
        }]
    }
    r = requests.post(API_URL, json=payload, timeout=30)
    r.raise_for_status()
    resp = r.json()
    
    if return_raw:
        return resp  # Return raw response for debugging
    
    stats = resp.get("stats", {})
    if isinstance(stats, list):
        stats = stats[0] if stats else {}
    ts, vals = resp["charts"]["history"]
    dates = pd.to_datetime(ts, unit="s")
    
    extra_data = {
        "rebalancing_events": resp.get("rebalancing_events", []),
        "rebalancing_stats": resp.get("rebalancing_stats", [])
    }
    
    if include_raw:
        extra_data["raw_response"] = resp
    
    return pd.Series(vals, index=dates, name="Portfolio"), stats, extra_data

def simulate_margin(port, starting_loan, rate_annual, draw_monthly, maint_pct, tax_series=None, repayment_series=None):
    """
    Simulates margin loan and calculates equity/usage metrics.
    """
    rate_daily = (rate_annual / 100) / 252
    
    # 1. Create Cashflow Series (Draws + Taxes)
    # Initialize with zeros
    cashflows = pd.Series(0.0, index=port.index)
    
    # Add Monthly Draws
    if draw_monthly > 0:
        # Identify month start/changes
        # We want to add draw at the first available date of each new month
        # Use numpy for speed and to avoid index alignment issues
        months = port.index.month.values
        # Compare with shifted version (roll)
        # np.roll shifts elements. element at 0 moves to 1.
        # We want to compare i with i-1.
        # shifted = np.roll(months, 1)
        # changes = months != shifted
        # changes[0] = False # First element is boundary condition, ignore or handle.
        # Original logic: prev_m initialized to first month. Loop starts at 0.
        # So first day: month == prev_m. No draw.
        # So changes[0] should be False.
        
        month_changes = months != np.roll(months, 1)
        month_changes[0] = False # Explicitly ignore first day (start of sim)
        
        # cashflows is a Series. We can index it with a boolean array.
        cashflows.values[month_changes] += draw_monthly

    # Add Tax Payments
    if tax_series is not None:
        # Align tax_series to port.index, filling missing with 0
        aligned_taxes = tax_series.reindex(port.index, fill_value=0.0)
        cashflows += aligned_taxes
        
    # Add Repayments (Reduce Loan)
    if repayment_series is not None:
        aligned_repayments = repayment_series.reindex(port.index, fill_value=0.0)
        cashflows -= aligned_repayments
        
    # 2. Calculate Loan Balance (Vectorized)
    # Recurrence: L_t = L_{t-1} * (1 + r) + C_t
    # Solution: L_t = L_0 * (1+r)^t + Sum(C_i * (1+r)^(t-i))
    #               = (1+r)^t * [ L_0 + Sum(C_i / (1+r)^i) ]
    # Note: The C_i term usually assumes cashflow happens at end of period?
    # Original loop:
    #   loan *= 1 + rate_daily
    #   loan += cashflow
    # This means interest is applied to previous balance, THEN cashflow is added.
    # So L_t = L_{t-1} * (1+r) + C_t
    
    # Cumulative Interest Factor (1+r)^t
    # We use cumprod.
    # However, for constant rate, (1+r)^t is cleaner.
    # But let's use cumprod to be generic (if we ever want variable rates).
    rate_factor = 1 + rate_daily
    # cum_rate[t] = (1+r)^(t+1) if we just do cumprod?
    # We want factor[t] such that L_t part 1 = L_0 * factor[t]
    # Loop 0: L_0 -> L_0 * (1+r) + C_0.
    # So factor at t=0 should be (1+r).
    cum_rate = pd.Series(rate_factor, index=port.index).cumprod()
    
    # Discounted Cashflows: C_i / (1+r)^(i+1) ?
    # Let's trace t=0: L = L_0*(1+r) + C_0.
    # Formula: L_t = (1+r)^(t+1) * [ L_0 + Sum( C_i / (1+r)^(i+1) ) ]
    # Let's check t=0: (1+r)^1 * [ L_0 + C_0 / (1+r)^1 ] = L_0(1+r) + C_0. Correct.
    
    discounted_cashflows = cashflows / cum_rate
    cum_discounted_cashflows = discounted_cashflows.cumsum()
    
    loan_series = cum_rate * (starting_loan + cum_discounted_cashflows)
    loan_series.name = "Loan"
    
    equity = port - loan_series
    equity_pct = (equity / port).rename("Equity %")
    usage_pct = (loan_series / (port * (1 - maint_pct))).rename("Margin usage %")
    return loan_series, equity, equity_pct, usage_pct

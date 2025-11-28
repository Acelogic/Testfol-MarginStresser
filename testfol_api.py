import requests
import pandas as pd

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

def simulate_margin(port, starting_loan, rate_annual, draw_monthly, maint_pct, tax_series=None):
    """
    Simulates margin loan and calculates equity/usage metrics.
    """
    rate_daily = (rate_annual / 100) / 252
    loan_vals, loan = [], starting_loan
    prev_m = port.index[0].month
    
    # Ensure tax_series is aligned or reindexed if provided
    # We assume tax_series is indexed by Date and contains 0.0 for non-payment days
    
    for d in port.index:
        loan *= 1 + rate_daily
        if draw_monthly and d.month != prev_m:
            loan += draw_monthly
            prev_m = d.month
            
        # Add tax payment if applicable
        if tax_series is not None:
            # Check if there's a tax payment for this day
            # We use .get() which is safe if d is not in index
            payment = tax_series.get(d, 0.0)
            if payment > 0:
                loan += payment
                
        loan_vals.append(loan)
    loan_series = pd.Series(loan_vals, index=port.index, name="Loan")
    equity = port - loan_series
    equity_pct = (equity / port).rename("Equity %")
    usage_pct = (loan_series / (port * (1 - maint_pct))).rename("Margin usage %")
    return loan_series, equity, equity_pct, usage_pct

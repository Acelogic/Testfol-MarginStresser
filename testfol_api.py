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
                   invest_div, rebalance, allocation):
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
    stats = resp.get("stats", {})
    if isinstance(stats, list):
        stats = stats[0] if stats else {}
    ts, vals = resp["charts"]["history"]
    dates = pd.to_datetime(ts, unit="s")
    return pd.Series(vals, index=dates, name="Portfolio"), stats

def simulate_margin(port, starting_loan, rate_annual, draw_monthly, maint_pct):
    """
    Simulates margin loan and calculates equity/usage metrics.
    """
    rate_daily = (rate_annual / 100) / 252
    loan_vals, loan = [], starting_loan
    prev_m = port.index[0].month
    for d in port.index:
        loan *= 1 + rate_daily
        if draw_monthly and d.month != prev_m:
            loan += draw_monthly
            prev_m = d.month
        loan_vals.append(loan)
    loan_series = pd.Series(loan_vals, index=port.index, name="Loan")
    equity = port - loan_series
    equity_pct = (equity / port).rename("Equity %")
    usage_pct = (loan_series / (port * (1 - maint_pct))).rename("Margin usage %")
    return loan_series, equity, equity_pct, usage_pct

import requests
import pandas as pd
import numpy as np
import hashlib
import json
import os
import pickle
import time

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
                   invest_div, rebalance, allocation, return_raw=False, include_raw=False,
                   rebalance_offset=0, cashflow_offset=0):
    """
    Fetches backtest data from testfol.io API with universal disk caching.
    """
    # 1. Generate Cache Key
    # We serialize the arguments to create a unique fingerprint
    cache_payload = {
        "start_date": str(start_date),
        "end_date": str(end_date),
        "start_val": start_val,
        "cashflow": cashflow,
        "cashfreq": cashfreq,
        "rolling": rolling,
        "invest_div": invest_div,
        "rebalance": rebalance,
        "allocation": allocation,
        # return_raw/include_raw affect output format, so they must be part of key
        "return_raw": return_raw,
        "include_raw": include_raw,
        "rebalance_offset": rebalance_offset,
        "cashflow_offset": cashflow_offset
    }
    
    # Sort keys for deterministic JSON
    payload_str = json.dumps(cache_payload, sort_keys=True, default=str)
    req_hash = hashlib.md5(payload_str.encode("utf-8")).hexdigest()
    
    CACHE_DIR = "data/api_cache"
    os.makedirs(CACHE_DIR, exist_ok=True)
    cache_path = os.path.join(CACHE_DIR, f"{req_hash}.pkl")
    
    # 2. Try Cache Load
    if os.path.exists(cache_path):
        try:
            with open(cache_path, "rb") as f:
                # print(f"DEBUG: Cache HIT {req_hash}")
                return pickle.load(f)
        except Exception as e:
            print(f"Cache read failed: {e}")
            # Fall through to API fetch
            pass
            
    # 3. API Request (Cache Miss)
    # print(f"DEBUG: Cache MISS {req_hash} - Fetching from API")
    
    API_URL = "https://testfol.io/api/backtest"
    
    # Format dates as YYYY-MM-DD (API requirement)
    start_str = pd.Timestamp(start_date).strftime('%Y-%m-%d')
    end_str = pd.Timestamp(end_date).strftime('%Y-%m-%d')
    
    payload = {
        "start_date": start_str,
        "end_date":   end_str,
        "start_val":  start_val,
        "adj_inflation": False,
        "cashflow": cashflow,
        "cashflow_freq": cashfreq,
        "cashflow_offset": cashflow_offset,
        "rolling_window": rolling,
        "backtests": [{
            "invest_dividends": invest_div,
            "rebalance_freq":   rebalance,
            "rebalance_offset": rebalance_offset,
            "allocation":       allocation,
            "drag": 0,
            "absolute_dev": 0,
            "relative_dev": 0
        }]
    }
    try:
        r = requests.post(API_URL, json=payload, timeout=30)
        
        # Rate Limit Protection (Only on actual API call)
        time.sleep(2.0) 
        
        r.raise_for_status()
    except requests.exceptions.HTTPError as e:
        # Include response text in error message for debugging
        error_msg = f"HTTP Error: {e}\nResponse: {r.text}"
        raise requests.exceptions.HTTPError(error_msg, response=r)
    except Exception as e:
        raise e


    def _format_size(size_bytes):
        for unit in ['B', 'KB', 'MB', 'GB']:
            if size_bytes < 1024.0:
                return f"{size_bytes:.2f} {unit}"
            size_bytes /= 1024.0
        return f"{size_bytes:.2f} TB"

    print(f"DEBUG: API Success {req_hash} (Msg Size: {_format_size(len(r.content))})")
    resp = r.json()
    
    if return_raw:
        result = resp
        # Save validation
        with open(cache_path, "wb") as f:
            pickle.dump(result, f)
        return result
    
    stats = resp.get("stats", {})
    if isinstance(stats, list):
        stats = stats[0] if stats else {}
    
    # Validation: Ensure charts exist
    if "charts" not in resp or "history" not in resp["charts"]:
        # Don't cache bad/empty responses?
         raise ValueError("Invalid API response: missing chart history")

    ts, vals = resp["charts"]["history"]
    dates = pd.to_datetime(ts, unit="s")
    
    extra_data = {
        "rebalancing_events": resp.get("rebalancing_events", []),
        "rebalancing_stats": resp.get("rebalancing_stats", [])
    }
    
    if include_raw:
        extra_data["raw_response"] = resp
    
    # Construct Result Tuple
    result = (pd.Series(vals, index=dates, name="Portfolio"), stats, extra_data)
    
    # 4. Save to Cache
    try:
        with open(cache_path, "wb") as f:
            pickle.dump(result, f)
    except Exception as e:
        print(f"Cache write failed: {e}")
    
    return result

def simulate_margin(port, starting_loan, rate_annual, draw_monthly, maint_pct, tax_series=None, repayment_series=None):
    """
    Simulates margin loan and calculates equity/usage metrics.
    """
    # 1. Create Cashflow Series (Draws + Taxes)
    # Initialize with zeros
    cashflows = pd.Series(0.0, index=port.index)
    
    # Add Monthly Draws
    if draw_monthly > 0:
        months = port.index.month.values
        month_changes = months != np.roll(months, 1)
        month_changes[0] = False # Explicitly ignore first day
        cashflows.values[month_changes] += draw_monthly

    # Add Tax Payments
    if tax_series is not None:
        aligned_taxes = tax_series.reindex(port.index, fill_value=0.0)
        cashflows += aligned_taxes
        
    # Add Repayments (Reduce Loan)
    if repayment_series is not None:
        aligned_repayments = repayment_series.reindex(port.index, fill_value=0.0)
        cashflows -= aligned_repayments

    # 2. Determine Rate Logic
    # Legacy: rate_annual is a float -> Fixed Rate
    # New: rate_annual can be a dict -> Margin Model
    
    use_vectorized = True
    rate_factor = 0.0 # Will be array or scalar
    effective_rate_series = pd.Series(0.0, index=port.index) # To store annualized % for display
    
    if isinstance(rate_annual, dict):
        model = rate_annual
        mode = model.get("type", "Fixed")
        
        if mode == "Fixed":
            r = model.get("rate_pct", 5.0)
            rate_daily = (r / 100) / 252
            rate_factor = 1 + rate_daily
            effective_rate_series[:] = r
            
        elif mode == "Variable":
            # Variable: Base + Spread
            base_series = model.get("base_series", None)
            spread = model.get("spread_pct", 1.0)
            
            if base_series is not None:
                aligned_base = base_series.reindex(port.index, method='ffill').fillna(0.0)
                daily_rates_pct = aligned_base + spread
                daily_rates_pct = daily_rates_pct.clip(lower=0.0) 
                
                rate_daily_series = (daily_rates_pct / 100) / 252
                rate_factor = 1 + rate_daily_series
                effective_rate_series = daily_rates_pct
            else:
                # Fallback
                rate_factor = 1 + (0.05 / 252)
                effective_rate_series[:] = 5.0

        elif mode == "Tiered":
            use_vectorized = False
            tiers = model.get("tiers", []) 
            base_series = model.get("base_series", None)
            
            if base_series is not None:
                aligned_base = base_series.reindex(port.index, method='ffill').fillna(0.0)
            else:
                aligned_base = pd.Series(5.0, index=port.index)
            
            # --- ITERATIVE CALCULATION FOR TIERED RATES ---
            loan_vals = np.zeros(len(port))
            eff_rate_vals = np.zeros(len(port)) # Store effective rate
            current_loan = starting_loan
            
            base_vals = aligned_base.values
            cf_vals = cashflows.values
            
            for t in range(len(port)):
                daily_base_decimal = (base_vals[t] / 100) / 252
                interest_accrued = 0.0
                calc_balance = current_loan
                
                # Calculate Blended Interest
                for i, (limit, spread) in enumerate(tiers):
                    next_limit = tiers[i+1][0] if i+1 < len(tiers) else float('inf')
                    chunk = min(max(0, calc_balance - limit), next_limit - limit)
                    
                    if chunk > 0:
                        tier_rate_daily = daily_base_decimal + ((spread/100)/252)
                        interest_accrued += chunk * tier_rate_daily
                    
                    if calc_balance < next_limit:
                        break
                
                # Back-calculate effective annualized rate for this day
                # Rate = (Interest / Balance) * 252 * 100
                if current_loan > 1e-9:
                     day_rate = (interest_accrued / current_loan)
                     eff_rate_vals[t] = day_rate * 252 * 100
                else:
                     # If no loan, what is the rate? Technically undefined or Base + lowest spread.
                     # Let's show Base + Tier 1 spread as 'potential' rate
                     base_s = tiers[0][1] if tiers else 0
                     eff_rate_vals[t] = base_vals[t] + base_s
                
                # Update Loan
                current_loan = current_loan + interest_accrued + cf_vals[t]
                loan_vals[t] = current_loan
                
            loan_series = pd.Series(loan_vals, index=port.index)
            effective_rate_series = pd.Series(eff_rate_vals, index=port.index)
            
    else:
        # Legacy Float
        r = rate_annual
        rate_daily = (r / 100) / 252
        rate_factor = 1 + rate_daily
        effective_rate_series[:] = r
    
    if use_vectorized:
        cum_rate = pd.Series(rate_factor, index=port.index).cumprod()
        discounted_cashflows = cashflows / cum_rate
        cum_discounted_cashflows = discounted_cashflows.cumsum()
        loan_series = cum_rate * (starting_loan + cum_discounted_cashflows)

    loan_series.name = "Loan"
    effective_rate_series.name = "Margin Rate %"

    
    equity = port - loan_series
    equity_pct = (equity / port).rename("Equity %")
    usage_pct = (loan_series / (port * (1 - maint_pct))).rename("Margin usage %")
    return loan_series, equity, equity_pct, usage_pct, effective_rate_series

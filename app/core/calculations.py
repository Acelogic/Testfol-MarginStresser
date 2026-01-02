import pandas as pd
import numpy as np

def calculate_cagr(series):
    if series.empty: return 0.0
    start_val = series.iloc[0]
    end_val = series.iloc[-1]
    years = (series.index[-1] - series.index[0]).days / 365.25
    if years <= 0: return 0.0
    cagr = ((end_val / start_val) ** (1 / years) - 1) * 100
    return cagr

def calculate_max_drawdown(series):
    if series.empty: return 0.0
    roll_max = series.cummax()
    drawdown = (series - roll_max) / roll_max
    max_dd = drawdown.min() * 100
    return max_dd

def calculate_sharpe_ratio(series, risk_free_rate=0.0):
    if series.empty: return 0.0
    # Calculate daily returns
    returns = series.pct_change().dropna()
    if returns.empty: return 0.0
    
    # Calculate excess returns
    # risk_free_rate is annual, convert to daily
    rf_daily = risk_free_rate / 252
    excess_returns = returns - rf_daily
    
    # Calculate Sharpe
    std = excess_returns.std()
    if std == 0: return 0.0
    
    sharpe = (excess_returns.mean() / std) * (252 ** 0.5)
    return sharpe

def calculate_tax_adjusted_equity(base_equity_series, tax_payment_series, port_series, loan_series, rate_annual, draw_monthly=0.0):
    """
    Simulates the equity curve if taxes AND draws were paid from capital (reducing the base).
    Accounts for lost compounding and scales future taxes down proportionally.
    Correctly models leverage dynamics: Assets shrink, but Loan (and Interest) remains constant (or zero).
    
    Returns:
        (adjusted_equity_series, adjusted_tax_series)
    """
    # 1. Calculate Asset Returns
    asset_returns = port_series.pct_change().fillna(0)
    
    # 2. Daily Interest Rate
    daily_rate = (1 + rate_annual/100)**(1/365.25) - 1
    
    # 3. Create "External Flow" Series (B_t)
    # B_t = Loan_{t-1} * (r_asset - r_loan) - Draws - Taxes
    
    loan_component = loan_series * (asset_returns - daily_rate)
    
    draws = pd.Series(0.0, index=base_equity_series.index)
    if draw_monthly > 0:
        months = base_equity_series.index.month
        month_changes = months != pd.Series(months, index=base_equity_series.index).shift(1)
        month_changes[0] = False
        draws[month_changes] = draw_monthly
        
    taxes = tax_payment_series.reindex(base_equity_series.index, fill_value=0.0)
    
    external_flows = loan_component - draws - taxes
    
    # Solve Recurrence: E_t = E_{t-1} * A_t + B_t
    growth_factors = 1 + asset_returns
    
    external_flows.iloc[0] = 0
    growth_factors.iloc[0] = 1.0
    cum_growth = growth_factors.cumprod()
    
    discounted_flows = external_flows / cum_growth
    cum_discounted_flows = discounted_flows.cumsum()
    
    e_start = base_equity_series.iloc[0]
    
    adj_equity_series = cum_growth * (e_start + cum_discounted_flows)
    
    # Reconstruct Tax Series (just the input taxes)
    adj_tax_series = taxes
    
    return adj_equity_series, adj_tax_series

def process_rebalancing_data(rebal_events, port_series, allocation):
    """
    Process rebalancing events to calculate trade amounts and realized P&L.
    """
    if not rebal_events:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
        
    trades = []
    composition = []
    
    # Initialize cost basis with initial allocation
    initial_val = port_series.iloc[0]
    cost_basis = {}


    
    # Normalize allocation to 100% just in case
    total_alloc = sum(allocation.values())
    if total_alloc > 0:
        for ticker, weight in allocation.items():
            cost_basis[ticker] = initial_val * (weight / total_alloc)
    
    # Process each event group
    for group in rebal_events:
        tickers = group.get("tickers", [])
        events = group.get("events", [])
        
        for event in events:
            date_str = event[0]
            date = pd.to_datetime(date_str)
            
            # Find portfolio value at rebalance date
            try:
                port_val = port_series.asof(date)
            except:
                continue
                
            if pd.isna(port_val):
                continue
            
            n_tickers = len(tickers)
            
            for i, ticker in enumerate(tickers):
                drift_idx = 1 + i
                weight_idx = 1 + n_tickers + i
                trade_idx = 1 + 2*n_tickers + i
                
                if trade_idx >= len(event):
                    break
                    
                # drift_pct = event[drift_idx]
                trade_pct = event[trade_idx] # This is the actual trade % executed
                
                target_weight = allocation.get(ticker, 0)
                
                trade_amt = port_val * (trade_pct / 100)
                
                # Current Value (Post-Rebalance)
                curr_val = port_val * (target_weight / 100)
                
                # Record composition snapshot
                composition.append({
                    "Date": date,
                    "Ticker": ticker,
                    "Value": curr_val
                })
                
                # Update Cost Basis and Calculate P&L
                realized_pl = 0
                
                if ticker not in cost_basis:
                    cost_basis[ticker] = 0
                    
                if trade_amt > 0: # BUY
                    cost_basis[ticker] += trade_amt
                elif trade_amt < 0: # SELL
                    sell_amt = -trade_amt
                    if curr_val > 0:
                        fraction_sold = sell_amt / curr_val
                        # Cap fraction at 1.0
                        fraction_sold = min(fraction_sold, 1.0)
                        
                        basis_reduction = cost_basis[ticker] * fraction_sold
                        realized_pl = sell_amt - basis_reduction
                        cost_basis[ticker] -= basis_reduction
                
                if trade_amt != 0:
                    trades.append({
                        "Date": date,
                        "Ticker": ticker,
                        "Trade Amount": trade_amt,
                        "Realized P&L": realized_pl,
                        "Year": date.year
                    })
    
    trades_df = pd.DataFrame(trades)
    if not trades_df.empty:
        trades_df["Date"] = pd.to_datetime(trades_df["Date"])
        trades_df = trades_df.sort_values("Date")
        
    composition_df = pd.DataFrame(composition)
    if not composition_df.empty:
        composition_df["Date"] = pd.to_datetime(composition_df["Date"])
        
    # Aggregate P&L by Year
    if not trades_df.empty:
        pl_by_year = trades_df.groupby("Year")["Realized P&L"].sum()
    else:
        pl_by_year = pd.Series(dtype=float)
        
    return trades_df, pl_by_year, composition_df

def generate_stats(series):
    """
    Calculates portfolio statistics from a daily value series.
    Returns a dict matching the structure of Testfol API 'stats'.
    """
    if series.empty:
        return {}
        
    cagr = calculate_cagr(series)
    mdd = calculate_max_drawdown(series)
    sharpe = calculate_sharpe_ratio(series)
    
    # Calculate Standard Deviation (Annualized)
    returns = series.pct_change().dropna()
    std = returns.std() * (252 ** 0.5) * 100
    
    best_year = 0.0
    worst_year = 0.0
    
    # Yearly Returns
    if not series.empty:
        yearly = series.resample('YE').last().pct_change()
        if not yearly.empty:
             best_year = yearly.max() * 100
             worst_year = yearly.min() * 100
    
    return {
        "cagr": cagr,
        "std": std,
        "sharpe": sharpe,
        "max_drawdown": mdd,
        "best_year": best_year,
        "worst_year": worst_year,
        "ulcer": 0.0, # Not vital for now
        "sortino": 0.0 # Not vital for now
    }

def analyze_200dma(series, tolerance_days=0):
    """
    Analyzes the series against its 200-day Moving Average.
    Args:
        series: Price series
        tolerance_days: Maximum days above 200DMA to consider as same event (noise filter)
    Returns:
        dma_series: The 200DMA series.
        events_df: DataFrame of periods where price < 200DMA.
    """
    if series.empty:
        return None, pd.DataFrame()

    dma_series = series.rolling(window=200).mean()
    
    # Identify where Price < DMA
    is_under = series < dma_series
    
    # Identify switch points
    state = is_under.astype(int)
    change = state.diff().fillna(0)
    
    starts = series.index[change == 1]
    ends = series.index[change == -1]
    
    raw_events = []
    
    # 1. Collect Raw Events
    for s_date in starts:
        valid_ends = ends[ends > s_date]
        
        if len(valid_ends) > 0:
            e_date = valid_ends[0]
            raw_events.append({"Start": s_date, "End": e_date, "Status": "Recovered"})
        else:
            raw_events.append({"Start": s_date, "End": pd.NaT, "Status": "Ongoing"})
            
    if not raw_events:
        return dma_series, pd.DataFrame()
        
    # 2. Merge Events Logic
    merged_events = []
    if raw_events:
        current_event = raw_events[0]
        
        for next_event in raw_events[1:]:
            # Check gap
            # Gap is time between Prev End and Next Start
            if pd.notna(current_event["End"]):
                gap = (next_event["Start"] - current_event["End"]).days
                
                if gap <= tolerance_days:
                    # Merge!
                    # Extend current event end to next event end
                    current_event["End"] = next_event["End"]
                    current_event["Status"] = next_event["Status"] # Inherit ongoing status if applicable
                else:
                    # No merge, push current and start new
                    merged_events.append(current_event)
                    current_event = next_event
            else:
                # Current event is ongoing, cannot merge a "next" event (shouldn't happen logically if sorted)
                merged_events.append(current_event)
                current_event = next_event
                
        merged_events.append(current_event)
    
    # 3. Calculate Stats for Merged Events
    final_output = []
    for evt in merged_events:
        s_date = evt["Start"]
        e_date = evt["End"]
        
        if pd.isna(e_date):
            # Ongoing
            now = series.index[-1]
            duration = (now - s_date).days
            calc_end = series.index[-1] # For depth calculation
            status = "Ongoing"
        else:
            duration = (e_date - s_date).days
            calc_end = e_date
            status = "Recovered"
            
        # Analyze period
        # Note: If merged, this includes the 'gap' days where price > 200DMA
        # This is strictly correct for "Time since first break" but might affect "Max Depth" slightly if the gap was huge (but it's limited by tolerance)
        sub_series = series[s_date:calc_end]
        sub_dma = dma_series[s_date:calc_end]
        
        if not sub_series.empty:
            drawdown_from_dma = (sub_series - sub_dma) / sub_dma
            max_depth = drawdown_from_dma.min() * 100
        else:
            max_depth = 0.0
            
        final_output.append({
            "Start Date": s_date,
            "End Date": e_date,
            "Duration (Days)": duration,
            "Duration (Weeks)": duration / 7,
            "Duration (Months)": duration / 30.44,
            "Max Depth (%)": max_depth,
            "Status": status
        })

    return dma_series, pd.DataFrame(final_output)

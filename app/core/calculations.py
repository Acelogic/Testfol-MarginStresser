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

def analyze_ma(series, window=200, tolerance_days=0):
    """
    Analyzes the series against its Moving Average (e.g., 200-day, 150-day).
    Args:
        series: Price series
        window: Moving Average window (default 200)
        tolerance_days: Maximum days above MA to consider as same event (noise filter)
    Returns:
        ma_series: The MA series.
        events_df: DataFrame of periods where price < MA.
    """
    if series.empty:
        return None, pd.DataFrame()

    dma_series = series.rolling(window=window).mean()
    
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
    for i, evt in enumerate(merged_events):
        s_date = evt["Start"]
        e_date = evt["End"]
        
        peak_pct = None
        peak_date = pd.NaT
        days_bottom_to_peak = None
        bottom_to_peak_pct = None
        
        # Determine calc_end for depth/bottom calculation
        if pd.isna(e_date):
            # Ongoing
            now = series.index[-1]
            duration = (now - s_date).days
            calc_end = series.index[-1] 
            status = "Ongoing"
        else:
            duration = (e_date - s_date).days
            calc_end = e_date
            status = "Recovered"

        # Analyze Period (calculate Bottom/Depth first)
        sub_series = series[s_date:calc_end]
        sub_dma = dma_series[s_date:calc_end]
        
        bottom_date = pd.NaT
        if not sub_series.empty:
            drawdown_from_dma = (sub_series - sub_dma) / sub_dma
            max_depth = drawdown_from_dma.min() * 100
            # bottom_date is the date where (Price - DMA)/DMA is minimal (deepest)
            bottom_date = drawdown_from_dma.idxmin()
        else:
            max_depth = 0.0

        # Calculate Subsequent Peak (Only if Recovered)
        if status == "Recovered":
            # Period: From Recovery (e_date) to Next Drop (next_evt.Start) or Now
            if i + 1 < len(merged_events):
                next_start = merged_events[i+1]["Start"]
            else:
                next_start = series.index[-1]
            
            recovery_slice = series[e_date:next_start]
            
            if not recovery_slice.empty:
                recovery_price = series.loc[e_date]
                max_price = recovery_slice.max()
                peak_date = recovery_slice.idxmax()
                peak_pct = ((max_price / recovery_price) - 1) * 100
                
                # Calculate Bottom to Peak % (from actual low to peak)
                if pd.notna(bottom_date):
                    bottom_price = series.loc[bottom_date]
                    bottom_to_peak_pct = ((max_price / bottom_price) - 1) * 100
                else:
                    bottom_to_peak_pct = None
                
                # Check for Active/Current Status
                # If this is the last event and it is recovered, it means the recovery is ongoing (Current)
                if i == len(merged_events) - 1:
                    status = "Recovered (Current)"
                
                # Calculate Days from Bottom to Peak
                if pd.notna(bottom_date) and pd.notna(peak_date):
                    days_bottom_to_peak = (peak_date - bottom_date).days
            else:
                peak_pct = 0.0
                bottom_to_peak_pct = None
            
        final_output.append({
            "Start Date": s_date,
            "End Date": e_date,
            "Duration (Days)": duration,
            "Duration (Weeks)": duration / 7,
            "Duration (Months)": duration / 30.44,
            "Max Depth (%)": max_depth,
            "Subsequent Peak (%)": peak_pct,
            "Bottom to Peak (%)": bottom_to_peak_pct,
            "Peak Date": peak_date,
            "Days Bottom to Peak": days_bottom_to_peak,
            "Status": status
        })

    return dma_series, pd.DataFrame(final_output)

def calculate_pivot_points(high, low, close):
    """Calculates Classic Pivot Points."""
    p = (high + low + close) / 3
    r1 = (2 * p) - low
    s1 = (2 * p) - high
    r2 = p + (high - low)
    s2 = p - (high - low)
    r3 = high + 2 * (p - low)
    s3 = low - 2 * (high - p)
    
    return [
        {"Price": r3, "Label": "Pivot Point 3rd Level Resistance", "Type": "Pivot Resistance"},
        {"Price": r2, "Label": "Pivot Point 2nd Level Resistance", "Type": "Pivot Resistance"},
        {"Price": r1, "Label": "Pivot Point 1st Level Resistance", "Type": "Pivot Resistance"},
        {"Price": p, "Label": "Pivot Point", "Type": "Pivot Point"},
        {"Price": s1, "Label": "Pivot Point 1st Level Support", "Type": "Pivot Support"},
        {"Price": s2, "Label": "Pivot Point 2nd Level Support", "Type": "Pivot Support"},
        {"Price": s3, "Label": "Pivot Point 3rd Level Support", "Type": "Pivot Support"}
    ]

def calculate_cheat_sheet(series, ohlc_data=None):
    """
    Calculates technical levels for a "Trader's Cheat Sheet" (Ladder View).
    Returns:
        pd.DataFrame: Sorted DataFrame with columns ['Price', 'Label', 'Type'].
    Args:
        series (pd.Series): Price history (Close).
        ohlc_data (dict, optional): 'High', 'Low', 'Close' of Previous Period for Pivot Points.
    """
    if series.empty or len(series) < 20:
        return None

    current_price = series.iloc[-1]
    levels = []
    
    # 1. Current Price
    levels.append({"Price": current_price, "Label": "Current Price", "Type": "Current"})
    
    # 2. Moving Averages
    for w in [9, 20, 50, 100, 200]:
        if len(series) >= w:
            val = series.rolling(w).mean().iloc[-1]
            label = f"Price Crosses {w} Day Moving Average"
            levels.append({"Price": val, "Label": label, "Type": "Moving Average"})
        
    # 3. Highs / Lows & Retracements
    periods = {
        "52 Week": 252,
        "13 Week": 65, 
        "1 Month": 21
    }
    
    for name, p in periods.items():
        if len(series) >= p:
            slice_pd = series.iloc[-p:]
            h = slice_pd.max()
            l = slice_pd.min()
            rng = h - l
            
            levels.append({"Price": h, "Label": f"{name} High", "Type": "High/Low"})
            levels.append({"Price": l, "Label": f"{name} Low", "Type": "High/Low"})
            
            if rng > 0:
                # Retracements from High (Down)
                levels.append({"Price": h - (rng * 0.382), "Label": f"38.2% Retracement From {name} High", "Type": "Fibonacci"})
                levels.append({"Price": h - (rng * 0.50), "Label": f"50% Retracement From {name} High/Low", "Type": "Fibonacci"})
                levels.append({"Price": h - (rng * 0.618), "Label": f"61.8% Retracement From {name} High", "Type": "Fibonacci"})
                
                # Retracements from Low (Up)
                levels.append({"Price": l + (rng * 0.382), "Label": f"38.2% Retracement From {name} Low", "Type": "Fibonacci"})
                levels.append({"Price": l + (rng * 0.618), "Label": f"61.8% Retracement From {name} Low", "Type": "Fibonacci"})

    # 4. Standard Deviations (using 20-day std)
    if len(series) >= 20:
        ma20 = series.rolling(20).mean().iloc[-1]
        std20 = series.rolling(20).std().iloc[-1]
        
        # Standard Deviation Resistance/Support approx
        levels.append({"Price": current_price + std20, "Label": "Price 1 Standard Deviation Resistance", "Type": "StdDev"})
        levels.append({"Price": current_price + (2*std20), "Label": "Price 2 Standard Deviations Resistance", "Type": "StdDev"})
        levels.append({"Price": current_price - std20, "Label": "Price 1 Standard Deviation Support", "Type": "StdDev"})
        levels.append({"Price": current_price - (2*std20), "Label": "Price 2 Standard Deviations Support", "Type": "StdDev"})

    # 5. Pivot Points & Session Levels
    if ohlc_data and all(k in ohlc_data for k in ['High', 'Low', 'Close']):
        pivots = calculate_pivot_points(ohlc_data['High'], ohlc_data['Low'], ohlc_data['Close'])
        levels.extend(pivots)
        
        # Add Session Levels
        levels.append({"Price": ohlc_data['High'], "Label": "High", "Type": "Session Level"})
        levels.append({"Price": ohlc_data['Low'], "Label": "Low", "Type": "Session Level"})
        levels.append({"Price": ohlc_data['Close'], "Label": "Previous Close", "Type": "Session Level"})

    df = pd.DataFrame(levels)
    df = df.drop_duplicates(subset=["Label"])
    df = df.sort_values("Price", ascending=False).reset_index(drop=True)
    return df

def analyze_stage(series, ma_window=150, slope_period=20, slope_threshold=None, smoothing_window=5):
    """
    Estimates the Stan Weinstein Stage based on Price vs MA and MA Slope.
    
    Weinstein's methodology uses weekly charts with a 30-week MA. This implementation
    adapts to daily data with a 150-day MA (equivalent to ~30 weeks).

    Args:
        series: Price series.
        ma_window: Period for the Moving Average (default 150, ~30 weeks).
        slope_period: Period to calculate the slope (ROC) of the MA (default 20).
        slope_threshold: Threshold for "Flat" vs Rising/Falling. If None, uses 
                         adaptive threshold based on asset volatility (0.5 * daily std).
        smoothing_window: Window for smoothing stage output to reduce whipsaw (default 5).

    Returns:
        stage_series: Series of strings (e.g., "Stage 2 (Advancing)", "Stage 4 (Declining)")
        slope_series: Series of slope values (ROC of MA).
        ma_series: The calculated MA series.
    """
    if series.empty:
        return None, None, None

    # 1. Calculate MA
    ma_series = series.rolling(window=ma_window).mean()

    # 2. Calculate Slope of MA (Rate of Change over slope_period)
    # Slope = (MA_t / MA_{t-n}) - 1
    slope_series = ma_series.pct_change(periods=slope_period)
    
    # 3. Adaptive Slope Threshold (if not provided)
    # Use 0.5 * the rolling standard deviation of daily returns over slope_period
    # This makes the "flat" determination relative to the asset's volatility
    if slope_threshold is None:
        daily_returns = series.pct_change()
        rolling_vol = daily_returns.rolling(window=slope_period).std()
        # Scale: sqrt(slope_period) to annualize-ish the threshold comparison
        adaptive_threshold = 0.5 * rolling_vol * np.sqrt(slope_period)
        # Fallback minimum threshold to avoid zero
        adaptive_threshold = adaptive_threshold.clip(lower=0.001)
    else:
        adaptive_threshold = slope_threshold

    # 4. Determine Raw Stage
    # Conditions (vectorized, handle adaptive threshold being a Series)
    if isinstance(adaptive_threshold, pd.Series):
        is_rising = slope_series > adaptive_threshold
        is_falling = slope_series < -adaptive_threshold
    else:
        is_rising = slope_series > adaptive_threshold
        is_falling = slope_series < -adaptive_threshold
    
    is_flat = (~is_rising) & (~is_falling)
    
    price_above = series > ma_series
    price_below = series < ma_series
    
    # Create numeric stage encoding for smoothing
    # 2 = Stage 2, 4 = Stage 4, 1 = Stage 1, 3 = Stage 3
    # Sub-stages: 2.1 = Correction, 4.1 = Bear Rally
    stage_codes = pd.Series(0.0, index=series.index)
    
    # Rising MA + Price Above -> Stage 2 (Advancing)
    stage_codes.loc[is_rising & price_above] = 2.0
    
    # Rising MA + Price Below -> Stage 2 (Correction)
    stage_codes.loc[is_rising & price_below] = 2.1
    
    # Falling MA + Price Below -> Stage 4 (Declining)
    stage_codes.loc[is_falling & price_below] = 4.0
    
    # Falling MA + Price Above -> Stage 4 (Bear Rally)
    stage_codes.loc[is_falling & price_above] = 4.1
    
    # Flat MA -> Need to determine Stage 1 vs Stage 3 from prior trend
    # Look back to find the last non-flat stage
    stage_codes.loc[is_flat] = np.nan  # Mark for later resolution
    
    # 5. Distinguish Stage 1 (Basing) vs Stage 3 (Topping)
    # Stage 1: Flat MA following a decline (came from Stage 4)
    # Stage 3: Flat MA following an advance (came from Stage 2)
    
    # Forward-fill the last known trending stage to determine context
    last_trend = stage_codes.copy()
    # Only keep clear trend stages (2.0, 2.1, 4.0, 4.1)
    last_trend[last_trend == 0] = np.nan
    last_trend = last_trend.ffill()
    
    # Now assign Stage 1 or 3 based on prior trend
    was_stage_2 = (last_trend >= 2.0) & (last_trend < 3.0)  # 2.0 or 2.1
    was_stage_4 = (last_trend >= 4.0) & (last_trend < 5.0)  # 4.0 or 4.1
    
    # Apply Stage 1/3 logic
    stage_codes.loc[is_flat & was_stage_4] = 1.0  # Basing after decline
    stage_codes.loc[is_flat & was_stage_2] = 3.0  # Topping after advance
    
    # Any remaining flat with no prior context -> default to neutral
    stage_codes.loc[is_flat & stage_codes.isna()] = 1.5  # Indeterminate
    
    # Fill any remaining NaN (start of series before MA is valid)
    stage_codes = stage_codes.fillna(0)
    
    # 6. Apply Smoothing to reduce whipsaw
    # Use median over the smoothing window (fast, preserves stage values)
    if smoothing_window > 1:
        # Median is fast and reasonably preserves discrete codes
        smoothed_codes = stage_codes.rolling(window=smoothing_window, min_periods=1, center=True).median()
        # Round back to nearest valid code
        smoothed_codes = smoothed_codes.round(1)
    else:
        smoothed_codes = stage_codes
    
    # 7. Map codes back to string labels
    def code_to_stage(code):
        if code == 2.0:
            return "Stage 2 (Advancing)"
        elif code == 2.1:
            return "Stage 2 (Correction)"
        elif code == 4.0:
            return "Stage 4 (Declining)"
        elif code == 4.1:
            return "Stage 4 (Bear Rally)"
        elif code == 1.0:
            return "Stage 1 (Basing)"
        elif code == 3.0:
            return "Stage 3 (Topping)"
        elif code == 1.5:
            return "Stage 1/3 (Neutral)"
        else:
            return "Unknown"
    
    stages = smoothed_codes.apply(code_to_stage)

    return stages, slope_series, ma_series

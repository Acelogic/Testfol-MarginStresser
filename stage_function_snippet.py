
def analyze_stage(series, ma_window=150, slope_period=20, slope_threshold=0.001):
    """
    Estimates the Stan Weinstein Stage based on Price vs MA and MA Slope.

    Args:
        series: Price series.
        ma_window: Period for the Moving Average (default 150).
        slope_period: Period to calculate the slope (ROC) of the MA (default 20).
        slope_threshold: Threshold for "Flat" vs Rising/Falling (default 0.1%).

    Returns:
        stage_series: Series of strings (e.g., "Stage 2 (Advancing)", "Stage 4 (Declining)", "Stage 1/3 (Neutral)")
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

    # 3. Determine Stage
    # We'll use a simple vectorised approach
    
    # Conditions
    is_rising = slope_series > slope_threshold
    is_falling = slope_series < -slope_threshold
    is_flat = (~is_rising) & (~is_falling)
    
    price_above = series > ma_series
    price_below = series < ma_series
    
    # Create Series
    stages = pd.Series("Unknown", index=series.index)
    
    # Logic Table
    # Rising MA + Price Above -> Stage 2
    stages.loc[is_rising & price_above] = "Stage 2 (Advancing)"
    
    # Rising MA + Price Below -> Uptrend Correction (Still technically Stage 2 context, or early 3/4)
    # Weinstein: If price breaks below rising 30w MA, it's a warning, but trend technically up until MA turns.
    stages.loc[is_rising & price_below] = "Stage 2 (Correction)"
    
    # Falling MA + Price Below -> Stage 4
    stages.loc[is_falling & price_below] = "Stage 4 (Declining)"
    
    # Falling MA + Price Above -> Bear Rally
    stages.loc[is_falling & price_above] = "Stage 4 (Bear Rally)"
    
    # Flat MA -> Stage 1 or 3
    # Distinguishing 1 (Base) vs 3 (Top) requires context (prior move). 
    # For simplicity/statelessness:
    stages.loc[is_flat] = "Stage 1/3 (Neutral)"

    return stages, slope_series, ma_series

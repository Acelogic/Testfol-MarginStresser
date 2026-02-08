from __future__ import annotations

import pandas as pd
import numpy as np


def calculate_pivot_points(
    high: float, low: float, close: float,
) -> list[dict]:
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

def calculate_cheat_sheet(
    series: pd.Series,
    ohlc_data: dict | None = None,
) -> pd.DataFrame | None:
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

def analyze_stage(
    series: pd.Series,
    ma_window: int = 150,
    slope_period: int = 20,
    slope_threshold: float | None = None,
    smoothing_window: int = 5,
) -> tuple[pd.Series | None, pd.Series | None, pd.Series | None]:
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

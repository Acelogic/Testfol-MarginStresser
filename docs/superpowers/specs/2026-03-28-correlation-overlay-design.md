# Correlation Overlay on Component Performance Chart — Design Spec

## Overview

Add a rolling average pairwise correlation line (purple, right y-axis) and red shaded regions (all-assets-declining periods) overlaid on the existing component performance chart in `render_portfolio_allocation()`. Includes a configurable rolling window dropdown.

## Architecture

All changes are contained within `app/ui/charts/rebalancing.py`, inside `render_portfolio_allocation()`. No new files or modules. The correlation math is simple enough to live inline.

## Calculation

### Rolling Average Pairwise Correlation

1. Take the daily returns DataFrame (`modified_returns`) already computed in the function for each available component.
2. For each rolling window, compute `returns[available].rolling(window).corr()` — this produces a pairwise correlation matrix at each timestep.
3. Extract the upper triangle (excluding diagonal) at each timestep and average the values to produce a single "average correlation" series.
4. Simpler approach: iterate over all pairs `(i, j)` where `i < j`, compute `returns[i].rolling(window).corr(returns[j])`, then average across all pair series.

The result is a single `pd.Series` with values in [-1.0, +1.0] at each date.

### All-Assets-Declining Detection

For each date, check if all assets have negative rolling returns over the same window:
1. Compute `returns[t].rolling(window).sum()` for each asset `t` (cumulative return over the window).
2. A date is "all declining" if every asset's rolling return is negative.
3. Collect contiguous date ranges where this is true to draw red shaded rectangles.

## Rendering

### Dropdown Control

`st.selectbox` placed above the chart:
- Label: "Correlation Window"
- Options: `["1 month", "3 months", "6 months", "1 year", "2 years", "3 years", "5 years"]`
- Mapped to trading days: `[21, 63, 126, 252, 504, 756, 1260]`
- Default: "1 year" (252 days)
- Key: `f"corr_window_{unique_id}"`

### Correlation Line

- Added as a `go.Scatter` trace on the existing `fig_lines` figure.
- Assigned to `yaxis="y2"` (right y-axis).
- Color: `#a78bfa` (purple), width 1.5, semi-transparent.
- Name: "Avg Correlation" in legend.
- Right y-axis range: [-1.0, 1.0], with a dashed zero line.

### Red Shaded Regions

- Added as `fig_lines.add_vrect()` shapes for each contiguous all-declining period.
- Fill: `rgba(239, 68, 68, 0.10)` — semi-transparent red.
- Border: `rgba(239, 68, 68, 0.3)` on left/right edges.
- No annotation text (would be too cluttered over long backtests).

### Y-Axis Layout Update

Add to the existing `fig_lines.update_layout()`:
```python
yaxis2=dict(
    title="Avg Correlation",
    overlaying="y",
    side="right",
    range=[-1.05, 1.05],
    showgrid=False,
    tickvals=[-1, -0.5, 0, 0.5, 1],
    titlefont=dict(color="#a78bfa"),
    tickfont=dict(color="#a78bfa"),
)
```

## Edge Cases

- **1 asset**: Skip correlation entirely — no line, no dropdown, no red shading.
- **2 assets**: Single pair, no averaging needed. Works naturally.
- **Window > data length**: Show `st.warning` and fall back to the largest window that fits.
- **All NaN in window**: The rolling correlation will naturally produce NaN for early dates before enough data exists. Plotly handles NaN gaps gracefully.

## Files Modified

| File | Change |
|------|--------|
| `app/ui/charts/rebalancing.py` | Add correlation overlay logic inside `render_portfolio_allocation()`: dropdown, correlation calculation, red shading, second y-axis trace |

## What This Does NOT Change

- No changes to the calculation layer (`stats.py`) — this is purely a chart feature.
- No changes to the data flow — uses returns already computed in the function.
- The existing component performance chart behavior is unchanged (lines, rebalance markers, stacked area chart below).

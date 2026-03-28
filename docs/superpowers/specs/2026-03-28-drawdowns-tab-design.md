# Drawdowns Tab — Design Spec

## Overview

Add a "Drawdowns" tab to the Returns Analysis section (`📉 Drawdowns`) as the 6th sub-tab after Daily. It shows all portfolio corrections >5% with SPY comparison, severity filtering, and market event labels. Works for any portfolio, not just NDXMEGASPLIT.

## Architecture

**Approach B**: Calculations in `app/core/calculations/stats.py`, rendering in `app/ui/charts/returns.py`.

### Calculation Layer — `app/core/calculations/stats.py`

#### `find_drawdown_episodes(series, threshold=-0.05) -> list[dict]`

Moved from `data/ndx_simulation/scripts/ad_hoc/corrections_analysis.py`. Detects all drawdown episodes exceeding the threshold.

Each episode dict contains:
- `peak_date`, `peak_val` — start of drawdown
- `trough_date`, `trough_val` — bottom of drawdown
- `dd` — max drawdown as decimal (e.g. -0.25 for -25%)
- `recovery` — date portfolio returned to prior peak, or `None` if ongoing

#### `EVENT_MAP` dict and `get_market_event(peak_date) -> str`

Moved from corrections script. Maps (year, month) tuples to event description strings. `get_market_event()` does fuzzy matching (+/- 2 months) to find the nearest event.

Events are keyed to SPY drawdown peaks, so they work as a universal market reference regardless of portfolio.

#### `build_drawdown_table(port_series, spy_series) -> pd.DataFrame`

Orchestrates the full drawdown analysis. Steps:
1. Call `find_drawdown_episodes()` on `port_series`
2. For each episode, compute SPY drawdown metrics over the same peak-to-recovery window
3. Map each episode to a market event via `get_market_event()` using the peak date
4. Format recovery durations using `fmt_duration()` helper (e.g. "5.0yr", "3mo", "59d")
5. For ongoing episodes, show elapsed time: "ongoing (2mo)"

Returns a DataFrame with columns:

| Column | Type | Description |
|--------|------|-------------|
| `Correction Period` | str | "Jul 17, 2000 - Oct 09, 2002" (* if ongoing) |
| `Days` | int | Decline duration (peak to trough) |
| `% Decline` | float | Max drawdown % (negative) |
| `Recovery from Bottom` | str | Time from trough to prior high |
| `Decline + Recovery Time` | str | Total time underwater |
| `SPY DD` | float | SPY max drawdown % during same period |
| `SPY Recovery from Bottom` | str | SPY recovery duration |
| `SPY Decline + Recovery Time` | str | SPY total underwater duration |
| `Ratio` | float | abs(portfolio DD / SPY DD) |
| `Market Event` | str | Event label from EVENT_MAP |

Hidden metadata columns (used for filtering, not displayed):
| Column | Type | Description |
|--------|------|-------------|
| `_ongoing` | bool | Whether episode is ongoing |
| `_severity` | str | "Severe", "Moderate", "Mild", or "Minor" |
| `_decline_raw` | float | Raw decline value for median calculation |
| `_spy_dd_raw` | float | Raw SPY DD value for median calculation |
| `_ratio_raw` | float | Raw ratio value for median calculation |
| `_days_raw` | int | Raw days value for median calculation |

#### `fmt_duration(days) -> str`

Moved from corrections script's `fmt_recov()`. Formats day counts into human-readable durations:
- >= 365 days: "X.Xyr"
- >= 60 days: "Xmo"
- < 60 days: "Xd"

### Market Benchmark Data Sourcing

Uses SPYSIM (not SPY) to maximize date range coverage. SPY only goes back to 1993, but SPYSIM provides simulated S&P 500 data back to ~2000, matching portfolios that use SIM tickers (NDXMEGASIM, etc.).

The rendering layer fetches SPYSIM prices via `fetch_component_data(["SPYSIM"], start_date, end_date)` from `app/services/data_service.py` — the same provider chain and SIM ticker handling used elsewhere in the app. The SPYSIM series is normalized to the portfolio's start value for consistent comparison.

Column headers still say "SPY" for user clarity (SPYSIM is an implementation detail).

### Rendering Layer — `app/ui/charts/returns.py`

#### Tab Definition

Change line 536 from 5 tabs to 6:

```python
tab_summary, tab_annual, tab_quarterly, tab_monthly, tab_daily, tab_drawdowns = st.tabs(
    ["📋 Summary", "📅 Annual", "📆 Quarterly", "🗓️ Monthly", "📊 Daily", "📉 Drawdowns"]
)
```

#### Tab Content (`with tab_drawdowns:`)

**1. Data preparation:**
- Fetch SPYSIM prices via `fetch_component_data(["SPYSIM"], ...)` using the portfolio's date range
- Normalize SPYSIM to portfolio start value
- Call `build_drawdown_table(port_series, spy_series)` to get the full DataFrame

**2. Summary stats bar** — `st.columns(5)` with `st.metric()`:
- Corrections: total count
- Median Decline: median of `_decline_raw`
- Severe (>25%): count where `_severity == "Severe"`
- Moderate (15-25%): count where `_severity == "Moderate"`
- Ongoing: count where `_ongoing == True`

**3. Severity filter** — `st.radio()` with `horizontal=True`:
- Options: "All (N)", "Severe >25% (N)", "Moderate 15-25% (N)", "Mild 10-15% (N)", "Minor 5-10% (N)"
- Filters the DataFrame by `_severity` column
- Counts are computed from the full (unfiltered) DataFrame

**4. Drawdowns table** — `st.dataframe()` with `.style` chain:
- Drop hidden `_` columns before display
- Color coding for `% Decline`:
  - Severe (>25%): red, bold
  - Moderate (15-25%): orange, bold
  - Mild (10-15%): yellow
  - Minor (5-10%): default gray
- Color coding for `Ratio`:
  - < 1.5x: green
  - 1.5-3.0x: default white
  - > 3.0x: orange
- Ongoing rows: "ongoing (Xd/Xmo)" text in red bold
- SPY columns: gray text (secondary emphasis)

**5. Median row:**
- Appended as the last row of the filtered DataFrame
- Shows: "Median" label, median Days, median % Decline, median SPY DD, median Ratio
- Recovery/duration columns left blank (medians of durations aren't meaningful)
- Styled with blue text to distinguish from data rows

## Severity Classification

| Category | Range | Color |
|----------|-------|-------|
| Severe | > 25% decline | Red |
| Moderate | 15-25% decline | Orange |
| Mild | 10-15% decline | Yellow |
| Minor | 5-10% decline | Gray |

## Files Modified

| File | Change |
|------|--------|
| `app/core/calculations/stats.py` | Add `find_drawdown_episodes()`, `build_drawdown_table()`, `fmt_duration()`, `get_market_event()`, `EVENT_MAP` |
| `app/ui/charts/returns.py` | Add 6th tab `📉 Drawdowns` with summary stats, filter, and styled table |

## Files NOT Modified

- `data/ndx_simulation/scripts/ad_hoc/corrections_analysis.py` — keeps its own copy of `find_episodes()` and `EVENT_MAP`. The ad-hoc script is standalone and generates static HTML/PNG. No need to couple it to the app module.

## Edge Cases

- **Short portfolios**: Portfolios with <1 year of data may have 0 corrections >5%. Show "No corrections >5% found in this period." message.
- **No SPYSIM overlap**: If SPYSIM data is unavailable for the portfolio's date range, SPY columns show "N/A".
- **Single ongoing drawdown**: If the portfolio has never recovered from its only drawdown, show 1 row with all "ongoing" values.

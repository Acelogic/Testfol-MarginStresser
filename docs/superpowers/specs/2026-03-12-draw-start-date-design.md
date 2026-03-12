# Draw Start Date Feature

## Problem
Monthly margin draws currently start from the first month boundary of the backtest. Users need control over when draws begin.

## Solution
Add a global `draw_start_date` date picker to the Margin & Financing tab. Draws only occur on month boundaries at or after this date.

## UI
- `st.date_input` labeled "Draw Start Date" in `app/ui/configuration.py`, Margin & Financing tab
- Conditionally visible: only shown when `draw_monthly > 0`
- Default: `None` (draws start from backtest start — preserves current behavior)
- Stored as `config['draw_start_date']` (`datetime.date | None`)

## Data Flow
```
config['draw_start_date']
  → testfol_charting.py: pm_config['draw_start_date']
  → backtest_orchestrator.py: extract, clamp to backtest range, pass downstream
  → shadow_backtest.py: guard `if date >= draw_start_date` before loan_balance += draw_monthly
  → testfol_api.py: AND mask `port.index.date >= draw_start_date` with month-change mask
  → stats.py: same mask in calculate_tax_adjusted_equity()
```

## Clamping (in orchestrator)
- `draw_start_date` before backtest start → use backtest start
- `draw_start_date` after backtest end → no draws occur (naturally: no month boundaries match)
- `draw_start_date` is `None` → use backtest start date

## Files Modified

| File | Change |
|---|---|
| `app/ui/configuration.py` | Add `st.date_input` after Monthly Draw input |
| `testfol_charting.py` | Pass `draw_start_date` into `pm_config` |
| `app/core/backtest_orchestrator.py` | Extract, clamp, forward to shadow backtest and results |
| `app/core/shadow_backtest.py` | Add date guard in month-change draw logic (~line 536) |
| `app/services/testfol_api.py` | Add date mask in `simulate_margin()` (~line 201) |
| `app/core/calculations/stats.py` | Add date mask in `calculate_tax_adjusted_equity()` (~line 69) |

## Chart Impact
None. Charts render `loan_series`, `equity_series`, `usage_pct` which are already downstream of the simulation. The delayed draw start naturally flows through.

## Testing
- Unit test: `simulate_margin()` with `draw_start_date` set mid-backtest — verify loan is zero before that date, grows after
- Unit test: clamping logic — date before range, after range, `None`
- Integration: run app, set draw start date, verify chart series reflect delayed draws

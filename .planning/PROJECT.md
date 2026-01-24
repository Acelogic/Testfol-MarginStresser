# Margin Stresser - 200DMA Breach Analysis

## What This Is

A Streamlit-based portfolio backtesting tool with margin simulation, tax-aware calculations, and technical analysis. This milestone adds a 200DMA breach analysis view that compares buying portfolio dips vs. holding SPY during the same period.

## Core Value

Users can determine if historically buying their portfolio's 200DMA breaches would have outperformed simply holding SPY — helping inform whether "buying the dip" is a winning strategy for their specific allocation.

## Requirements

### Validated

<!-- Existing capabilities from current codebase -->

- ✓ Portfolio backtesting with margin loan simulation — existing
- ✓ Tax-aware calculations with FIFO/LIFO/HIFO lot tracking — existing
- ✓ Monte Carlo stress testing — existing
- ✓ ETF X-Ray holdings decomposition — existing
- ✓ 200DMA and moving average technical analysis — existing
- ✓ Variable and tiered margin rate models — existing
- ✓ Multi-portfolio comparison — existing

### Active

<!-- New feature: 200DMA Breach Analysis -->

- [ ] Detect all historical 200DMA breach events for a portfolio price series
- [ ] Calculate breach-to-recovery returns (entry at breach date)
- [ ] Calculate max-depth-to-recovery returns (entry at lowest point during breach)
- [ ] Fetch SPYSIM returns for the same time windows
- [ ] Compute alpha (outperformance/underperformance) vs SPYSIM for both entry strategies
- [ ] Display breach events in a single table with both entry scenarios
- [ ] Show summary stats: total events, win rates, average alpha per strategy
- [ ] Add view below existing table in portfolio's 200DMA tab

### Out of Scope

- Individual ticker breach analysis within a multi-ticker portfolio — user can create single-ticker portfolio if needed
- Real-time breach alerts or notifications — this is historical analysis only
- Comparison against benchmarks other than SPYSIM — SPY is the default comparison

## Context

The existing 200DMA section shows moving average overlays and technical indicators. This feature extends it with a historical "breach performance" analysis to answer: "When my portfolio dropped below its 200DMA, would buying the dip have beaten holding SPY?"

Two entry scenarios are modeled:
1. **Breach Entry**: Buy when portfolio crosses below 200DMA, sell at recovery (crosses back above)
2. **Max Depth Entry**: Buy at the lowest point during the breach, sell at recovery

The portfolio price series can be a single ticker or an aggregate of multiple weighted holdings — the analysis works on whatever combined series is configured.

## Constraints

- **Tech stack**: Python/Streamlit — must integrate with existing UI patterns in `app/ui/`
- **Data source**: Use existing data service infrastructure (Testfol API, yfinance) for SPYSIM comparison
- **UI placement**: Add below existing table in 200DMA portfolio tab — no new tabs or pages

## Key Decisions

| Decision | Rationale | Outcome |
|----------|-----------|---------|
| Single table with both entry scenarios | User requested unified view rather than toggle | — Pending |
| SPYSIM as benchmark | Consistent with existing app's SPY simulation data | — Pending |
| Summary stats included | Win rates and average alpha provide quick insight | — Pending |

---
*Last updated: 2026-01-23 after initialization*

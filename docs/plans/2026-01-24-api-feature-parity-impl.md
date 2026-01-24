# API Feature Parity Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add 4 new FastAPI endpoints (`/chart_data`, `/performance`, `/tax_rebal`, `/compare`) to achieve feature parity with the Streamlit app.

**Architecture:** Each endpoint is a feature bundle serving mobile app screens. Endpoints reuse existing core functions (`calculations.py`, `shadow_backtest.py`, `tax_library.py`) with Pydantic request/response validation. A shared helper module handles JSON serialization of numpy/pandas types.

**Tech Stack:** FastAPI, Pydantic, pandas, numpy

---

## Task 1: Add JSON Serialization Helper

**Files:**
- Create: `app/services/json_utils.py`

**Step 1: Create the helper module**

```python
# app/services/json_utils.py
"""JSON serialization helpers for numpy/pandas types."""
import numpy as np
import pandas as pd
from typing import Any, Dict, List, Union


def safe_float(val: Any) -> Union[float, None]:
    """Convert numpy/pandas float to Python float, handling NaN/Inf."""
    if val is None:
        return None
    if isinstance(val, (np.floating, np.integer)):
        val = float(val)
    if isinstance(val, float):
        if np.isnan(val) or np.isinf(val):
            return None
        return val
    return val


def safe_int(val: Any) -> Union[int, None]:
    """Convert numpy int to Python int."""
    if val is None:
        return None
    if isinstance(val, (np.integer,)):
        return int(val)
    return val


def series_to_list(series: pd.Series) -> List[float]:
    """Convert pandas Series to list of floats, handling NaN."""
    return [safe_float(v) for v in series.values]


def dates_to_strings(index: pd.DatetimeIndex) -> List[str]:
    """Convert DatetimeIndex to ISO date strings."""
    return [d.strftime('%Y-%m-%d') for d in index]


def safe_dict(d: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively convert dict values to JSON-safe types."""
    result = {}
    for k, v in d.items():
        if isinstance(v, (np.floating, np.integer)):
            result[k] = safe_float(v)
        elif isinstance(v, dict):
            result[k] = safe_dict(v)
        elif isinstance(v, pd.Timestamp):
            result[k] = v.strftime('%Y-%m-%d')
        elif pd.isna(v):
            result[k] = None
        else:
            result[k] = v
    return result
```

**Step 2: Commit**

```bash
git add app/services/json_utils.py
git commit -m "feat(api): add JSON serialization helpers for numpy/pandas types"
```

---

## Task 2: Add Pydantic Request Models

**Files:**
- Modify: `app/services/backend.py` (add after line 66, before `@app.get("/")`)

**Step 1: Add new request models**

Add the following after `class MonteCarloRequest` and before `@app.get("/")`:

```python
# --- New Request Models for Feature Parity ---

class ChartDataRequest(BaseModel):
    """Request for /chart_data endpoint."""
    # Option A: Pass existing series
    portfolio_series: Optional[List[List]] = None  # [[timestamps], [values]]

    # Option B: Fetch fresh backtest
    tickers: Optional[Dict[str, float]] = None
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    rebalance_freq: str = "Quarterly"
    invest_div: bool = True

    # Analysis options
    ma_windows: List[int] = [150, 200]
    include_technicals: bool = True
    include_stage: bool = True
    tolerance_days: int = 14


class PerformanceRequest(BaseModel):
    """Request for /performance endpoint."""
    tickers: Dict[str, float]
    start_date: str
    end_date: Optional[str] = None
    start_val: float = 10000.0
    rebalance_freq: str = "Quarterly"
    cashflow: float = 0.0
    cashflow_freq: str = "Monthly"
    invest_div: bool = True

    benchmark: Optional[str] = "SPYSIM"
    include_drawdown_series: bool = True


class TaxConfig(BaseModel):
    """Tax configuration for /tax_rebal endpoint."""
    income: float = 100000
    filing_status: str = "Single"
    state_tax_rate: float = 0.0
    method: str = "2025_fixed"
    use_std_deduction: bool = True


class TaxRebalRequest(BaseModel):
    """Request for /tax_rebal endpoint."""
    tickers: Dict[str, float]
    start_date: str
    end_date: Optional[str] = None
    start_val: float = 10000.0
    rebalance_freq: str = "Quarterly"
    cashflow: float = 0.0
    cashflow_freq: str = "Monthly"
    invest_div: bool = True

    tax_config: TaxConfig = TaxConfig()
    include_composition: bool = True
    include_trades: bool = True


class PortfolioConfig(BaseModel):
    """Single portfolio configuration for /compare endpoint."""
    name: str
    tickers: Dict[str, float]
    rebalance_freq: str = "Quarterly"


class CompareRequest(BaseModel):
    """Request for /compare endpoint."""
    portfolios: List[PortfolioConfig]
    start_date: str
    end_date: Optional[str] = None
    start_val: float = 10000.0

    benchmark: Optional[str] = "SPYSIM"
    align_to_common_start: bool = True
```

**Step 2: Add import for calculations module**

Add to the imports at the top of `backend.py` (after line 12):

```python
from app.core import calculations
from app.services.json_utils import safe_float, safe_int, series_to_list, dates_to_strings, safe_dict
```

**Step 3: Commit**

```bash
git add app/services/backend.py
git commit -m "feat(api): add Pydantic request models for new endpoints"
```

---

## Task 3: Implement `/chart_data` Endpoint

**Files:**
- Modify: `app/services/backend.py` (add after `/monte_carlo` endpoint, before `/run_stress_test`)

**Step 1: Add the endpoint**

Add after the `run_monte_carlo` function (after line 116):

```python
@app.post("/chart_data")
def get_chart_data(req: ChartDataRequest):
    """
    Returns MA analysis, technical levels, and stage analysis for chart overlays.
    """
    try:
        # 1. Get or fetch portfolio series
        if req.portfolio_series is not None:
            # Use provided series
            timestamps, values = req.portfolio_series
            dates = pd.to_datetime(timestamps, unit='s')
            port_series = pd.Series(values, index=dates, name="Portfolio")
        elif req.tickers is not None and req.start_date is not None:
            # Fetch fresh backtest
            port_series, _, _ = api.fetch_backtest(
                start_date=req.start_date,
                end_date=req.end_date if req.end_date else str(dt.date.today()),
                start_val=10000.0,
                cashflow=0,
                cashfreq="Monthly",
                rolling=1,
                invest_div=req.invest_div,
                rebalance=req.rebalance_freq,
                allocation=req.tickers
            )
        else:
            raise HTTPException(status_code=400, detail="Must provide either portfolio_series or tickers+start_date")

        if port_series.empty:
            raise HTTPException(status_code=400, detail="No data returned for portfolio")

        response = {}

        # 2. MA Analysis for each window
        ma_analysis = {}
        for window in req.ma_windows:
            ma_series, events_df = calculations.analyze_ma(
                port_series,
                window=window,
                tolerance_days=req.tolerance_days
            )

            current_price = port_series.iloc[-1]
            current_ma = ma_series.iloc[-1] if ma_series is not None and not ma_series.empty else None

            # Build breach events list
            breach_events = []
            if not events_df.empty:
                # Get comparison data with SPYSIM
                comparison_df = calculations.compare_breach_events(
                    port_series,
                    window=window,
                    tolerance_days=req.tolerance_days
                )

                for _, row in comparison_df.iterrows():
                    breach_events.append({
                        "start_date": row["Start Date"].strftime('%Y-%m-%d') if pd.notna(row["Start Date"]) else None,
                        "end_date": row["End Date"].strftime('%Y-%m-%d') if pd.notna(row["End Date"]) else None,
                        "duration_days": safe_int(row.get("Duration (Days)")),
                        "max_depth_pct": safe_float(row.get("Max Depth (%)")),
                        "bottom_date": row["Bottom Date"].strftime('%Y-%m-%d') if pd.notna(row.get("Bottom Date")) else None,
                        "recovery_days": safe_int(row.get("Recovery Days")),
                        "breach_return_pct": safe_float(row.get("Breach Entry Return (%)")),
                        "max_depth_return_pct": safe_float(row.get("Max-Depth Entry Return (%)")),
                        "spysim_breach_return_pct": safe_float(row.get("SPYSIM Breach Return (%)")),
                        "spysim_maxdepth_return_pct": safe_float(row.get("SPYSIM Max-Depth Return (%)")),
                        "breach_alpha_pct": safe_float(row.get("Breach Entry Alpha (%)")),
                        "maxdepth_alpha_pct": safe_float(row.get("Max-Depth Entry Alpha (%)")),
                        "status": row.get("Status", "Unknown")
                    })

                # Calculate summary statistics
                total_events = len(comparison_df)
                avg_duration = comparison_df["Duration (Days)"].mean() if "Duration (Days)" in comparison_df else 0
                avg_depth = comparison_df["Max Depth (%)"].mean() if "Max Depth (%)" in comparison_df else 0

                # Win rates (alpha > 0)
                breach_alphas = comparison_df["Breach Entry Alpha (%)"].dropna()
                maxdepth_alphas = comparison_df["Max-Depth Entry Alpha (%)"].dropna()
                breach_win_rate = (breach_alphas > 0).mean() * 100 if len(breach_alphas) > 0 else 0
                maxdepth_win_rate = (maxdepth_alphas > 0).mean() * 100 if len(maxdepth_alphas) > 0 else 0

                summary = {
                    "total_events": total_events,
                    "avg_duration_days": safe_float(avg_duration),
                    "avg_depth_pct": safe_float(avg_depth),
                    "breach_win_rate": safe_float(breach_win_rate),
                    "maxdepth_win_rate": safe_float(maxdepth_win_rate),
                    "avg_breach_alpha": safe_float(breach_alphas.mean()) if len(breach_alphas) > 0 else 0,
                    "avg_maxdepth_alpha": safe_float(maxdepth_alphas.mean()) if len(maxdepth_alphas) > 0 else 0
                }
            else:
                summary = {
                    "total_events": 0,
                    "avg_duration_days": 0,
                    "avg_depth_pct": 0,
                    "breach_win_rate": 0,
                    "maxdepth_win_rate": 0,
                    "avg_breach_alpha": 0,
                    "avg_maxdepth_alpha": 0
                }

            ma_analysis[str(window)] = {
                "current_ma": safe_float(current_ma),
                "price_vs_ma_pct": safe_float(((current_price / current_ma) - 1) * 100) if current_ma and current_ma > 0 else None,
                "is_above_ma": bool(current_price > current_ma) if current_ma else None,
                "breach_events": breach_events,
                "summary": summary
            }

        response["ma_analysis"] = ma_analysis

        # 3. Stage Analysis (Weinstein)
        if req.include_stage:
            stages, slope_series, ma_series_stage = calculations.analyze_stage(port_series, ma_window=150)

            if stages is not None:
                current_stage = stages.iloc[-1] if not stages.empty else "Unknown"
                current_slope = slope_series.iloc[-1] if slope_series is not None and not slope_series.empty else 0

                # Convert stage history to lists
                stage_timestamps = [int(d.timestamp()) for d in stages.index]
                stage_codes = stages.tolist()

                response["stage_analysis"] = {
                    "current_stage": current_stage,
                    "ma_slope": safe_float(current_slope),
                    "stage_history": [stage_timestamps, stage_codes]
                }
            else:
                response["stage_analysis"] = None

        # 4. Technical Levels (Cheat Sheet)
        if req.include_technicals:
            # Get previous period OHLC for pivot points
            ohlc_data = None
            if len(port_series) > 1:
                # Use last full month as "previous period"
                monthly = port_series.resample("ME").agg(['first', 'max', 'min', 'last'])
                if len(monthly) >= 2:
                    prev_month = monthly.iloc[-2]
                    ohlc_data = {
                        'High': prev_month['max'],
                        'Low': prev_month['min'],
                        'Close': prev_month['last']
                    }

            cheat_sheet_df = calculations.calculate_cheat_sheet(port_series, ohlc_data=ohlc_data)

            if cheat_sheet_df is not None and not cheat_sheet_df.empty:
                technical_levels = []
                for _, row in cheat_sheet_df.iterrows():
                    technical_levels.append({
                        "price": safe_float(row["Price"]),
                        "label": row["Label"],
                        "type": row["Type"]
                    })
                response["technical_levels"] = technical_levels
            else:
                response["technical_levels"] = []

        return response

    except HTTPException:
        raise
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
```

**Step 2: Commit**

```bash
git add app/services/backend.py
git commit -m "feat(api): implement /chart_data endpoint with MA analysis and technicals"
```

---

## Task 4: Implement `/performance` Endpoint

**Files:**
- Modify: `app/services/backend.py` (add after `/chart_data` endpoint)

**Step 1: Add the endpoint**

Add after the `get_chart_data` function:

```python
@app.post("/performance")
def get_performance(req: PerformanceRequest):
    """
    Returns detailed performance stats, returns breakdown, and drawdown series.
    """
    try:
        # 1. Fetch portfolio data
        port_series, stats, _ = api.fetch_backtest(
            start_date=req.start_date,
            end_date=req.end_date if req.end_date else str(dt.date.today()),
            start_val=req.start_val,
            cashflow=req.cashflow,
            cashfreq=req.cashflow_freq,
            rolling=1,
            invest_div=req.invest_div,
            rebalance=req.rebalance_freq,
            allocation=req.tickers
        )

        if port_series.empty:
            raise HTTPException(status_code=400, detail="No data returned for portfolio")

        # 2. Calculate detailed stats
        portfolio_stats = calculations.generate_stats(port_series)

        # Merge with API stats (API may have additional fields)
        merged_stats = {
            "cagr": safe_float(portfolio_stats.get("cagr", stats.get("cagr", 0))),
            "sharpe": safe_float(portfolio_stats.get("sharpe", stats.get("sharpe", 0))),
            "sortino": safe_float(stats.get("sortino", 0)),
            "max_drawdown": safe_float(portfolio_stats.get("max_drawdown", stats.get("max_drawdown", 0))),
            "volatility": safe_float(portfolio_stats.get("std", stats.get("std", 0))),
            "best_year": safe_float(portfolio_stats.get("best_year", stats.get("best_year", 0))),
            "worst_year": safe_float(portfolio_stats.get("worst_year", stats.get("worst_year", 0)))
        }

        response = {"stats": merged_stats}

        # 3. Fetch and calculate benchmark stats
        benchmark_stats = None
        benchmark_series = None

        if req.benchmark:
            try:
                benchmark_series, bench_api_stats, _ = api.fetch_backtest(
                    start_date=req.start_date,
                    end_date=req.end_date if req.end_date else str(dt.date.today()),
                    start_val=req.start_val,
                    cashflow=0,
                    cashfreq="Monthly",
                    rolling=1,
                    invest_div=True,
                    rebalance="None",
                    allocation={req.benchmark: 100}
                )

                if not benchmark_series.empty:
                    bench_calc_stats = calculations.generate_stats(benchmark_series)
                    benchmark_stats = {
                        "cagr": safe_float(bench_calc_stats.get("cagr", 0)),
                        "sharpe": safe_float(bench_calc_stats.get("sharpe", 0)),
                        "max_drawdown": safe_float(bench_calc_stats.get("max_drawdown", 0)),
                        "volatility": safe_float(bench_calc_stats.get("std", 0))
                    }
            except Exception as e:
                print(f"Warning: Could not fetch benchmark {req.benchmark}: {e}")

        if benchmark_stats:
            response["benchmark_stats"] = benchmark_stats
            response["alpha"] = {
                "cagr_diff": safe_float(merged_stats["cagr"] - benchmark_stats["cagr"]),
                "sharpe_diff": safe_float(merged_stats["sharpe"] - benchmark_stats["sharpe"]),
                "max_dd_diff": safe_float(merged_stats["max_drawdown"] - benchmark_stats["max_drawdown"])
            }

        # 4. Calculate returns breakdown
        # Monthly
        monthly_data = port_series.resample("ME").last()
        monthly_returns = monthly_data.pct_change() * 100

        monthly_list = []
        if benchmark_series is not None and not benchmark_series.empty:
            bench_monthly = benchmark_series.resample("ME").last().pct_change() * 100
        else:
            bench_monthly = pd.Series(dtype=float)

        for date, val in monthly_returns.items():
            if not pd.isna(val):
                bench_val = bench_monthly.get(date, None)
                monthly_list.append({
                    "year": date.year,
                    "month": date.month,
                    "value": safe_float(val),
                    "benchmark": safe_float(bench_val) if bench_val is not None and not pd.isna(bench_val) else None
                })

        # Quarterly
        quarterly_data = port_series.resample("QE").last()
        quarterly_returns = quarterly_data.pct_change() * 100

        quarterly_list = []
        if benchmark_series is not None and not benchmark_series.empty:
            bench_quarterly = benchmark_series.resample("QE").last().pct_change() * 100
        else:
            bench_quarterly = pd.Series(dtype=float)

        for date, val in quarterly_returns.items():
            if not pd.isna(val):
                bench_val = bench_quarterly.get(date, None)
                quarterly_list.append({
                    "year": date.year,
                    "quarter": (date.month - 1) // 3 + 1,
                    "value": safe_float(val),
                    "benchmark": safe_float(bench_val) if bench_val is not None and not pd.isna(bench_val) else None
                })

        # Yearly
        yearly_data = port_series.resample("YE").last()
        yearly_returns = yearly_data.pct_change() * 100

        yearly_list = []
        if benchmark_series is not None and not benchmark_series.empty:
            bench_yearly = benchmark_series.resample("YE").last().pct_change() * 100
        else:
            bench_yearly = pd.Series(dtype=float)

        for date, val in yearly_returns.items():
            if not pd.isna(val):
                bench_val = bench_yearly.get(date, None)
                yearly_list.append({
                    "year": date.year,
                    "value": safe_float(val),
                    "benchmark": safe_float(bench_val) if bench_val is not None and not pd.isna(bench_val) else None
                })

        response["returns"] = {
            "monthly": monthly_list,
            "quarterly": quarterly_list,
            "yearly": yearly_list
        }

        # 5. Drawdown series
        if req.include_drawdown_series:
            running_max = port_series.cummax()
            drawdown = (port_series / running_max) - 1.0

            dd_dates = dates_to_strings(drawdown.index)
            dd_values = [safe_float(v) for v in drawdown.values]

            response["drawdown_series"] = {
                "dates": dd_dates,
                "portfolio": dd_values
            }

            if benchmark_series is not None and not benchmark_series.empty:
                bench_running_max = benchmark_series.cummax()
                bench_drawdown = (benchmark_series / bench_running_max) - 1.0
                # Align to portfolio dates
                bench_drawdown_aligned = bench_drawdown.reindex(drawdown.index, method='ffill')
                response["drawdown_series"]["benchmark"] = [safe_float(v) for v in bench_drawdown_aligned.values]

        return response

    except HTTPException:
        raise
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
```

**Step 2: Commit**

```bash
git add app/services/backend.py
git commit -m "feat(api): implement /performance endpoint with stats and returns"
```

---

## Task 5: Implement `/tax_rebal` Endpoint

**Files:**
- Modify: `app/services/backend.py` (add after `/performance` endpoint)

**Step 1: Add the endpoint**

Add after the `get_performance` function:

```python
@app.post("/tax_rebal")
def get_tax_rebal(req: TaxRebalRequest):
    """
    Returns tax impact analysis and rebalancing history.
    """
    try:
        # 1. Run shadow backtest to get trades and tax data
        tax_cfg = {
            "method": req.tax_config.method,
            "other_income": req.tax_config.income,
            "filing_status": req.tax_config.filing_status,
            "state_tax_rate": req.tax_config.state_tax_rate,
            "use_std_deduction": req.tax_config.use_std_deduction
        }

        trades_df, pl_by_year, comp_df, unrealized, logs, port_series, twr_series = shadow_backtest.run_shadow_backtest(
            allocation=req.tickers,
            start_val=req.start_val,
            start_date=req.start_date,
            end_date=req.end_date if req.end_date else str(dt.date.today()),
            cashflow=req.cashflow,
            cashflow_freq=req.cashflow_freq,
            rebalance_freq=req.rebalance_freq,
            tax_config=tax_cfg,
            pay_down_margin=False
        )

        response = {}

        # 2. Calculate tax by year
        tax_by_year = []
        total_tax_paid = 0.0
        total_realized_pl = 0.0
        current_loss_cf = 0.0

        if not pl_by_year.empty:
            # Calculate federal tax series with carryforward
            fed_tax_series = tax_library.calculate_tax_series_with_carryforward(
                pl_by_year,
                req.tax_config.income,
                req.tax_config.filing_status,
                method=req.tax_config.method,
                use_standard_deduction=req.tax_config.use_std_deduction
            )

            # Calculate state tax with carryforward
            state_tax_series = pd.Series(0.0, index=fed_tax_series.index)
            loss_cf_state = 0.0

            if isinstance(pl_by_year, pd.DataFrame):
                total_pl_series = pl_by_year["Realized P&L"]
            else:
                total_pl_series = pl_by_year

            for y, pl in total_pl_series.sort_index().items():
                net = pl - loss_cf_state
                if net > 0:
                    state_tax_series[y] = net * req.tax_config.state_tax_rate
                    loss_cf_state = 0.0
                else:
                    loss_cf_state = abs(net)

            # Build year-by-year breakdown
            for year in sorted(fed_tax_series.index):
                realized_pl = total_pl_series.get(year, 0)
                fed_tax = fed_tax_series.get(year, 0)
                state_tax = state_tax_series.get(year, 0)

                # Get short-term vs long-term breakdown if available
                st_pl = 0.0
                lt_pl = 0.0
                if isinstance(pl_by_year, pd.DataFrame):
                    st_pl = pl_by_year.get("Short-Term P&L", pd.Series()).get(year, 0)
                    lt_pl = pl_by_year.get("Long-Term P&L", pd.Series()).get(year, 0)

                tax_by_year.append({
                    "year": int(year),
                    "realized_pl": safe_float(realized_pl),
                    "short_term_pl": safe_float(st_pl),
                    "long_term_pl": safe_float(lt_pl),
                    "federal_tax": safe_float(fed_tax),
                    "state_tax": safe_float(state_tax),
                    "total_tax": safe_float(fed_tax + state_tax),
                    "loss_carryforward": 0.0  # TODO: track actual carryforward
                })

                total_tax_paid += fed_tax + state_tax
                total_realized_pl += realized_pl

        response["tax_by_year"] = tax_by_year

        # Tax summary
        effective_rate = (total_tax_paid / total_realized_pl) if total_realized_pl > 0 else 0
        response["tax_summary"] = {
            "total_tax_paid": safe_float(total_tax_paid),
            "total_realized_pl": safe_float(total_realized_pl),
            "effective_rate": safe_float(effective_rate),
            "current_loss_carryforward": safe_float(current_loss_cf)
        }

        # 3. Trades history
        if req.include_trades and not trades_df.empty:
            trades_list = []
            for _, row in trades_df.iterrows():
                trade_amt = row.get("Trade Amount", 0)
                trades_list.append({
                    "date": row["Date"].strftime('%Y-%m-%d') if pd.notna(row.get("Date")) else None,
                    "ticker": row.get("Ticker", ""),
                    "action": "BUY" if trade_amt > 0 else "SELL",
                    "amount": safe_float(abs(trade_amt)),
                    "realized_pl": safe_float(row.get("Realized P&L", 0)),
                    "holding_period": row.get("Holding Period", "unknown")
                })
            response["trades"] = trades_list
        else:
            response["trades"] = []

        # 4. Composition snapshots
        if req.include_composition and not comp_df.empty:
            composition_list = []
            for date in comp_df["Date"].unique():
                date_df = comp_df[comp_df["Date"] == date]
                total_val = date_df["Value"].sum()

                holdings = []
                for _, row in date_df.iterrows():
                    weight = row["Value"] / total_val if total_val > 0 else 0
                    holdings.append({
                        "ticker": row["Ticker"],
                        "value": safe_float(row["Value"]),
                        "weight": safe_float(weight)
                    })

                composition_list.append({
                    "date": date.strftime('%Y-%m-%d') if hasattr(date, 'strftime') else str(date),
                    "holdings": holdings,
                    "total_value": safe_float(total_val)
                })
            response["composition"] = composition_list
        else:
            response["composition"] = []

        # 5. Unrealized P&L
        if unrealized is not None and not unrealized.empty:
            unrealized_total = unrealized["Unrealized P&L"].sum() if "Unrealized P&L" in unrealized.columns else 0
            by_ticker = []
            for _, row in unrealized.iterrows():
                by_ticker.append({
                    "ticker": row.get("Ticker", ""),
                    "cost_basis": safe_float(row.get("Cost Basis", 0)),
                    "current_value": safe_float(row.get("Current Value", 0)),
                    "unrealized": safe_float(row.get("Unrealized P&L", 0))
                })
            response["unrealized_pl"] = {
                "total": safe_float(unrealized_total),
                "by_ticker": by_ticker
            }
        else:
            response["unrealized_pl"] = {"total": 0, "by_ticker": []}

        return response

    except HTTPException:
        raise
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
```

**Step 2: Commit**

```bash
git add app/services/backend.py
git commit -m "feat(api): implement /tax_rebal endpoint with tax breakdown and trades"
```

---

## Task 6: Implement `/compare` Endpoint

**Files:**
- Modify: `app/services/backend.py` (add after `/tax_rebal` endpoint)

**Step 1: Add the endpoint**

Add after the `get_tax_rebal` function:

```python
@app.post("/compare")
def compare_portfolios(req: CompareRequest):
    """
    Returns side-by-side comparison of multiple portfolios.
    """
    try:
        if not req.portfolios:
            raise HTTPException(status_code=400, detail="At least one portfolio required")

        if len(req.portfolios) > 10:
            raise HTTPException(status_code=400, detail="Maximum 10 portfolios allowed")

        # 1. Fetch all portfolios
        portfolio_data = []
        all_start_dates = []
        all_end_dates = []

        for pf in req.portfolios:
            try:
                series, stats, _ = api.fetch_backtest(
                    start_date=req.start_date,
                    end_date=req.end_date if req.end_date else str(dt.date.today()),
                    start_val=req.start_val,
                    cashflow=0,
                    cashfreq="Monthly",
                    rolling=1,
                    invest_div=True,
                    rebalance=pf.rebalance_freq,
                    allocation=pf.tickers
                )

                if not series.empty:
                    all_start_dates.append(series.index.min())
                    all_end_dates.append(series.index.max())
                    portfolio_data.append({
                        "name": pf.name,
                        "series": series,
                        "stats": stats
                    })
            except Exception as e:
                print(f"Warning: Could not fetch portfolio {pf.name}: {e}")
                portfolio_data.append({
                    "name": pf.name,
                    "series": pd.Series(dtype=float),
                    "stats": {},
                    "error": str(e)
                })

        # 2. Determine common date range
        if req.align_to_common_start and all_start_dates:
            common_start = max(all_start_dates)
            common_end = min(all_end_dates)
        else:
            common_start = min(all_start_dates) if all_start_dates else None
            common_end = max(all_end_dates) if all_end_dates else None

        # 3. Fetch benchmark
        benchmark_data = None
        if req.benchmark:
            try:
                bench_series, bench_stats, _ = api.fetch_backtest(
                    start_date=req.start_date,
                    end_date=req.end_date if req.end_date else str(dt.date.today()),
                    start_val=req.start_val,
                    cashflow=0,
                    cashfreq="Monthly",
                    rolling=1,
                    invest_div=True,
                    rebalance="None",
                    allocation={req.benchmark: 100}
                )

                if not bench_series.empty:
                    # Align to common range
                    if common_start:
                        bench_series = bench_series[bench_series.index >= common_start]
                    if common_end:
                        bench_series = bench_series[bench_series.index <= common_end]

                    # Rebase to start_val
                    if not bench_series.empty and bench_series.iloc[0] != 0:
                        bench_series = (bench_series / bench_series.iloc[0]) * req.start_val

                    bench_calc_stats = calculations.generate_stats(bench_series)

                    # Drawdown
                    bench_running_max = bench_series.cummax()
                    bench_dd = (bench_series / bench_running_max) - 1.0

                    benchmark_data = {
                        "name": req.benchmark,
                        "stats": {
                            "cagr": safe_float(bench_calc_stats.get("cagr", 0)),
                            "sharpe": safe_float(bench_calc_stats.get("sharpe", 0)),
                            "max_drawdown": safe_float(bench_calc_stats.get("max_drawdown", 0)),
                            "volatility": safe_float(bench_calc_stats.get("std", 0)),
                            "final_value": safe_float(bench_series.iloc[-1]) if not bench_series.empty else 0
                        },
                        "series": {
                            "dates": dates_to_strings(bench_series.index),
                            "values": series_to_list(bench_series)
                        }
                    }
            except Exception as e:
                print(f"Warning: Could not fetch benchmark {req.benchmark}: {e}")

        # 4. Process each portfolio
        processed_portfolios = []
        cagr_ranking = []
        sharpe_ranking = []
        dd_ranking = []

        for pf_data in portfolio_data:
            series = pf_data["series"]
            name = pf_data["name"]

            if series.empty:
                processed_portfolios.append({
                    "name": name,
                    "stats": {},
                    "series": {"dates": [], "values": []},
                    "drawdown": {"dates": [], "values": []},
                    "error": pf_data.get("error")
                })
                continue

            # Align to common range
            if common_start:
                series = series[series.index >= common_start]
            if common_end:
                series = series[series.index <= common_end]

            if series.empty:
                continue

            # Rebase to start_val at common start
            if series.iloc[0] != 0:
                series = (series / series.iloc[0]) * req.start_val

            # Calculate stats
            calc_stats = calculations.generate_stats(series)

            # Drawdown
            running_max = series.cummax()
            drawdown = (series / running_max) - 1.0

            stats_dict = {
                "cagr": safe_float(calc_stats.get("cagr", 0)),
                "sharpe": safe_float(calc_stats.get("sharpe", 0)),
                "max_drawdown": safe_float(calc_stats.get("max_drawdown", 0)),
                "volatility": safe_float(calc_stats.get("std", 0)),
                "final_value": safe_float(series.iloc[-1]) if not series.empty else 0
            }

            processed_portfolios.append({
                "name": name,
                "stats": stats_dict,
                "series": {
                    "dates": dates_to_strings(series.index),
                    "values": series_to_list(series)
                },
                "drawdown": {
                    "dates": dates_to_strings(drawdown.index),
                    "values": series_to_list(drawdown)
                }
            })

            # Track for rankings
            cagr_ranking.append((name, stats_dict["cagr"] or 0))
            sharpe_ranking.append((name, stats_dict["sharpe"] or 0))
            dd_ranking.append((name, stats_dict["max_drawdown"] or 0))

        # Add benchmark to rankings if exists
        if benchmark_data:
            cagr_ranking.append((benchmark_data["name"], benchmark_data["stats"]["cagr"] or 0))
            sharpe_ranking.append((benchmark_data["name"], benchmark_data["stats"]["sharpe"] or 0))
            dd_ranking.append((benchmark_data["name"], benchmark_data["stats"]["max_drawdown"] or 0))

        # Sort rankings
        cagr_ranking.sort(key=lambda x: x[1], reverse=True)
        sharpe_ranking.sort(key=lambda x: x[1], reverse=True)
        dd_ranking.sort(key=lambda x: x[1], reverse=True)  # Less negative = better

        response = {
            "common_start_date": common_start.strftime('%Y-%m-%d') if common_start else None,
            "common_end_date": common_end.strftime('%Y-%m-%d') if common_end else None,
            "portfolios": processed_portfolios,
            "rankings": {
                "by_cagr": [name for name, _ in cagr_ranking],
                "by_sharpe": [name for name, _ in sharpe_ranking],
                "by_max_drawdown": [name for name, _ in dd_ranking]
            }
        }

        if benchmark_data:
            response["benchmark"] = benchmark_data

        return response

    except HTTPException:
        raise
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
```

**Step 2: Commit**

```bash
git add app/services/backend.py
git commit -m "feat(api): implement /compare endpoint for multi-portfolio comparison"
```

---

## Task 7: Manual Testing

**Step 1: Start the API server**

```bash
cd /Users/mcruz/Developer/Testfol-MarginStresser/.worktrees/api-feature-parity
uvicorn app.services.backend:app --reload --port 8000
```

**Step 2: Test each endpoint with curl**

Test `/chart_data`:
```bash
curl -X POST http://localhost:8000/chart_data \
  -H "Content-Type: application/json" \
  -d '{"tickers": {"UPRO": 55, "ZROZ": 45}, "start_date": "2015-01-01", "ma_windows": [200]}'
```

Test `/performance`:
```bash
curl -X POST http://localhost:8000/performance \
  -H "Content-Type: application/json" \
  -d '{"tickers": {"UPRO": 55, "ZROZ": 45}, "start_date": "2015-01-01", "benchmark": "SPYSIM"}'
```

Test `/tax_rebal`:
```bash
curl -X POST http://localhost:8000/tax_rebal \
  -H "Content-Type: application/json" \
  -d '{"tickers": {"UPRO": 55, "ZROZ": 45}, "start_date": "2015-01-01", "tax_config": {"income": 150000, "filing_status": "Single"}}'
```

Test `/compare`:
```bash
curl -X POST http://localhost:8000/compare \
  -H "Content-Type: application/json" \
  -d '{"portfolios": [{"name": "HFEA", "tickers": {"UPRO": 55, "ZROZ": 45}}, {"name": "60/40", "tickers": {"VTI": 60, "BND": 40}}], "start_date": "2015-01-01", "benchmark": "SPYSIM"}'
```

**Step 3: Verify responses match expected structure from design doc**

**Step 4: Final commit**

```bash
git add -A
git commit -m "test: verify all new API endpoints working"
```

---

## Summary

| Task | Description | Files |
|------|-------------|-------|
| 1 | JSON serialization helpers | `app/services/json_utils.py` (new) |
| 2 | Pydantic request models | `app/services/backend.py` (modify) |
| 3 | `/chart_data` endpoint | `app/services/backend.py` (modify) |
| 4 | `/performance` endpoint | `app/services/backend.py` (modify) |
| 5 | `/tax_rebal` endpoint | `app/services/backend.py` (modify) |
| 6 | `/compare` endpoint | `app/services/backend.py` (modify) |
| 7 | Manual testing | N/A |

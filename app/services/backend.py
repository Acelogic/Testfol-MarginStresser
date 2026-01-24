from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, List, Optional
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import numpy as np
import datetime as dt

# Import existing logic
from app.services import testfol_api as api
from app.services import xray_engine
from app.core import shadow_backtest, tax_library, monte_carlo
from app.core import calculations
from app.services.json_utils import safe_float, safe_int, series_to_list, dates_to_strings, safe_dict

app = FastAPI()

# Allow CORS for Expo Web
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class Allocation(BaseModel):
    ticker: str
    percent: float

class BacktestRequest(BaseModel):
    tickers: Dict[str, float] # { "UPRO": 100 }
    margin_debt: float
    margin_rate: float = 8.0 # Annual % (legacy/fixed)
    margin_config: Optional[Dict] = None # { "type": "Tiered", "tiers": [...] }
    
    start_date: str # "2010-01-01"
    end_date: Optional[str] = None # "2025-01-01" or today
    rebalance_freq: str = "Quarterly"
    
    # Global Capital & Cashflow
    cashflow: float = 0.0
    cashflow_freq: str = "Monthly"
    invest_div: bool = True
    pay_down_margin: bool = False

    # Tax Configuration
    pay_tax_mode: str = "None" # "None", "Cash", "Margin"
    income: float = 100000
    filing_status: str = "Single"
    state_tax_rate: float = 0.0
    tax_method: str = "2025_fixed"
    use_std_deduction: bool = True
    
    maintenance_margin: float = 0.25
    pm_enabled: bool = False

class XRayRequest(BaseModel):
    portfolio: Dict[str, float] # { "QQQ": 0.5, "AAPL": 0.5 }
    portfolio_name: str = "Portfolio"

class MonteCarloRequest(BaseModel):
    returns: List[float]
    n_sims: int = 1000
    n_years: int = 10
    initial_val: float = 10000.0
    monthly_cashflow: float = 0.0
    block_size: int = 1


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


@app.get("/")
def health_check():
    return {"status": "ok", "service": "Margin Stresser API"}

@app.post("/xray")
def run_xray(req: XRayRequest):
    try:
        # Expected allocation is {ticker: fractional_weight}
        xray_data = xray_engine.compute_xray(req.portfolio)
        return {
            "portfolio_name": req.portfolio_name,
            "holdings": xray_data.to_dict(orient="records") if not xray_data.empty else []
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/monte_carlo")
def run_monte_carlo(req: MonteCarloRequest):
    try:
        print(f"DEBUG: Running Monte Carlo for returns list of length {len(req.returns)}")
        daily_rets = pd.Series(req.returns)
        mc_results = monte_carlo.run_monte_carlo(
            daily_rets,
            n_sims=req.n_sims,
            n_years=req.n_years,
            initial_val=req.initial_val,
            monthly_cashflow=req.monthly_cashflow,
            block_size=req.block_size
        )
        
        # Safe float conversion for JSON metrics
        metrics = {}
        for k, v in mc_results["metrics"].items():
            try:
                metrics[k] = float(v) if isinstance(v, (np.float64, np.float32, np.integer)) else v
            except:
                metrics[k] = str(v)
        
        return {
            "metrics": metrics,
            "percentiles": mc_results["percentiles"].fillna(0).to_dict(orient="list"),
            "p10": mc_results["percentiles"]["P10"].tolist(),
            "p50": mc_results["percentiles"]["Median"].tolist(),
            "p90": mc_results["percentiles"]["P90"].tolist()
        }
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

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
            stages, slope_series, _ = calculations.analyze_stage(port_series, ma_window=150)

            if stages is not None:
                current_stage = stages.iloc[-1] if not stages.empty else "Unknown"
                current_slope = slope_series.iloc[-1] if slope_series is not None and not slope_series.empty else 0

                # Convert stage history to lists
                stage_timestamps = [int(pd.Timestamp(d).timestamp()) for d in stages.index if pd.notna(d)]
                stage_codes = [v for v, d in zip(stages.tolist(), stages.index) if pd.notna(d)]

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
            # Calculate alpha (handle None values with defaults)
            port_cagr = merged_stats["cagr"] or 0
            port_sharpe = merged_stats["sharpe"] or 0
            port_dd = merged_stats["max_drawdown"] or 0
            bench_cagr = benchmark_stats["cagr"] or 0
            bench_sharpe = benchmark_stats["sharpe"] or 0
            bench_dd = benchmark_stats["max_drawdown"] or 0
            response["alpha"] = {
                "cagr_diff": safe_float(port_cagr - bench_cagr),
                "sharpe_diff": safe_float(port_sharpe - bench_sharpe),
                "max_dd_diff": safe_float(port_dd - bench_dd)
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

        trades_df, pl_by_year, comp_df, unrealized, _, _, _ = shadow_backtest.run_shadow_backtest(
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
                realized_pl = float(total_pl_series.get(year, 0) or 0)
                fed_tax = float(fed_tax_series.get(year, 0) or 0)
                state_tax = float(state_tax_series.get(year, 0) or 0)

                # Get short-term vs long-term breakdown if available
                st_pl = 0.0
                lt_pl = 0.0
                if isinstance(pl_by_year, pd.DataFrame):
                    st_col = pl_by_year.get("Short-Term P&L")
                    lt_col = pl_by_year.get("Long-Term P&L")
                    if st_col is not None:
                        st_pl = float(st_col.get(year, 0) or 0)
                    if lt_col is not None:
                        lt_pl = float(lt_col.get(year, 0) or 0)

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

@app.post("/run_stress_test")
def run_stress_test(req: BacktestRequest):
    try:

        # 3. Simulate Margin
        # simulate_margin(port, starting_loan, rate_annual, draw_monthly, maint_pct)
        # Port series is required.
        
        # 1. Fetch Backtest Data (Main Curve)
        # Use broadest possible window, then simulate
        print(f"DEBUG: Running stress test for {req.tickers}")
        
        try:
            # We use Broad Start to capture historical context for tax/margin
            port_series, stats, extra = api.fetch_backtest(
                start_date=req.start_date,
                end_date=req.end_date if req.end_date else str(dt.date.today()),
                start_val=10000.0, # Scale to standard for calculation
                cashflow=req.cashflow if req.cashflow_freq != 'Yearly' else 0, # Handle yearly injection outside if needed
                cashfreq=req.cashflow_freq,
                rolling=1,
                invest_div=req.invest_div,
                rebalance=req.rebalance_freq,
                allocation=req.tickers
            )
        except Exception as e:
            print(f"ERROR in fetch_backtest: {e}")
            raise HTTPException(status_code=400, detail=f"Data fetch failed: {str(e)}")

        # 2. Simulate Margin
        # Handle Hierarchical/Tiered Margin Config
        margin_input = req.margin_rate
        if req.margin_config:
            margin_input = req.margin_config

        loan_series, equity, equity_pct, usage_pct, _ = api.simulate_margin(
            port=port_series,
            starting_loan=req.margin_debt,
            rate_annual=margin_input,
            draw_monthly=0, # Fixed 0 for now as draw is usually handled via cashflow
            maint_pct=req.maintenance_margin
        )

        # 3. Run Shadow Backtest for Taxes & Detailed Logs
        tax_cfg = {
            "method": req.tax_method,
            "other_income": req.income,
            "filing_status": req.filing_status,
            "state_tax_rate": req.state_tax_rate,
            "use_std_deduction": req.use_std_deduction
        }

        # run_shadow_backtest returns: trades_df, pl_by_year, composition_df, unrealized, logs, portfolio_series, twr_series (7 values)
        trades_df, pl_by_year, comp_df, unrealized, logs, _, _ = shadow_backtest.run_shadow_backtest(
            allocation=req.tickers,
            start_val=10000.0,
            start_date=req.start_date,
            end_date=req.end_date if req.end_date else str(dt.date.today()),
            cashflow=req.cashflow,
            cashflow_freq=req.cashflow_freq,
            rebalance_freq=req.rebalance_freq,
            tax_config=tax_cfg,
            pay_down_margin=req.pay_down_margin
        )
        
        # 4. Assemble Response
        df_result = pd.DataFrame({
            "Portfolio": port_series,
            "Equity": equity,
            "Loan": loan_series,
            "MarginUsage": usage_pct
        }).fillna(0)
        
        # 5. Calculate Monthly Returns
        port = df_result["Portfolio"]
        monthly_data = port.resample("ME").last()
        monthly_returns = monthly_data.pct_change() * 100
        
        # Format for UI: { year: 2023, month: 1, value: 5.2 }
        monthly_returns_list = []
        for date, val in monthly_returns.items():
            if not pd.isna(val):
                monthly_returns_list.append({
                    "year": date.year,
                    "month": date.month,
                    "value": float(val)
                })

        # 5. Calculate Quarterly Returns
        quarterly_data = port.resample("QE").last()
        quarterly_returns = quarterly_data.pct_change() * 100
        
        quarterly_returns_list = []
        for date, val in quarterly_returns.items():
            if not pd.isna(val):
                quarterly_returns_list.append({
                    "year": date.year,
                    "quarter": (date.month - 1) // 3 + 1,
                    "value": float(val)
                })

        # 6. Calculate Yearly Returns
        yearly_data = port.resample("YE").last()
        yearly_returns = yearly_data.pct_change() * 100
        
        yearly_returns_list = []
        for date, val in yearly_returns.items():
            if not pd.isna(val):
                yearly_returns_list.append({
                    "year": date.year,
                    "value": float(val)
                })

        # 7. Calculate Daily Returns
        daily_returns = port.pct_change() * 100
        daily_returns_list = []
        # Return last 250 days to avoid massive payload, or maybe user wants all? 
        # Let's return all for now, but formatted efficiently?
        # Actually for "List", maybe just return all. It's not THAT much data (10 years = 2500 points).
        for date, val in daily_returns.items():
            if not pd.isna(val):
                daily_returns_list.append({
                    "date": date.strftime('%Y-%m-%d'),
                    "value": float(val)
                })

        # 8. Calculate Daily Stats & Histogram
        daily_ret_vals = daily_returns.dropna().values
        # Handle case where daily_ret_vals might be empty
        if len(daily_ret_vals) > 0:
            best_day = float(daily_ret_vals.max()) * 100
            worst_day = float(daily_ret_vals.min()) * 100
            pos_days = float((daily_ret_vals > 0).mean()) * 100
            
            # Histogram (50 bins)
            counts, bin_edges = np.histogram(daily_ret_vals * 100, bins=50)
            daily_histogram = []
            for i in range(len(counts)):
                # simple label: formatted start of bin
                label = f"{bin_edges[i]:.2f}"
                daily_histogram.append({
                    "value": int(counts[i]),
                    "label": label,
                    "frontColor": '#16a34a' if bin_edges[i] >= 0 else '#dc2626' # Conditional coloring
                })
        else:
            best_day = 0.0
            worst_day = 0.0
            pos_days = 0.0
            daily_histogram = []

        daily_stats = {
            "best": best_day,
            "worst": worst_day,
            "positive_pct": pos_days
        }

        return {
            "dates": df_result.index.strftime('%Y-%m-%d').tolist(),
            "equity": df_result["Equity"].tolist(),
            "loan": df_result["Loan"].tolist(),
            "margin_usage": df_result["MarginUsage"].tolist(),
            "stats": stats,
            "monthly_returns": monthly_returns_list,
            "quarterly_returns": quarterly_returns_list,
            "yearly_returns": yearly_returns_list,
            "daily_returns": daily_returns_list,
            "daily_stats": daily_stats,
            "daily_histogram": daily_histogram,
            "logs": logs[:20] # Return first 20 logs
        }

    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

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

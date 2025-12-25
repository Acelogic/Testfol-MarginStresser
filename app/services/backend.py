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

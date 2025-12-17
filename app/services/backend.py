from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, List, Optional
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import numpy as np
import datetime as dt

# Import existing logic
from . import testfol_api as api
from app.core import shadow_backtest
from app.core import tax_library

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
    start_val: float
    margin_debt: float
    margin_rate: float # Annual %
    start_date: str # "2010-01-01"
    end_date: str # "2025-01-01" or today
    rebalance_freq: str = "Quarterly"
    cashflow: float = 0.0
    cashflow_freq: str = "Monthly"
    state_tax: float = 0.0
    filing_status: str = "Single"
    state_tax: float = 0.0
    filing_status: str = "Single"
    income: float = 100000
    maintenance_margin: float = 0.25

@app.get("/")
def health_check():
    return {"status": "ok", "service": "Margin Stresser API"}

@app.post("/run_stress_test")
def run_stress_test(req: BacktestRequest):
    try:
        # 1. Run Shadow Backtest (Price Fetching + Tax Lots)
        # Note: shadow_backtest expects allocation as dict {ticker: amount} if using value, 
        # but here we pass weights. run_shadow_backtest docstring says allocation is dict.
        # Let's verify input. 
        
        # shadow_backtest.run_shadow_backtest params:
        # allocation (dict), start_val, start_date, end_date ...
        
        trades_df, pl_by_year, comp_df, unrealized, logs = shadow_backtest.run_shadow_backtest(
            allocation=req.tickers,
            start_val=req.start_val,
            start_date=req.start_date,
            end_date=req.end_date if req.end_date else str(dt.date.today()),
            cashflow=req.cashflow,
            cashflow_freq=req.cashflow_freq,
            rebalance_freq=req.rebalance_freq
        )
        
        if comp_df.empty:
            raise HTTPException(status_code=400, detail="Backtest failed (no data)")
            
        # 2. Fetch Backtest Data (Main Curve)
        print(f"DEBUG: Fetching backtest for {req.tickers} from {req.start_date} to {req.end_date}")
        
        try:
            port_series, stats, extra = api.fetch_backtest(
                start_date=req.start_date,
                end_date=req.end_date if req.end_date else str(dt.date.today()),
                start_val=req.start_val,
                cashflow=req.cashflow,
                cashfreq=req.cashflow_freq,
                rolling=36,
                invest_div=True,
                rebalance=req.rebalance_freq,
                allocation=req.tickers
            )
        except Exception as e:
            print("ERROR in fetch_backtest:")
            import requests
            if isinstance(e, requests.exceptions.HTTPError):
                 print(f"HTTP Error: {e}")
                 if e.response is not None:
                     print(f"Response Text: {e.response.text}")
            raise e

        # 3. Simulate Margin
        # simulate_margin(port, starting_loan, rate_annual, draw_monthly, maint_pct)
        # Port series is required.
        
        loan_series, equity, equity_pct, usage_pct = api.simulate_margin(
            port=port_series,
            starting_loan=req.margin_debt,
            rate_annual=req.margin_rate,
            draw_monthly=0, # Assuming draw is handled via cashflow param in fetch_backtest? No.
            maint_pct=req.maintenance_margin
        )
        
        # 3. Assemble Response
        # Convert Series to JSON-friendly list
        # Dates, Equity, Loan, Margin Usage
        
        df_result = pd.DataFrame({
            "Portfolio": port_series,
            "Equity": equity,
            "Loan": loan_series,
            "MarginUsage": usage_pct
        }).fillna(0)
        
        # 4. Calculate Monthly Returns
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


import pandas as pd
import numpy as np

def run_monte_carlo(returns_series, n_sims=1000, n_years=10, initial_val=10000, monthly_cashflow=0.0):
    """
    Runs a Monte Carlo simulation using Historical Bootstrap (resampling with replacement).
    Supports optional monthly cashflow injections.
    """
    if returns_series.empty:
        return {"percentiles": pd.DataFrame(), "metrics": {}}

    # Pre-calculate simulation parameters
    n_days = int(n_years * 252) # Trading days
    
    # Extract daily returns as numpy array (drop NaNs)
    daily_rets = returns_series.dropna().values
    if len(daily_rets) == 0:
        return {"percentiles": pd.DataFrame(), "metrics": {}}

    # Generate random indices for bootstrapping
    # Shape: (n_days, n_sims)
    random_indices = np.random.randint(0, len(daily_rets), size=(n_days, n_sims))
    sim_returns = daily_rets[random_indices]
    
    # --- Simulation Engine ---
    if monthly_cashflow == 0:
        # Fast Vectorized Path (No Cashflows)
        sim_paths = initial_val * np.cumprod(1 + sim_returns, axis=0)
        
        # Insert start value
        start_row = np.full((1, n_sims), initial_val)
        sim_paths = np.vstack([start_row, sim_paths])
        
    else:
        # Cashflow Path (Iterative Day-by-Day)
        # We prefer iteration to handle cashflow timing correctly relative to compounding
        sim_paths = np.zeros((n_days + 1, n_sims))
        sim_paths[0] = initial_val
        
        current_vals = np.full(n_sims, initial_val, dtype=float)
        
        # Monthly injection approx every 21 trading days
        rebal_period = 21
        
        for t in range(n_days):
            # Apply Return
            current_vals *= (1 + sim_returns[t])
            
            # Apply Cashflow (End of Month approx)
            if (t + 1) % rebal_period == 0:
                current_vals += monthly_cashflow
            
            sim_paths[t+1] = current_vals
    
    # Calculate Percentiles across simulations (axis 1)
    # p10 = Worst Case (10th percentile)
    # p50 = Median Case
    # p90 = Best Case (90th percentile)
    p10 = np.percentile(sim_paths, 10, axis=1)
    p25 = np.percentile(sim_paths, 25, axis=1) # Moderate Downside
    p50 = np.percentile(sim_paths, 50, axis=1)
    p75 = np.percentile(sim_paths, 75, axis=1) # Moderate Upside
    p90 = np.percentile(sim_paths, 90, axis=1)
    
    # Create Date Index for the projection (Business Days)
    start_date = pd.Timestamp.today()
    future_dates = pd.bdate_range(start=start_date, periods=n_days + 1)
    
    # Package into DataFrame
    df_results = pd.DataFrame({
        "P10": p10,
        "P25": p25,
        "Median": p50,
        "P75": p75,
        "P90": p90
    }, index=future_dates)
    
    # Package Paths (Raw)
    # Transpose so columns are paths (0..n_sims-1)
    # Shape of sim_paths is (n_days+1, n_sims), so it matches index length
    df_paths = pd.DataFrame(sim_paths, index=future_dates)
    
    # --- Advanced Metrics ---
    
    # 1. TWR Calculation (Strategy Performance, Independent of Cashflows)
    # This ensures "Median CAGR" is true TWR, not distorted by contributions
    sim_cum_returns = np.cumprod(1 + sim_returns, axis=0) # Shape: (n_days, n_sims)
    final_twr_multiples = sim_cum_returns[-1, :]
    
    # 2. Max Drawdown (Vectorized) on Portfolio Value (Client Experience)
    peaks = np.maximum.accumulate(sim_paths, axis=0)
    drawdowns = (peaks - sim_paths) / peaks
    max_dd_per_path = np.max(drawdowns, axis=0)
    
    # 3. Summary Metrics
    end_values = sim_paths[-1, :]
    
    # Calculate Total Invested Capital (Base for Multiple)
    # Approx n_days / 21 = months
    n_months = n_days // 21
    total_invested = initial_val + (monthly_cashflow * n_months)
    
    # Percentiles (Portfolio Value)
    median_final = np.median(end_values)
    
    # Percentiles (TWR) - Restored
    median_twr = np.median(final_twr_multiples)
    p10_twr = np.percentile(final_twr_multiples, 10)
    p90_twr = np.percentile(final_twr_multiples, 90)
    
    def calc_cagr_from_multiple(multiple, years):
        if years <= 0: return 0.0
        return (multiple) ** (1/years) - 1

    metrics = {
        "initial_val": initial_val,
        "total_invested": total_invested, # Added
        "median_final": median_final,
        # ... Other Portfolio Value Stats ...
        "prob_loss": np.mean(end_values < initial_val),
        
        # CAGR Stats (Based on TWR)
        "cagr_median": calc_cagr_from_multiple(median_twr, n_years),
        "cagr_p10": calc_cagr_from_multiple(p10_twr, n_years), 
        "cagr_p90": calc_cagr_from_multiple(p90_twr, n_years),
        
        # Risk Stats (Based on Portfolio Value / MWR Experience)
        "max_dd_median": np.median(max_dd_per_path),
        "max_dd_p90": np.percentile(max_dd_per_path, 90), 
        "max_dd_p10": np.percentile(max_dd_per_path, 10), 
    }
    
    return {
        "percentiles": df_results,
        "metrics": metrics,
        "paths": df_paths,
        "path_metrics": {
            "max_dd": max_dd_per_path,
            "final_twr": final_twr_multiples
        }
    }

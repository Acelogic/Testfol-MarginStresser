
import pandas as pd
import numpy as np

def run_monte_carlo(returns_series, n_sims=1000, n_years=10, initial_val=10000, monthly_cashflow=0.0,
                    filter_start_date=None, filter_end_date=None,
                    custom_mean_annual=None, custom_vol_annual=None,
                    block_size=1):
    """
    Runs a Monte Carlo simulation with advanced regime options:
    - Historical Period Filtering
    - Custom Return/Vol Parameters
    - Block Bootstrapping (Preserves autocorrelation)
    """
    if returns_series.empty:
        return {"percentiles": pd.DataFrame(), "metrics": {}}

    # Pre-calculate simulation parameters
    n_days = int(n_years * 252) # Trading days
    
    # --- Mode 2: Custom Parameters (Synthetic) ---
    if custom_mean_annual is not None and custom_vol_annual is not None:
        # Convert annual to daily
        daily_mean = custom_mean_annual / 252.0
        daily_vol = custom_vol_annual / np.sqrt(252.0)
        
        # Generate Gaussian noise (IID)
        sim_returns = np.random.normal(daily_mean, daily_vol, size=(n_days, n_sims))
        
    else:
        # --- Mode 1 & 3: Historical Bootstrap (Filtered or Full) ---
        
        # 1. Date Range Filtering
        data_to_sample = returns_series
        if filter_start_date and filter_end_date:
            try:
                start_ts = pd.to_datetime(filter_start_date)
                end_ts = pd.to_datetime(filter_end_date)
                # Ensure we have data in range
                filtered = returns_series[
                    (returns_series.index >= start_ts) & 
                    (returns_series.index <= end_ts)
                ]
                if not filtered.empty:
                    data_to_sample = filtered
            except Exception:
                pass # Fallback to full series if parsing fails
        
        # Extract daily returns
        daily_rets = data_to_sample.dropna().values
        if len(daily_rets) == 0:
            return {"percentiles": pd.DataFrame(), "metrics": {}}

        # 2. Bootstrapping Logic
        if block_size > 1:
            # --- Circular Block Bootstrapping ---
            # Using Circular Bootstrap ensures equal probability for all data points
            # (including start/end of the series).
            n_samples = len(daily_rets)
            
            # Allow picking ANY start index (0 to n_samples - 1)
            # The block will wrap around using modulo if it hits the end
            num_blocks_needed = (n_days // block_size) + 1
            
            # Generate random start indices
            # Shape: (num_blocks_needed, n_sims)
            block_starts = np.random.randint(0, n_samples, size=(num_blocks_needed, n_sims))
            
            # Create offset array [0, 1, 2, ... block_size-1]
            block_offsets = np.arange(block_size) # Shape: (block_size,)
            
            sim_returns_list = []
            for b in range(num_blocks_needed):
                starts = block_starts[b, :] # Shape: (n_sims,)
                
                # Matrix of raw indices: starts + offset
                # shape: (block_size, n_sims)
                raw_indices = starts[np.newaxis, :] + block_offsets[:, np.newaxis]
                
                # Modulo arithmetic for circular wrapping
                wrapped_indices = raw_indices % n_samples
                
                block_rets = daily_rets[wrapped_indices]
                sim_returns_list.append(block_rets)
                
            # Concatenate all blocks along time axis (axis 0)
            full_sim = np.concatenate(sim_returns_list, axis=0)
            
            # Trim to exact n_days
            sim_returns = full_sim[:n_days, :]

        else:
            # --- Simple IID Bootstrap (Default) ---
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

def run_seasonal_monte_carlo(returns_series, n_sims=1000, initial_val=10000.0, monthly_cashflow=0.0):
    """
    Runs a Seasonal Monte Carlo simulation for a single "Typical Year".
    
    Logic:
    1. Group historical daily returns by Month (1-12).
    2. Simulate a year day-by-day.
    3. If simulation day is in Jan, sample from Jan bucket. If Feb, sample from Feb bucket.
    4. Applies monthly cashflows if provided.
    
    Returns:
    DataFrame of percentiles (P10, P50, P90) indexed by day of year.
    Values are in DOLLARS (Projected Portfolio Value), not normalized multipliers.
    """
    if returns_series.empty:
         return pd.DataFrame()

    # 1. Bucket Returns by Month
    # We use explicit looping for safety or boolean indexing
    # Optimization: Create a list of 12 numpy arrays
    df_rets = returns_series.to_frame(name="ret")
    df_rets["month"] = df_rets.index.month
    
    month_buckets = {}
    for m in range(1, 13):
        # Dropna inside bucket
        vals = df_rets[df_rets["month"] == m]["ret"].dropna().values
        if len(vals) == 0:
            # Fallback: if a month has no data (weird), use full history
            vals = df_rets["ret"].dropna().values
        month_buckets[m] = vals
        
    # 2. Simulate Calendar Year
    # We'll assume 21 trading days per month for simplicity
    days_per_month = 21
    total_days = days_per_month * 12
    
    # Store daily returns for all sims
    # Shape: (total_days, n_sims)
    sim_returns = np.zeros((total_days, n_sims))
    
    current_day = 0
    for m in range(1, 13):
        bucket = month_buckets[m]
        n_days_in_this_month = days_per_month
        
        # Sample for this month
        # Shape: (days_in_month, n_sims)
        # Random choice from bucket
        random_indices = np.random.randint(0, len(bucket), size=(n_days_in_this_month, n_sims))
        monthly_sim_rets = bucket[random_indices]
        
        # Fill results array
        sim_returns[current_day : current_day + n_days_in_this_month, :] = monthly_sim_rets
        current_day += n_days_in_this_month
        
    # 3. Calculate Cumulative Path with Cashflows
    # Shape: (n_days + 1, n_sims) to include start
    sim_paths = np.zeros((total_days + 1, n_sims))
    sim_paths[0, :] = initial_val
    
    for t in range(total_days):
        # 3a. Apply Market Return
        # Value(t+1) = Value(t) * (1 + Ret)
        sim_paths[t+1, :] = sim_paths[t, :] * (1 + sim_returns[t, :])
        
        # 3b. Apply Monthly Cashflow (Approx every 21 days)
        # Inject at the END of the month (after return)
        if (t + 1) % days_per_month == 0:
            sim_paths[t+1, :] += monthly_cashflow
    
    # 4. Calculate Percentiles (on Dollar Values)
    p10 = np.percentile(sim_paths, 10, axis=1)
    p25 = np.percentile(sim_paths, 25, axis=1)
    p50 = np.percentile(sim_paths, 50, axis=1)
    p75 = np.percentile(sim_paths, 75, axis=1)
    p90 = np.percentile(sim_paths, 90, axis=1)
    
    # Index: 0 to 252
    df_result = pd.DataFrame({
        "P10": p10,
        "P25": p25,
        "Median": p50,
        "P75": p75,
        "P90": p90
    })
    
    return df_result

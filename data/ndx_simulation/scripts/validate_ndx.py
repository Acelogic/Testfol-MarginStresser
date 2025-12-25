import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), "../src"))
import config
import chart_style
import price_manager

# Configuration
WEIGHTS_FILE = config.WEIGHTS_FILE
BENCHMARK_TICKER = "QQQ" # Use QQQ (Total Return) instead of ^NDX (Price Return)

def validate():
    print("Loading weights...")
    weights = pd.read_csv(WEIGHTS_FILE)
    weights['Date'] = pd.to_datetime(weights['Date'])
    
    # Get unique tickers
    tickers = weights[weights['IsMapped'] == True]['Ticker'].unique().tolist()
    print(f"Fetching prices for {len(tickers)} mapped tickers + {BENCHMARK_TICKER}...")
    
    # Fetch all prices including benchmark
    tickers_to_fetch = tickers + [BENCHMARK_TICKER]
    
    # Optimize: Chunking not needed for 200 items usually, but good practice
    start_date = weights['Date'].min().strftime('%Y-%m-%d')
    
    try:
        data = price_manager.get_price_data(tickers_to_fetch, start_date)
    except Exception as e:
        print(f"Download failed: {e}")
        return

    # Calculate Synthetic Index
    # We need to simulate the portfolio value over time.
    # Start Value = 100 on first date.
    
    # Align dates
    dates = sorted(weights['Date'].unique())
    
    portfolio_values = [] # (Date, Value)
    current_value = 100.0
    
    # Ensure data index is datetime
    data.index = pd.to_datetime(data.index)
    
    print("Simulating portfolio performance...")
    print(f"Data Columns: {data.columns[:5]}")
    print(f"Weights Sample: {weights.head()}")
    
    # Create a full daily date range for the simulation
    full_dates = data.index
    sim_values = pd.Series(index=full_dates, dtype=float)
    sim_values.iloc[0] = 100.0 # It might not be exactly the first date of weights, but let's assume match for now
    
    # Actually, simpler approach:
    # Iterate through quarters.
    # For each quarter, we have a portfolio of tickers and weights.
    # Hold that portfolio until next quarter.
    # Calculate return of that portfolio.
    
    for i in range(len(dates) - 1):
        start_dt = dates[i]
        end_dt = dates[i+1]
        
        # Get Weights at Start Date
        w_df = weights[weights['Date'] == start_dt]
        
        # Filter for tickers we have prices for
        valid_w = w_df[w_df['Ticker'].isin(data.columns)]
        
        if i == 0:
            print(f"Date: {start_dt}")
            print(f"Total Weights in CSV: {len(w_df)}")
            print(f"Valid Tickers in Prices: {len(valid_w)}")
            if len(valid_w) == 0:
                print(f"Sample Tickers in CSV: {w_df['Ticker'].iloc[:5].tolist()}")
                print(f"Sample Columns in Data: {data.columns[:5].tolist()}")

        if len(valid_w) == 0:
            continue
            
        current_sum = valid_w['Weight'].sum()
        if current_sum == 0:
            continue
            
        # Calculate Missing Weight (Unmapped / Delisted)
        missing_weight = 1.0 - current_sum
        
        # Tickers and Valid Weights (do NOT re-normalize yet)
        port_tickers = valid_w['Ticker'].values
        # Keep absolute weights for contribution calculation
        abs_weights = valid_w['Weight'].values 
        
        # Get Prices
        try:
            # We need start and end prices to calculate return
            price_slice = data.loc[start_dt:end_dt, port_tickers]
            price_slice = price_slice.ffill()
            
            p_start = price_slice.iloc[0]
            p_end = price_slice.iloc[-1]
            
            # Identify valid tickers
            valid_mask = (p_start > 0) & (p_start.notna()) & (p_end.notna())
            
            if not valid_mask.any():
                print(f"Period {start_dt}: No valid pricing.")
                continue
                
            valid_tickers = valid_mask.index[valid_mask].tolist()
            
            # Calculate Return of Mapped Portion
            p_s = p_start[valid_tickers]
            p_e = p_end[valid_tickers]
            w_s = pd.Series(abs_weights, index=port_tickers)[valid_tickers]
            
            used_weight = w_s.sum()
            effective_missing = 1.0 - used_weight
            
            # Calculate weighted return of survivors
            survivor_contrib = (w_s * (p_e / p_s)).sum()
            
            # Calculate Benchmark Return for the period (for proxying missing portion)
            bm_ret = None
            if BENCHMARK_TICKER in data.columns:
                 bm_slice = data.loc[start_dt:end_dt, BENCHMARK_TICKER].ffill()
                 if not bm_slice.empty:
                      bm_s = bm_slice.iloc[0]
                      bm_e = bm_slice.iloc[-1]
                      if pd.notna(bm_s) and bm_s > 0:
                          bm_ret = bm_e / bm_s
            
            # Default fallback for missing return (if we can't solve for it)
            missing_ret = bm_ret if (bm_ret is not None and pd.notna(bm_ret)) else 1.0
            
            if bm_ret is not None and effective_missing > 0.01:
                # Solve for R_M (Implied Return of Missing Stocks) to force match
                implied_R_M = (bm_ret - survivor_contrib) / effective_missing
                
                missing_ret = implied_R_M
                period_factor = bm_ret 
            else:
                period_factor = survivor_contrib + (effective_missing * missing_ret)

            # 5. Generate Daily Series (for Plotting)
            price_valid = price_slice[valid_tickers]
            shares = w_s / p_s
            daily_survivor_val = price_valid.dot(shares)
            dates_period = daily_survivor_val.index
            n_days = len(dates_period)
            
            daily_missing_val = pd.Series(0.0, index=dates_period)
            
            if effective_missing > 0.0001 and n_days > 1:
                miss_start = effective_missing
                miss_end = effective_missing * missing_ret
                
                # Geometric interpolation
                if miss_start > 0 and miss_end > 0:
                    daily_rate = np.power(miss_end / miss_start, 1/(n_days-1))
                    factors = np.power(daily_rate, np.arange(n_days))
                    daily_missing_val[:] = miss_start * factors
                else:
                    daily_missing_val[:] = np.linspace(miss_start, miss_end, n_days)
            elif effective_missing > 0.0001:
                 daily_missing_val[:] = effective_missing

            # Total Daily Value (Relative to Start of Period = 1.0)
            daily_rel = daily_survivor_val + daily_missing_val
            
            # Scale to Global Current Value
            daily_vals_scaled = daily_rel * current_value
            
            sim_values.loc[daily_vals_scaled.index] = daily_vals_scaled
            current_value = daily_vals_scaled.iloc[-1]
            
        except Exception as e:
            print(f"Error in period {start_dt}: {e}")
            pass

    sim_values = sim_values.dropna()
    
    if sim_values.empty:
        print("Simulation yielded no values.")
        return

    # Compare with Benchmark
    if BENCHMARK_TICKER not in data.columns:
        print(f"Benchmark {BENCHMARK_TICKER} data not found.")
        return

    ndx = data[BENCHMARK_TICKER].reindex(sim_values.index)
    
    if ndx.dropna().empty:
         print(f"Benchmark {BENCHMARK_TICKER} contains only NaNs for the simulation period.")
         try:
             print("Refetching benchmark...")
             ndx_data = price_manager.get_price_data([BENCHMARK_TICKER], start_date)
             if isinstance(ndx_data, pd.DataFrame): ndx_data = ndx_data.iloc[:, 0]
             ndx = ndx_data.reindex(sim_values.index)
         except:
             pass

    ndx = ndx.dropna()
    common_idx = sim_values.index.intersection(ndx.index)
    
    if len(common_idx) < 10:
        print("Not enough overlapping data between Simulation and Benchmark.")
        return
        
    sim_values = sim_values.loc[common_idx]
    ndx = ndx.loc[common_idx]

    ndx = ndx / ndx.iloc[0] * sim_values.iloc[0]
    
    # Calculate Stats
    correlation = sim_values.corr(ndx)
    
    r_sim = sim_values.pct_change().dropna()
    r_bm = ndx.pct_change().dropna()
    te = (r_sim - r_bm).std() * np.sqrt(252)
    
    print(f"\n--- Validation Results ---")
    print(f"Correlation: {correlation:.4f}")
    print(f"Tracking Error (Annualized): {te:.2%}")
    print(f"Total Return Sim: {sim_values.iloc[-1]/sim_values.iloc[0] - 1:.2%}")
    print(f"Total Return NDX: {ndx.iloc[-1]/ndx.iloc[0] - 1:.2%}")
    
    # Plot
    chart_style.apply_style()
    plt.figure(figsize=(14, 8))
    ax = plt.gca()
    
    plt.plot(sim_values, label='Reconstructed (From Filings)', linewidth=2.0)
    plt.plot(ndx, label='Nasdaq-100 (^NDX)', linestyle='--', alpha=0.8, color='#555555')
    
    plt.yscale('log')
    chart_style.format_date_axis(ax)
    chart_style.format_y_axis(ax, log=True)
    plt.title(f"Reconstructed Nasdaq-100 vs Official Index\nCorr: {correlation:.4f}, TE: {te:.2%}")
    plt.legend()
    
    out_img = os.path.join(config.RESULTS_DIR, "charts", "ndx_validation.png")
    plt.savefig(out_img, dpi=300, bbox_inches='tight')
    print(f"\nChart saved to {out_img}")

def validate_qbig():
    print("\n\n=== Validating NDXMEGASIM vs QBIG ===")
    
    # 1. Load Simulation
    sim_path = os.path.join(config.BASE_DIR, "..", "NDXMEGA2SIM.csv")
    
    if not os.path.exists(sim_path):
        print(f"Skipping QBIG check: {sim_path} not found.")
        return

    print(f"Loading simulation from {sim_path}...")
    sim_df = pd.read_csv(sim_path)
    sim_df['Date'] = pd.to_datetime(sim_df['Date'])
    sim_df = sim_df.set_index('Date').sort_index()
    
    # Rename for clarity
    if 'Close' in sim_df.columns:
        sim_series = sim_df['Close']
    else:
        sim_series = sim_df.iloc[:, 0]
    
    sim_series.name = "NDXMEGASIM"

    # 2. Fetch Real (QBIG)
    ticker = "QBIG"
    print(f"Fetching real data for {ticker}...")
    try:
        real_df = price_manager.get_price_data([ticker], "2000-01-01")
        
        if isinstance(real_df, pd.Series):
             real_series = real_df
        else:
             if ticker in real_df.columns:
                 real_series = real_df[ticker]
             elif 'Close' in real_df.columns:
                 real_series = real_df['Close']
             else:
                 real_series = real_df.iloc[:, 0]
                 
        real_series.name = "Real (QBIG)"
        real_series.index = pd.to_datetime(real_series.index)
        real_series = real_series.dropna() 
        
    except Exception as e:
        print(f"Error fetching {ticker}: {e}")
        return

    # 3. Align and Compare
    common_idx = sim_series.index.intersection(real_series.index)
    
    if len(common_idx) < 10:
        print("Insufficient overlap between Simulation and Real QBIG.")
        return

    print(f"Comparing overlap: {common_idx.min().date()} to {common_idx.max().date()} ({len(common_idx)} days)")
    
    sim_slice = sim_series.loc[common_idx]
    real_slice = real_series.loc[common_idx]

    # Normalize to 100
    sim_norm = sim_slice / sim_slice.iloc[0] * 100
    real_norm = real_slice / real_slice.iloc[0] * 100

    # Calculate Metrics
    tr_sim = (sim_slice.iloc[-1] / sim_slice.iloc[0]) - 1
    tr_real = (real_slice.iloc[-1] / real_slice.iloc[0]) - 1
    
    corr = sim_slice.corr(real_slice)
    
    ret_sim = sim_slice.pct_change().dropna()
    ret_real = real_slice.pct_change().dropna()
    diff = ret_sim - ret_real
    te = diff.std() * (252 ** 0.5)

    print(f"--> Correlation:       {corr:.4f}")
    print(f"--> Tracking Error:    {te:.2%}")
    print(f"--> Sim Period Return: {tr_sim:.2%}")
    print(f"--> QBIG Period Return:{tr_real:.2%}")
    print(f"--> Difference:        {tr_sim - tr_real:.2%}")

    # Plot
    chart_style.apply_style()
    plt.figure(figsize=(14, 8))
    ax = plt.gca()
    
    plt.plot(sim_norm, label=f'NDX Mega 2.0 Index (Simulated Underlying) ({tr_sim:.1%})', linewidth=2.5)
    plt.plot(real_norm, label=f'QBIG ETF (Real) ({tr_real:.1%})', linestyle='--', linewidth=2.0, color='#C44E52')
    
    chart_style.format_date_axis(ax)
    chart_style.format_y_axis(ax, log=False)
    chart_style.add_watermark(ax, "QBIG Comparison")
    
    plt.title(f"Validation: NDX Mega 2.0 Index vs QBIG ETF\nCorr: {corr:.4f}, TE: {te:.2%}")
    plt.legend()
    
    out_img = os.path.join(config.RESULTS_DIR, "charts", "validation_qbig.png")
    plt.savefig(out_img, dpi=300, bbox_inches='tight')
    print(f"Chart saved to {out_img}")

def compare_strategies():
    print("\n\n=== Comparing NDX Mega 1.0 vs 2.0 ===")
    
    # Paths
    path1 = os.path.join(config.BASE_DIR, "..", "NDXMEGASIM.csv")
    path2 = os.path.join(config.BASE_DIR, "..", "NDXMEGA2SIM.csv")
    
    if not (os.path.exists(path1) and os.path.exists(path2)):
        print("Missing one or both simulation files.")
        return

    # Load
    s1 = pd.read_csv(path1)
    s1['Date'] = pd.to_datetime(s1['Date'])
    s1 = s1.set_index('Date').sort_index().iloc[:, 0]
    s1.name = "Mega 1.0"
    
    s2 = pd.read_csv(path2)
    s2['Date'] = pd.to_datetime(s2['Date'])
    s2 = s2.set_index('Date').sort_index().iloc[:, 0]
    s2.name = "Mega 2.0"
    
    # Align
    idx = s1.index.intersection(s2.index)
    if idx.empty:
        print("No overlap.")
        return
        
    slice1 = s1.loc[idx]
    slice2 = s2.loc[idx]
    
    # Stats
    ret1 = slice1.iloc[-1] / slice1.iloc[0] - 1
    ret2 = slice2.iloc[-1] / slice2.iloc[0] - 1
    
    # Plot
    norm1 = slice1 / slice1.iloc[0] * 100
    norm2 = slice2 / slice2.iloc[0] * 100
    
    chart_style.apply_style()
    plt.figure(figsize=(14, 8))
    ax = plt.gca()
    
    plt.plot(norm1, label=f'NDX Mega 1.0 ({ret1:.0%})', linewidth=2.0)
    plt.plot(norm2, label=f'NDX Mega 2.0 ({ret2:.0%})', linewidth=2.0)
    
    chart_style.format_date_axis(ax)
    chart_style.format_y_axis(ax, log=True)
    chart_style.add_watermark(ax, "Mega vs Mega 2.0")
    
    plt.title(f"Strategy Comparison: NDX Mega 1.0 vs 2.0")
    plt.legend()
    
    out_img = os.path.join(config.RESULTS_DIR, "charts", "ndx_mega_comparison.png")
    plt.savefig(out_img, dpi=300, bbox_inches='tight')
    print(f"Chart saved to {out_img}")

if __name__ == "__main__":
    validate()
    validate_qbig()
    compare_strategies()

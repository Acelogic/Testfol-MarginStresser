import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import numpy as np

# Configuration
WEIGHTS_FILE = "nasdaq_quarterly_weights.csv"
BENCHMARK_TICKER = "^NDX"
MEGA_TARGET_THRESHOLD = 0.47 # Target 47% for selection
MEGA_BUFFER_THRESHOLD = 0.50 # Keep existing if within 50%
SINGLE_STOCK_CAP = 0.35


# Iterative Capping
def apply_caps(w_series, cap):
    w = w_series.copy()
    w = w / w.sum()
    for _ in range(5):
        excess = w[w > cap]
        if excess.empty: break
        surplus = (excess - cap).sum()
        w[w > cap] = cap
        others = w[w < cap]
        if others.empty: break
        w[w < cap] = others + (surplus * others / others.sum())
    return w

def backtest():
    print("Loading data...")
    weights_df = pd.read_csv(WEIGHTS_FILE)
    weights_df['Date'] = pd.to_datetime(weights_df['Date'])
    
    # Get all tickers needed
    tickers = weights_df[weights_df['IsMapped'] == True]['Ticker'].unique().tolist()
    if BENCHMARK_TICKER not in tickers:
        tickers.append(BENCHMARK_TICKER)
        
    print(f"Fetching prices for {len(tickers)} tickers...")
    start_date = weights_df['Date'].min().strftime('%Y-%m-%d')
    try:
        data = yf.download(tickers, start=start_date, auto_adjust=True, progress=True)['Close']
        data.index = pd.to_datetime(data.index)
        print(f"Data Shape: {data.shape}")
    except Exception as e:
        print(f"Error fetching data: {e}")
        return

    # Simulation Variables
    dates = sorted(weights_df['Date'].unique())
    mega_values = pd.Series(index=data.index, dtype=float)
    mega_values.iloc[0] = 100.0 # Index base 100
    current_value = 100.0
    
    # Store constituents count
    constituents_history = []
    current_constituents = set()
    
    print("Simulating NDX Mega strategy...")
    
    for i in range(len(dates) - 1):
        start_dt = dates[i]
        end_dt = dates[i+1]
        
        # 1. Selection Phase (At rebalance date start_dt)
        # Get full NDX composition (Mapped AND Unmapped)
        q_weights = weights_df[weights_df['Date'] == start_dt].copy()
        
        # Sort by Weight Descending (on FULL universe)
        q_weights = q_weights.sort_values(by='Weight', ascending=False)
        q_weights['CumWeight'] = q_weights['Weight'].cumsum()
        
        # Identify Annual Reconstitution (December) vs Quarterly Rebalance
        is_annual_recon = (start_dt.month == 12)
        
        selected_tickers = []
        
        if is_annual_recon or not current_constituents:
            # Annual Reconstitution or First Run: Strict 47% Selection
            # Select from the FULL list until 47% cumulative weight is reached.
            # ONLY add if IsMapped is True (if we pick an unmapped stock, we technically "select" it but can't hold it -> gap).
            # But the methodology says "Top companies".
            # If #10 is unmapped, we select it. We just can't trade it.
            # So our portfolio will hold #1-#9, skip #10, hold #11...
            # The weights of #1-#9 should be re-normalized to 100% of the *Held* portfolio (capped at 35%).
            
            cutoff_mask = q_weights['CumWeight'] <= (MEGA_TARGET_THRESHOLD + 0.10) 
            
            curr_sum = 0.0
            for ticks, w, mapped in zip(q_weights['Ticker'], q_weights['Weight'], q_weights['IsMapped']):
                if curr_sum + w <= MEGA_TARGET_THRESHOLD + 0.01: # Small tolerance
                    if mapped:
                        selected_tickers.append(ticks)
                    # We count the weight towards the threshold regardless of mapping
                    curr_sum += w
                else:
                    break
            
            # Fallback
            if not selected_tickers and not q_weights.empty:
                 # Try to find the first mapped ticker
                 first_mapped = q_weights[q_weights['IsMapped'] == True]
                 if not first_mapped.empty:
                     selected_tickers.append(first_mapped.iloc[0]['Ticker'])
                 
        else:
            # Quarterly Rebalance: Buffer Rules
            # Rank and CumWeight are based on FULL universe.
            
            target_set = set() # Companies in Top 47%
            buffer_set = set() # Companies in Top 50%
            
            for ticks, w, cw, mapped in zip(q_weights['Ticker'], q_weights['Weight'], q_weights['CumWeight'], q_weights['IsMapped']):
                if cw <= MEGA_TARGET_THRESHOLD + 0.05:
                    if mapped: target_set.add(ticks)
                if cw <= MEGA_BUFFER_THRESHOLD + 0.05:
                    if mapped: buffer_set.add(ticks)
            
            next_portfolio = set()
            
            # Logic:
            # Maintain current if in buffer
            for c in current_constituents:
                if c in buffer_set:
                    next_portfolio.add(c)
            
            # Add new if in target
            for t in target_set:
                next_portfolio.add(t)
                
            selected_tickers = list(next_portfolio)
            
        # Update current constituents for next loop
        current_constituents = set(selected_tickers)
        
        if not selected_tickers:
             print(f"Warning: No selection for {start_dt}")
             continue

        # Filter for valid tickers in our price data
        valid_tickers = [t for t in selected_tickers if t in data.columns]
        
        # Prepare subset for weighting
        mega_subset = q_weights[q_weights['Ticker'].isin(valid_tickers)].copy()
        
        if mega_subset.empty:
            continue
            
        # Stats
        constituents_history.append({
            "Date": start_dt,
            "Count": len(mega_subset),
            "Top": mega_subset['Ticker'].iloc[0] if not mega_subset.empty else "",
            "Bottom": mega_subset['Ticker'].iloc[-1] if not mega_subset.empty else "",
            "Type": "Recon" if is_annual_recon else "Rebal"
        })



        # Final Weights (Cap 35%)
        final_weights = apply_caps(mega_subset.set_index('Ticker')['Weight'], SINGLE_STOCK_CAP)
        
        # 3. Perf Simulation (Buy and Hold for Quarter)
        try:
            price_slice = data.loc[start_dt:end_dt, final_weights.index]
            price_slice = price_slice.ffill()
            
            p_start = price_slice.iloc[0]
            valid_mask = (p_start > 0) & (p_start.notna())
            
            # Re-confirm validity at start date data
            if not valid_mask.all():
                # Drop invalid
                valid_tkrs = valid_mask.index[valid_mask].tolist()
                final_weights = final_weights[valid_tkrs]
                final_weights = final_weights / final_weights.sum()
                p_start = p_start[valid_tkrs]
                price_slice = price_slice[valid_tkrs]
            
            shares = final_weights / p_start
            daily_vals = price_slice.dot(shares)
            
            daily_vals_scaled = daily_vals * current_value
            mega_values.loc[daily_vals_scaled.index] = daily_vals_scaled
            current_value = daily_vals_scaled.iloc[-1]
            
        except Exception as e:
            print(f"Error simulating {start_dt}: {e}")

    mega_values = mega_values.dropna()
    
    # Validation against NDX
    if BENCHMARK_TICKER in data.columns:
        ndx = data[BENCHMARK_TICKER].reindex(mega_values.index).ffill()
        ndx = ndx / ndx.iloc[0] * mega_values.iloc[0]
        
        plt.figure(figsize=(12, 6))
        plt.plot(mega_values, label='NDX Mega (Simulated)', linewidth=1.5)
        plt.plot(ndx, label='Nasdaq-100 (^NDX)', linestyle='--', alpha=0.7)
        plt.title('NDX Mega Strategy vs Nasdaq-100 (2000-2025)')
        plt.yscale('log')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig("ndx_mega_backtest.png")
        print("Chart saved to ndx_mega_backtest.png")
        
        # Stats
        tot_ret_mega = mega_values.iloc[-1] / mega_values.iloc[0] - 1
        tot_ret_ndx = ndx.iloc[-1] / ndx.iloc[0] - 1
        print(f"Total Return Mega: {tot_ret_mega:.2%}")
        print(f"Total Return NDX:  {tot_ret_ndx:.2%}")
        
    # Save Constituents History
    pd.DataFrame(constituents_history).to_csv("ndx_mega_constituents.csv", index=False)
    
    # Save Daily Data for Testfol
    # Format: Date, Close (header implied or explicit)
    # Target: ../NDXMEGASIM.csv (Parent directory)
    # Make sure directory exists (it should, as we are in it)
    import os
    
    # Assuming run from data/edgar_parser
    output_path = "../NDXMEGASIM.csv"
    mega_values.name = "Close"
    mega_values.to_csv(output_path, header=True)
    print(f"Saved NDXMEGASIM data to {output_path}")

if __name__ == "__main__":
    backtest()

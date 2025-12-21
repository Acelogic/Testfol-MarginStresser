import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import numpy as np
import os

# Configuration
WEIGHTS_FILE = "nasdaq_quarterly_weights.csv"
BENCHMARK_TICKER = "^NDX"
MEGA_TARGET_THRESHOLD = 0.40 # Target 40% for selection (Mega 2.0)
MEGA_BUFFER_THRESHOLD = 0.45 # Keep existing if within 45% (Mega 2.0)
SINGLE_STOCK_CAP = 0.30      # 30% Cap (Mega 2.0)
MIN_CONSTITUENTS = 9         # 9 Constituent Minimum (Mega 2.0)

# Iterative Capping
def apply_caps(w_series, cap, total_target=1.0):
    w = w_series.copy()
    if w.sum() == 0: return w
    w = (w / w.sum()) * total_target
    
    for _ in range(10):
        excess = w[w > cap]
        if excess.empty: break
        
        surplus = (excess - cap).sum()
        w[w > cap] = cap
        
        others = w[w < cap]
        if others.empty: break
        
        # Redistribute surplus
        w[w < cap] = others + (surplus * others / others.sum())
        
    return w

def backtest():
    print("Loading data...")
    if not os.path.exists(WEIGHTS_FILE):
        print(f"Error: {WEIGHTS_FILE} not found.")
        return
        
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
    current_constituents = set() # Standard selection
    
    print("Simulating NDX Mega 2.0 strategy...")
    
    for i in range(len(dates) - 1):
        start_dt = dates[i]
        end_dt = dates[i+1]
        
        # 1. Selection Phase (At rebalance date start_dt)
        q_weights = weights_df[weights_df['Date'] == start_dt].copy()
        
        # Sort by Weight Descending
        q_weights = q_weights.sort_values(by='Weight', ascending=False)
        q_weights['CumWeight'] = q_weights['Weight'].cumsum()
        
        is_annual_recon = (start_dt.month == 12)
        
        # Standard Selection (Top 40%)
        standard_tickers = []
        curr_sum = 0.0
        
        if is_annual_recon or not current_constituents:
            # Reconstitution or First Run: Strict 40% Selection
            for ticks, w, mapped in zip(q_weights['Ticker'], q_weights['Weight'], q_weights['IsMapped']):
                if curr_sum + w <= MEGA_TARGET_THRESHOLD + 0.01:
                    if mapped:
                        standard_tickers.append(ticks)
                    curr_sum += w
                else:
                    break
        else:
            # Quarterly Rebalance: Buffer Rules (Top 45%)
            target_set = set() # Top 40%
            buffer_set = set() # Top 45%
            for ticks, w, cw, mapped in zip(q_weights['Ticker'], q_weights['Weight'], q_weights['CumWeight'], q_weights['IsMapped']):
                if cw <= MEGA_TARGET_THRESHOLD + 0.01:
                    if mapped: target_set.add(ticks)
                if cw <= MEGA_BUFFER_THRESHOLD + 0.05:
                    if mapped: buffer_set.add(ticks)
            
            next_portfolio = set()
            for c in current_constituents:
                if c in buffer_set:
                    next_portfolio.add(c)
            for t in target_set:
                next_portfolio.add(t)
            standard_tickers = list(next_portfolio)

        # Update current standard constituents
        current_constituents = set(standard_tickers)
        
        # Minimum Security Rule: Fill to 9 if needed
        # We need to filter for mapped and valid tickers for the filler
        valid_mapped_all = q_weights[q_weights['IsMapped'] == True]
        
        selected_tickers = standard_tickers.copy()
        is_min_security_triggered = False
        
        if len(selected_tickers) < MIN_CONSTITUENTS:
            is_min_security_triggered = True
            # Add next largest mapped tickers NOT in the standard set
            remaining = valid_mapped_all[~valid_mapped_all['Ticker'].isin(selected_tickers)]
            needed = MIN_CONSTITUENTS - len(selected_tickers)
            if not remaining.empty:
                fillers = remaining.head(needed)['Ticker'].tolist()
                selected_tickers.extend(fillers)

        if not selected_tickers:
             print(f"Warning: No selection for {start_dt}")
             continue

        # Filter for valid tickers in our price data
        valid_tickers = [t for t in selected_tickers if t in data.columns]
        mega_subset = q_weights[q_weights['Ticker'].isin(valid_tickers)].copy()
        
        if mega_subset.empty:
            continue
            
        # Stats
        constituents_history.append({
            "Date": start_dt,
            "Count": len(mega_subset),
            "Top": mega_subset['Ticker'].iloc[0],
            "BufferRule": is_min_security_triggered,
            "Type": "Recon" if is_annual_recon else "Rebal"
        })

        # Final Weighting
        if not is_min_security_triggered:
            # Normal: Equal Re-weight and 30% Cap
            final_weights = apply_caps(mega_subset.set_index('Ticker')['Weight'], SINGLE_STOCK_CAP)
        else:
            # Standard Tickers get 99%
            # Fillers get 1% total
            standard_subset = mega_subset[mega_subset['Ticker'].isin(standard_tickers)]
            filler_subset = mega_subset[~mega_subset['Ticker'].isin(standard_tickers)]
            
            # Apply caps to standard group (target 0.99)
            if not standard_subset.empty:
                w_standard = apply_caps(standard_subset.set_index('Ticker')['Weight'], SINGLE_STOCK_CAP, total_target=0.99)
            else:
                w_standard = pd.Series()
                
            # Distribute 1% equally to fillers
            if not filler_subset.empty:
                filler_count = len(filler_subset)
                w_filler = pd.Series(0.01 / filler_count, index=filler_subset['Ticker'])
            else:
                w_filler = pd.Series()
                
            final_weights = pd.concat([w_standard, w_filler])

        # Perf Simulation
        try:
            price_slice = data.loc[start_dt:end_dt, final_weights.index]
            price_slice = price_slice.ffill()
            
            p_start = price_slice.iloc[0]
            valid_mask = (p_start > 0) & (p_start.notna())
            
            if not valid_mask.all():
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
    
    # Comparison
    if BENCHMARK_TICKER in data.columns:
        ndx = data[BENCHMARK_TICKER].reindex(mega_values.index).ffill()
        ndx = ndx / ndx.iloc[0] * mega_values.iloc[0]
        
        plt.figure(figsize=(12, 6))
        plt.plot(mega_values, label='NDX Mega 2.0 (Simulated)', linewidth=1.5)
        plt.plot(ndx, label='Nasdaq-100 (^NDX)', linestyle='--', alpha=0.7)
        plt.title('NDX Mega 2.0 Strategy vs Nasdaq-100 (2000-2025)')
        plt.yscale('log')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig("ndx_mega2_backtest.png")
        print("Chart saved to ndx_mega2_backtest.png")
        
        tot_ret_mega = mega_values.iloc[-1] / mega_values.iloc[0] - 1
        tot_ret_ndx = ndx.iloc[-1] / ndx.iloc[0] - 1
        print(f"Total Return Mega 2.0: {tot_ret_mega:.2%}")
        print(f"Total Return NDX:      {tot_ret_ndx:.2%}")
        
    # Save Daily Data for Testfol
    output_path = "../NDXMEGA2SIM.csv"
    mega_values.name = "Close"
    mega_values.to_csv(output_path, header=True)
    print(f"Saved NDXMEGA2SIM data to {output_path}")

if __name__ == "__main__":
    backtest()

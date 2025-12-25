import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import numpy as np
import os
import config
import chart_style
import price_manager
import changes_parser

# Configuration (Loaded from config.py)

# Iterative Capping
def apply_caps(w_series, cap):
    w = w_series.copy()
    if w.sum() == 0: return w
    
    # Normalize first
    w = w / w.sum()
    
    for _ in range(config.MAX_CAP_ITERATIONS):
        excess = w[w > cap]
        if excess.empty: break
        
        surplus = (excess - cap).sum()
        w[w > cap] = cap
        
        others = w[w < cap]
        if others.empty: break
        
        # Redistributions
        w[w < cap] = others + (surplus * others / others.sum())
    
    # Validation Check
    if (w > cap + 0.001).any():
        print(f"Warning: Capping failed to converge for some stocks > {cap}")
    if abs(w.sum() - 1.0) > 0.001:
         print(f"Warning: Weights sum to {w.sum():.4f}, expected 1.0")
         
    return w

def backtest():
    print("Loading data...")
    weights_df = pd.read_csv(config.WEIGHTS_FILE)
    weights_df['Date'] = pd.to_datetime(weights_df['Date'])
    
    # Get all tickers needed
    tickers = weights_df[weights_df['IsMapped'] == True]['Ticker'].unique().tolist()
    
    # Add Tickers from Changes File (to ensure we have data for mid-quarter adds)
    changes_df = changes_parser.load_changes()
    if not changes_df.empty:
        added_tickers = changes_df['Added Ticker'].dropna().unique().tolist()
        tickers.extend(added_tickers)
        
    tickers = list(set(tickers))
    
    if config.BENCHMARK_TICKER not in tickers:
        tickers.append(config.BENCHMARK_TICKER)
        
    print(f"Fetching prices for {len(tickers)} tickers...")
    start_date = weights_df['Date'].min().strftime('%Y-%m-%d')
    
    data = price_manager.get_price_data(tickers, start_date)
    
    if data is None or data.empty:
        print("Data fetch failed.")
        return
        
    data.index = pd.to_datetime(data.index)
    print(f"Data Shape: {data.shape}")

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
            
            # Using MEGA1 constants
            curr_sum = 0.0
            for ticks, w, mapped in zip(q_weights['Ticker'], q_weights['Weight'], q_weights['IsMapped']):
                if curr_sum + w <= config.MEGA1_TARGET_THRESHOLD: # Strict threshold
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
            # Quarterly Rebalance: Swap/Replacement Rules
            # 1. Identify Buffer Set (Top 50%)
            buffer_set = set()
            
            # Map tickers to their current metrics for easy lookup
            ticker_metrics = {} 
            for ticks, w, cw, mapped in zip(q_weights['Ticker'], q_weights['Weight'], q_weights['CumWeight'], q_weights['IsMapped']):
                if mapped:
                    ticker_metrics[ticks] = {'Weight': w, 'CumWeight': cw}
                    if cw <= config.MEGA1_BUFFER_THRESHOLD:
                        buffer_set.add(ticks)
            
            # 2. Check for Dropouts (Current constituents NOT in Buffer)
            dropouts = []
            retention_list = []
            
            for c in current_constituents:
                if c in buffer_set:
                    retention_list.append(c)
                else:
                    dropouts.append(c)
            
            # 3. Apply Swap Logic
            if not dropouts:
                # Methodology: "If there are no current constituents outside of the top 50%, then no constituent changes are made"
                selected_tickers = list(current_constituents)
            else:
                # "Dynamic Threshold is determined as the maximum cumulative weight of current constituents outside the top 50%"
                # Note: Dropouts might not be in q_weights if they were delisted or totally removed from NDX.
                # If they are not in NDX, they are definitely dropouts, but have no CumWeight.
                # In that case, what is the threshold?
                # Presumably, if a stock is gone, we must replace it.
                # If a stock is just low rank, we use its CumWeight.
                
                dropout_cum_weights = []
                for d in dropouts:
                    if d in ticker_metrics:
                        dropout_cum_weights.append(ticker_metrics[d]['CumWeight'])
                    else:
                        # Removed from NDX entirely?
                        # Implicitly infinite threshold? Or simply it triggers replacement.
                        # If a stock is removed, it contributes to the count 'n' but provides no 'threshold' value itself.
                        # We should likely take the Max of the ones that ARE in the index.
                        # If ALL are removed, we likely default to something reasonable (like the last valid one?)
                        # Actually, if a stock is removed, it's effectively at CumWeight 100% (Rank Last).
                        dropout_cum_weights.append(1.0) 
                
                dynamic_threshold = max(dropout_cum_weights) if dropout_cum_weights else 1.0
                
                # "All companies... not already maintained... and are at or below the dynamic threshold are considered"
                candidates = []
                for ticks, metrics in ticker_metrics.items():
                    if ticks not in retention_list:
                        if metrics['CumWeight'] <= dynamic_threshold:
                             candidates.append((ticks, metrics['Weight']))
                
                # Sort by Base Universe Weight Descending
                candidates.sort(key=lambda x: x[1], reverse=True)
                
                # "The top 'n' companies are chosen... where 'n' represents the number of current constituents outside..."
                num_needed = len(dropouts)
                replacements = [x[0] for x in candidates[:num_needed]]
                
                selected_tickers = retention_list + replacements
            
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
        # Final Weights (Cap 35%)
        # Note: We re-normalize to 100% of the HELD portfolio here.
        final_weights = apply_caps(mega_subset.set_index('Ticker')['Weight'], config.MEGA1_SINGLE_STOCK_CAP)

        # Check for Mid-Quarter Changes (Pre-emptive Replacement)
        replacements = changes_parser.get_replacement_map(start_dt, end_dt)
        for old, new in replacements.items():
            if old in final_weights.index:
                # Only swap if we have data for the new one
                if new in data.columns:
                    print(f"  [Rebal Event] Replacing {old} with {new} (Pre-emptive)")
                    # Assign old weight to new ticker
                    final_weights[new] = final_weights[old]
                    final_weights = final_weights.drop(old)
                else:
                    print(f"  Warning: Cannot replace {old} -> {new}: Price data missing for {new}")

        # Stats
        constituents_history.append({
            "Date": start_dt,
            "Count": len(final_weights),
            "Top": final_weights.idxmax() if not final_weights.empty else "N/A",
            "Type": "Recon" if is_annual_recon else "Rebal",
            "Tickers": "|".join(final_weights.index),
            "Weights": "|".join([f"{w:.6f}" for w in final_weights])
        })
        
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
    if config.BENCHMARK_TICKER in data.columns:
        ndx = data[config.BENCHMARK_TICKER].reindex(mega_values.index).ffill()
        ndx = ndx / ndx.iloc[0] * mega_values.iloc[0]
        
        # Apply Styling
        chart_style.apply_style()
        
        plt.figure(figsize=(14, 8))
        ax = plt.gca()
        
        plt.plot(mega_values, label='NDX Mega (Simulated)', linewidth=2.5)
        plt.plot(ndx, label='Nasdaq-100 (^NDX)', linestyle='--', alpha=0.8, color='#555555')
        
        plt.title('NDX Mega Strategy vs Nasdaq-100 (2000-2025)')
        plt.yscale('log')
        plt.legend()
        
        chart_style.format_date_axis(ax)
        chart_style.format_y_axis(ax, log=True)
        chart_style.add_watermark(ax, "NDX Mega 1.0")
        
        out_img = os.path.join(config.RESULTS_DIR, "charts", "ndx_mega_backtest.png")
        plt.savefig(out_img, dpi=300, bbox_inches='tight')
        print(f"Chart saved to {out_img}")
        
        # Stats
        tot_ret_mega = mega_values.iloc[-1] / mega_values.iloc[0] - 1
        tot_ret_ndx = ndx.iloc[-1] / ndx.iloc[0] - 1
        print(f"Total Return Mega: {tot_ret_mega:.2%}")
        print(f"Total Return NDX:  {tot_ret_ndx:.2%}")
        
    # Save Constituents History
    pd.DataFrame(constituents_history).to_csv(os.path.join(config.RESULTS_DIR, "ndx_mega_constituents.csv"), index=False)
    
    # Save Daily Data for Testfol
    # Target: ../NDXMEGASIM.csv (Parent directory of BASE_DIR)
    
    output_path = os.path.join(config.BASE_DIR, "..", "NDXMEGASIM.csv")
    mega_values.name = "Close"
    mega_values.to_csv(output_path, header=True)
    print(f"Saved NDXMEGASIM data to {output_path}")

if __name__ == "__main__":
    backtest()

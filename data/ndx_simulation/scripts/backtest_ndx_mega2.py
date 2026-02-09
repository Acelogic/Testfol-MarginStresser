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
import changes_parser

# Configuration (Loaded from config.py)

# Iterative Capping
def apply_caps(w_series, cap, total_target=1.0):
    w = w_series.copy()
    if w.sum() == 0: return w
    
    # Normalize to total_target first
    w = (w / w.sum()) * total_target
    
    for _ in range(config.MAX_CAP_ITERATIONS):
        excess = w[w > cap]
        if excess.empty: break
        
        surplus = (excess - cap).sum()
        w[w > cap] = cap
        
        others = w[w < cap]
        if others.empty: break
        
        # Redistribute surplus
        w[w < cap] = others + (surplus * others / others.sum())
    
    # Validation Check
    if (w > cap + 0.001).any():
        print(f"Warning: Capping failed to converge for some stocks > {cap}")
    if abs(w.sum() - total_target) > 0.001:
         print(f"Warning: Weights sum to {w.sum():.4f}, expected {total_target}")
         
    return w

def backtest():
    print("Loading data...")
    if not os.path.exists(config.WEIGHTS_FILE):
        print(f"Error: {config.WEIGHTS_FILE} not found.")
        return
        
    weights_df = pd.read_csv(config.WEIGHTS_FILE)
    weights_df['Date'] = pd.to_datetime(weights_df['Date'])
    
    # Get all tickers needed
    tickers = weights_df[weights_df['IsMapped'] == True]['Ticker'].unique().tolist()
    
    # Add Tickers from Changes File
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
    current_constituents = set() # Standard selection
    
    print("Simulating NDX Mega 2.0 strategy...")

    prev_final_weights = None  # Track last good portfolio for carry-forward
    prev_top5 = None  # Track last known good top-5 for composition quality gate

    for i in range(len(dates) - 1):
        start_dt = dates[i]
        end_dt = dates[i+1]

        # 1. Selection Phase (At rebalance date start_dt)
        q_weights = weights_df[weights_df['Date'] == start_dt].copy()

        # Sort by Weight Descending
        q_weights = q_weights.sort_values(by='Weight', ascending=False)
        q_weights['CumWeight'] = q_weights['Weight'].cumsum()

        is_annual_recon = (start_dt.month == 12)

        # Data quality gate: flag quarters where top-5 changed too drastically
        # compared to last known good quarter (catches stale/distorted filing data).
        current_top5 = set(q_weights.head(5)['Ticker'].tolist())

        if prev_top5 is not None:
            overlap = len(current_top5 & prev_top5)
            if overlap < 2:
                if prev_final_weights is not None:
                    print(f"  Q {start_dt.date()}: DISTORTED weights (overlap={overlap}/5 with prev) — carrying forward")
                    # Carry forward previous portfolio
                    final_weights = prev_final_weights.copy()
                    # Still need to simulate performance for this quarter
                    valid_tickers = [t for t in final_weights.index if t in data.columns]
                    final_weights = final_weights[valid_tickers]
                    if final_weights.sum() > 0:
                        final_weights = final_weights / final_weights.sum()
                    constituents_history.append({
                        "Date": start_dt, "Count": len(final_weights),
                        "Top": final_weights.idxmax() if not final_weights.empty else "N/A",
                        "BufferRule": False, "Type": "CarryFwd",
                        "Tickers": "|".join(final_weights.index),
                        "Weights": "|".join([f"{w:.6f}" for w in final_weights])
                    })
                    # Simulate performance with carried-forward weights
                    try:
                        price_slice = data.loc[start_dt:end_dt, final_weights.index].ffill()
                        if not price_slice.empty:
                            p_start = price_slice.iloc[0]
                            valid_mask = (p_start > 0) & (p_start.notna())
                            valid_tkrs = valid_mask.index[valid_mask].tolist()
                            fw = final_weights[valid_tkrs]
                            if fw.sum() > 0:
                                fw = fw / fw.sum()
                            shares = (fw * current_value) / p_start[valid_tkrs]
                            daily_vals = price_slice[valid_tkrs].dot(shares)
                            mega_values.loc[daily_vals.index] = daily_vals
                            current_value = daily_vals.iloc[-1]
                    except Exception as e:
                        print(f"  Error in carry-forward period: {e}")
                    continue
                else:
                    print(f"  Q {start_dt.date()}: DISTORTED weights (overlap={overlap}/5) — no previous portfolio, skipping")
                    continue
        
        # Standard Selection (Top 40%) - Using MEGA2 constants
        standard_tickers = []
        curr_sum = 0.0
        
        if is_annual_recon or not current_constituents:
            # Reconstitution or First Run: Strict 47% Selection
            for ticks, w, mapped in zip(q_weights['Ticker'], q_weights['Weight'], q_weights['IsMapped']):
                if curr_sum < config.MEGA2_TARGET_THRESHOLD: # Include stock that crosses threshold ("at least 47%")
                    if mapped:
                        standard_tickers.append(ticks)
                    curr_sum += w
                else:
                    break
        else:
            # Quarterly Rebalance: Swap/Replacement Rules (Mega 2.0 Methodology)
            # 1. Identify Buffer Set (Top 50%)
            buffer_set = set()
            
            ticker_metrics = {}
            for ticks, w, cw, mapped in zip(q_weights['Ticker'], q_weights['Weight'], q_weights['CumWeight'], q_weights['IsMapped']):
                if mapped:
                    ticker_metrics[ticks] = {'Weight': w, 'CumWeight': cw}
                    if cw <= config.MEGA2_BUFFER_THRESHOLD:
                        buffer_set.add(ticks)
            
            # 2. Check for Dropouts
            dropouts = []
            retention_list = []
            
            for c in current_constituents:
                if c in buffer_set:
                    retention_list.append(c)
                else:
                    dropouts.append(c)
                    
            # 3. Apply Swap Logic
            if not dropouts:
                selected_tickers = list(current_constituents)
            else:
                dropout_cum_weights = []
                for d in dropouts:
                    if d in ticker_metrics:
                        dropout_cum_weights.append(ticker_metrics[d]['CumWeight'])
                    else:
                        dropout_cum_weights.append(1.0)
                        
                dynamic_threshold = max(dropout_cum_weights) if dropout_cum_weights else 1.0
                
                candidates = []
                for ticks, metrics in ticker_metrics.items():
                    if ticks not in retention_list:
                        if metrics['CumWeight'] <= dynamic_threshold:
                            candidates.append((ticks, metrics['Weight']))
                
                candidates.sort(key=lambda x: x[1], reverse=True)
                
                num_needed = len(dropouts)
                replacements = [x[0] for x in candidates[:num_needed]]
                
                selected_tickers = retention_list + replacements
                
            standard_tickers = selected_tickers

        # Update current standard constituents
        current_constituents = set(standard_tickers)
        
        # Minimum Security Rule: Fill to 9 if needed
        valid_mapped_all = q_weights[q_weights['IsMapped'] == True]
        
        selected_tickers = standard_tickers.copy()
        is_min_security_triggered = False
        
        if len(selected_tickers) < config.MEGA2_MIN_CONSTITUENTS:
            is_min_security_triggered = True
            # Add next largest mapped tickers NOT in the standard set
            remaining = valid_mapped_all[~valid_mapped_all['Ticker'].isin(selected_tickers)]
            needed = config.MEGA2_MIN_CONSTITUENTS - len(selected_tickers)
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


        if not is_min_security_triggered:
            # Normal: Equal Re-weight and 30% Cap
            final_weights = apply_caps(mega_subset.set_index('Ticker')['Weight'], config.MEGA2_SINGLE_STOCK_CAP, total_target=1.0)
        else:
            # Minimum Security Rule Active
            # 
            # Official spec says: Standards get 99%, Fillers get 1%
            # BUT: When 3 stocks hit 30% cap, they can only hold 90% max
            # 
            # Pragmatic interpretation (matches QBIG reality - 0.23% tracking):
            # - Apply 30% cap to standards
            # - Whatever standards CAN'T hold (due to caps) flows to fillers
            # - This ensures 100% investment (matching real ETF behavior)
            #
            standard_subset = mega_subset[mega_subset['Ticker'].isin(standard_tickers)]
            filler_subset = mega_subset[~mega_subset['Ticker'].isin(standard_tickers)]
            
            w_standard = pd.Series(dtype=float)
            w_filler = pd.Series(dtype=float)
            
            # Standards: Apply 30% cap, targeting 99% but accepting less if caps prevent it
            if not standard_subset.empty:
                n_standards = len(standard_subset)
                max_possible = n_standards * config.MEGA2_SINGLE_STOCK_CAP
                standard_target = min(0.99, max_possible)
                
                w_standard = apply_caps(
                    standard_subset.set_index('Ticker')['Weight'], 
                    config.MEGA2_SINGLE_STOCK_CAP, 
                    total_target=standard_target
                )
            
            # Fillers: Get whatever remains to reach 100%
            filler_budget = 1.0 - w_standard.sum()
            
            if not filler_subset.empty and filler_budget > 0:
                filler_count = len(filler_subset)
                w_filler = pd.Series(filler_budget / filler_count, index=filler_subset['Ticker'])

            # Combine
            final_weights = pd.concat([w_standard, w_filler])

        


        # Validation of Final Weights
        if abs(final_weights.sum() - 1.0) > 0.001:
             print(f"CRITICAL: Final weights for {start_dt} sum to {final_weights.sum():.4f}")
        # else:
        #      print(f"  Valid sum: {final_weights.sum():.4f}")

        # Save as previous good portfolio for carry-forward
        prev_final_weights = final_weights.copy()
        prev_top5 = current_top5

        # Stats Log
        constituents_history.append({
            "Date": start_dt,
            "Count": len(final_weights),
            "Top": final_weights.idxmax() if not final_weights.empty else "N/A",
            "BufferRule": is_min_security_triggered,
            "Type": "Recon" if is_annual_recon else "Rebal",
            "Tickers": "|".join(final_weights.index),
            "Weights": "|".join([f"{w:.6f}" for w in final_weights])
        })

        # 3. Perf Simulation (Event-Driven)
        quarter_changes = changes_parser.get_changes_between(start_dt, end_dt)
        
        # Build timeline
        timeline = sorted(list(set([start_dt] + quarter_changes['Date'].tolist() + [end_dt])))
        timeline = [d for d in timeline if start_dt <= d <= end_dt]
        
        # Ensure we have at least start and end
        if len(timeline) < 2: timeline = [start_dt, end_dt]
        if timeline[0] != start_dt: timeline.insert(0, start_dt)
        if timeline[-1] != end_dt: timeline.append(end_dt)
        timeline = sorted(list(set(timeline)))

        curr_w = final_weights.copy()
        dropped_this_quarter = set()
        
        for k in range(len(timeline) - 1):
            sub_start = timeline[k]
            sub_end = timeline[k+1]
            
            if sub_start >= sub_end: continue
            
            try:
                # Slice logic [sub_start, sub_end]
                price_slice = data.loc[sub_start:sub_end, curr_w.index]
                if price_slice.empty: continue
                
                price_slice = price_slice.ffill()
                
                p_start = price_slice.iloc[0]
                valid_mask = (p_start > 0) & (p_start.notna())
                
                if not valid_mask.all():
                    valid_tkrs = valid_mask.index[valid_mask].tolist()
                    curr_w = curr_w[valid_tkrs]
                    if curr_w.sum() > 0: curr_w = curr_w / curr_w.sum()
                    p_start = p_start[valid_tkrs]
                    price_slice = price_slice[valid_tkrs]
                
                if curr_w.empty: continue
                
                shares = (curr_w * current_value) / p_start
                daily_vals = price_slice.dot(shares)
                
                mega_values.loc[daily_vals.index] = daily_vals
                current_value = daily_vals.iloc[-1]
                
            except Exception as e:
                print(f"Error simulating sub-period {sub_start} to {sub_end}: {e}")

            # --- Handle Logic at Event Date (sub_end) ---
            if sub_end != end_dt:
                # Event Logic
                todays_changes = quarter_changes[quarter_changes['Date'] == sub_end]
                
                dropped_any = False
                for _, row in todays_changes.iterrows():
                     removed = row.get('Removed Ticker')
                     if pd.notna(removed) and removed in curr_w.index:
                         print(f"  [Event {sub_end.date()}] Dropping {removed}")
                         curr_w = curr_w.drop(removed)
                         dropped_any = True
                     
                     if pd.notna(removed):
                         dropped_this_quarter.add(removed)
                
                if dropped_any:
                    # Check < 9 Rule
                    current_tickers = curr_w.index.tolist()
                    needed = config.MEGA2_MIN_CONSTITUENTS - len(current_tickers)
                    
                    if needed > 0:
                        print(f"    Count fell to {len(current_tickers)}. Finding {needed} replacements...")
                        
                        # Exclude ALL tickers dropped so far this quarter (including today)
                        # We track this in 'dropped_this_quarter' set maintained in the outer loop
                        
                        # Candidates from q_weights (excluding current AND any dropped this quarter)
                        candidates_df = q_weights[
                            (~q_weights['Ticker'].isin(current_tickers)) & 
                            (~q_weights['Ticker'].isin(dropped_this_quarter))
                        ]
                        candidates_df = candidates_df[candidates_df['IsMapped'] == True]
                        
                        found_count = 0
                        for t in candidates_df['Ticker']:
                            if t in data.columns:
                                print(f"      Selected replacement: {t}")
                                current_tickers.append(t)
                                found_count += 1
                            if found_count >= needed: break
                    
                    # Recalculate Weights (Mini-Rebalance)
                    # Classify Standards vs Fillers
                    # Update standard_tickers (remove drops)
                    standard_tickers = [t for t in standard_tickers if t in current_tickers]
                    
                    # Note: Any replacement we just added is effectively a "Filler"?
                    # PDF: "The Index Securities that were selected... are kept... The Minimum Security Rule is applied... The Minimum Security Weighting Process is applied."
                    # This implies replacements are Fillers (1% pool).
                    # 'standard_tickers' tracks the original elite set.
                    
                    # Prepare subset data
                    # We need 'Weight' data for the new set.
                    # Use q_weights for the relative weights.
                    mega_subset = q_weights[q_weights['Ticker'].isin(current_tickers)].copy()
                    
                    # Re-Run Weighting Logic
                    is_min_security_triggered_now = (len(current_tickers) < config.MEGA2_MIN_CONSTITUENTS) # Should be false now unless we ran out of candidates
                    
                    # Logic block copied from main loop (simplified)
                    # Standards
                    standard_subset = mega_subset[mega_subset['Ticker'].isin(standard_tickers)]
                    # Fillers
                    filler_subset = mega_subset[~mega_subset['Ticker'].isin(standard_tickers)]
                    
                    w_standard = pd.Series(dtype=float)
                    w_filler = pd.Series(dtype=float)
                    
                    # Apply Logic
                    if not standard_subset.empty:
                        n_standards = len(standard_subset)
                        max_possible = n_standards * config.MEGA2_SINGLE_STOCK_CAP
                        standard_target = min(0.99, max_possible)
                        
                        w_standard = apply_caps(
                            standard_subset.set_index('Ticker')['Weight'], 
                            config.MEGA2_SINGLE_STOCK_CAP, 
                            total_target=standard_target
                        )
                    
                    filler_budget = 1.0 - w_standard.sum()
                    if not filler_subset.empty and filler_budget > 0:
                        filler_count = len(filler_subset)
                        w_filler = pd.Series(filler_budget / filler_count, index=filler_subset['Ticker'])

                    curr_w = pd.concat([w_standard, w_filler])
                    
                    # Validation
                    if abs(curr_w.sum() - 1.0) > 0.001:
                         print(f"    Warning: Re-calc weights sum to {curr_w.sum():.4f}")

    mega_values = mega_values.dropna()
    
    # Comparison
    if config.BENCHMARK_TICKER in data.columns:
        benchmark_data = data[config.BENCHMARK_TICKER].reindex(mega_values.index)
        
        # Determine first valid index for scaling
        first_valid_idx = benchmark_data.first_valid_index()
        if first_valid_idx:
            initial_val = benchmark_data.loc[first_valid_idx]
            if pd.notna(initial_val) and initial_val != 0:
                benchmark_curve = (benchmark_data / initial_val) * 100
                
                # Fill NaNs before the first valid index with 100 (flat start)
                benchmark_curve = benchmark_curve.fillna(100)
                
                # Apply Styling
                chart_style.apply_style()
                
                plt.figure(figsize=(14, 8))
                ax = plt.gca()
                
                plt.plot(mega_values.index, (mega_values / mega_values.iloc[0]) * 100, label='NDX Mega 2.0 (Simulated)', linewidth=2.5)
                plt.plot(benchmark_curve.index, benchmark_curve, label=f"{config.BENCHMARK_TICKER} (Total Return)", color='black', alpha=0.6, linestyle='--')
                
                plt.title('NDX Mega 2.0 Strategy vs Nasdaq-100 (2000-2025)')
                plt.yscale('log')
                plt.legend()
                
                chart_style.format_date_axis(ax)
                chart_style.format_y_axis(ax, log=True)
                chart_style.add_watermark(ax, "NDX Mega 2.0")
                
                out_img = os.path.join(config.RESULTS_DIR, "charts", "ndx_mega2_backtest.png")
                plt.savefig(out_img, dpi=300, bbox_inches='tight')
                print(f"Chart saved to {out_img}")
                
                tot_ret_mega = (mega_values.iloc[-1] / mega_values.iloc[0]) - 1
                final_bench = benchmark_curve.iloc[-1]
                print(f"Total Return Mega 2.0: {tot_ret_mega:.2%}")
                print(f"Total Return {config.BENCHMARK_TICKER}: {final_bench - 100:.2f}%")
            else:
                 print(f"Warning: Benchmark {config.BENCHMARK_TICKER} has 0 or NaN initial value.")
        else:
            print(f"Warning: Benchmark {config.BENCHMARK_TICKER} has NO valid data.")
    else:
        print(f"Warning: Benchmark ticker {config.BENCHMARK_TICKER} not found in price data.")
        
        # If no benchmark, still plot the mega_values
        chart_style.apply_style()
        plt.figure(figsize=(14, 8))
        ax = plt.gca()
        plt.plot(mega_values.index, (mega_values / mega_values.iloc[0]) * 100, label='NDX Mega 2.0 (Simulated)', linewidth=2.5)
        plt.title('NDX Mega 2.0 Strategy (2000-2025)')
        plt.yscale('log')
        plt.legend()
        
        chart_style.format_date_axis(ax)
        chart_style.format_y_axis(ax, log=True)
        chart_style.add_watermark(ax, "NDX Mega 2.0")
        
        out_img = os.path.join(config.RESULTS_DIR, "charts", "ndx_mega2_backtest.png")
        plt.savefig(out_img, dpi=300, bbox_inches='tight')
        print(f"Chart saved to {out_img}")
        
        tot_ret_mega = mega_values.iloc[-1] / mega_values.iloc[0] - 1
        tot_ret_ndx = ndx.iloc[-1] / ndx.iloc[0] - 1
        print(f"Total Return Mega 2.0: {tot_ret_mega:.2%}")
        print(f"Total Return NDX:      {tot_ret_ndx:.2%}")
        
    # Save Constituents History
    pd.DataFrame(constituents_history).to_csv(os.path.join(config.RESULTS_DIR, "ndx_mega2_constituents.csv"), index=False)
        
    # Save Daily Data for Testfol
    output_path = os.path.join(config.BASE_DIR, "..", "NDXMEGA2SIM.csv")
    mega_values.name = "Close"
    mega_values.to_csv(output_path, header=True)
    print(f"Saved NDXMEGA2SIM data to {output_path}")

if __name__ == "__main__":
    backtest()

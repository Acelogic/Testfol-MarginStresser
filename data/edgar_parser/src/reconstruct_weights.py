import pandas as pd
import json
import yfinance as yf
import numpy as np
import datetime
import os
import config
import changes_parser
import calendar

# Configuration
INPUT_CSV = config.COMPONENTS_FILE
MAPPING_FILE = os.path.join(config.ASSETS_DIR, "name_mapping.json")
OUTPUT_FILE = config.WEIGHTS_FILE
PROXY_TICKER = "QQQ"

def load_data():
    print("Loading data...")
    df = pd.read_csv(INPUT_CSV)
    df['Date'] = pd.to_datetime(df['Date'])
    
    with open(MAPPING_FILE, "r") as f:
        mapping = json.load(f)
    
    # --- Inject Missing Filings from Changes ---
    changes = changes_parser.load_changes()
    if not changes.empty:
        # Identify Additions
        additions = changes[pd.notna(changes['Added Ticker'])].copy()
        
        # Check if we have data for them
        known_tickers = set(mapping.values())
        
        # We also need to check if we already have a filing for them near the change date
        # But for now, let's just inject if the Ticker is completely unknown OR if it's a recent addition (2025)
        
        for _, row in additions.iterrows():
            ticker = row['Added Ticker']
            date = row['Date']
            name = row['Added Security'] if pd.notna(row['Added Security']) else ticker
            
            # Filter for recent relevant changes (e.g. 2025)
            if date.year < 2025: continue
            
            # Check if we already have filing data?
            # It's hard to check efficiently without scanning DF. 
            # But let's assume if it's in 'changes', we might need it.
            # Especially for Dec 2025.
            
            # Fetch Data if missing from DF for this period?
            # Simplest: Just inject. If duplicate, sorting/dedup logic later handles it (mostly).
            # But fetching is slow. Only fetch if missing.
            
            # Optimization: Only for Dec 2025 changes?
            if date.month == 12 and date.year == 2025:
                print(f"Injecting missing filing for {ticker} ({date.date()})...")
                try:
                    t = yf.Ticker(ticker)
                    info = t.info
                    shares = info.get('sharesOutstanding')
                    
                    # Fetch Price for Value
                    # We need price at 'Date' (Change Date)
                    # Use history
                    hist = t.history(start=date, end=date + pd.Timedelta(days=5))
                    if not hist.empty and shares:
                        price = hist['Close'].iloc[0]
                        value = shares * price
                        
                        # Add to DF
                        new_row = {
                            'Date': date,
                            'Company': name,
                            'Shares': shares, # Note: DF usually has string with commas, but numeric works
                            'Value': value     # formatting?
                        }
                        
                        # Add MAPPING
                        mapping[name] = ticker
                        
                        df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
                except Exception as e:
                    print(f"Failed to inject {ticker}: {e}")

    return df, mapping

def get_unique_tickers(mapping):
    tickers = list(set(mapping.values()))
    if PROXY_TICKER not in tickers:
        tickers.append(PROXY_TICKER)
    return tickers

def fetch_prices(tickers):
    print(f"Fetching prices for {len(tickers)} tickers...")
    # Chunking to avoid massive requests? YF handles mass download well.
    # But URL length limit might be an issue.
    # 270 tickers is fine.
    
    start_date = "1999-01-01"
    
    try:
        # Download Adj Close only
        data = yf.download(tickers, start=start_date, auto_adjust=True, threads=True, progress=True)['Close']
        return data
    except Exception as e:
        print(f"Error fetching prices: {e}")
        return pd.DataFrame()

def get_rebalance_dates(start_year=2000, end_year=2025):
    """
    Returns the 3rd Friday of Mar, Jun, Sep, Dec for each year.
    """
    dates = []
    # Generate dates up to recent future
    current_year = start_year
    target_months = [3, 6, 9, 12]
    
    while current_year <= end_year + 1: # Go a bit further to catch upcoming
        for month in target_months:
            if current_year > end_year and month > 3: break # Limit
            
            # Find 3rd Friday
            c = calendar.Calendar(firstweekday=calendar.SUNDAY)
            monthcal = c.monthdatescalendar(current_year, month)
            fridays = [day for week in monthcal for day in week if day.weekday() == calendar.FRIDAY and day.month == month]
            
            if len(fridays) >= 3:
                date_3rd_fri = fridays[2]
                dates.append(pd.Timestamp(date_3rd_fri))
        current_year += 1
        
    # Filter to not exceed now + 90 days too much
    limit = pd.Timestamp.now() + pd.Timedelta(days=90)
    dates = [d for d in dates if d <= limit]
    return dates

def reconstruct():
    df, mapping = load_data()
    tickers = get_unique_tickers(mapping)
    prices = fetch_prices(tickers)
    
    if prices.empty:
        print("No price data. Aborting.")
        return

    # Ensure index is Datetime
    prices.index = pd.to_datetime(prices.index)
    
    # Resample prices to daily if needed (fill fwd) to handle weekend quarters
    prices = prices.ffill()
    
    quarters = get_rebalance_dates()
    final_rows = []
    
    # Proxy Returns
    if PROXY_TICKER in prices.columns:
        proxy_prices = prices[PROXY_TICKER]
    else:
        proxy_prices = pd.Series(1.0, index=prices.index)

    print(f"Reconstructing weights for {len(quarters)} quarters...")
    
    for q_date in quarters:
        # 1. Find latest preceding filing
        # mask = df['Date'] <= q_date
        # But we want the latest AVAILABLE filing. 
        # A filing dated 2000-06-30 is available ON 2000-06-30? Yes.
        
        # 1. Find latest preceding filing
        # Must be within reasonable window (e.g. 400 days) to avoid zombie filings from delisted cos,
        # but allow for Annual-only filing cadence in the dataset.
        lookback_window = pd.Timedelta(days=400)
        valid_filings = df[(df['Date'] <= q_date) & (df['Date'] >= q_date - lookback_window)]
        
        if valid_filings.empty:
            continue
            
        if valid_filings.empty:
            continue
            
        # Fix: Take latest filing for EACH ticker, not just the single max date for the whole group.
        # "Ticker" is not yet in the DF, so use "Company".
        # Sort by Date descending to keep the latest.
        filing_data = valid_filings.sort_values('Date', ascending=False).drop_duplicates('Company', keep='first').copy()
        
        # 2. Project Value
        # We need to project each holding from its Specific Filing Date to the Quarter End Date.
        
        idx_quarter = prices.index.get_indexer([q_date], method='pad')[0]
        if idx_quarter == -1:
            continue
        date_q = prices.index[idx_quarter]
        
        # Pre-calculate Proxy Return for the Quarter Date? 
        # No, Proxy Return depends on the start date (filing date), which varies.
        # But efficiently, most filings are on the same few dates.
        # We can handle it per row.

        for _, row in filing_data.iterrows():
            name = row['Company']
            fil_date = row['Date'] # The specific filing date for this company
            
            # Skip invalid rows
            try:
                val_f = float(str(row['Value']).replace(',',''))
            except:
                continue

            ticker = mapping.get(name)
            
            # Find Price Index for this specific filing date
            if fil_date not in prices.index:
                # Approximate lookup
                idx_f = prices.index.get_indexer([fil_date], method='pad')[0]
                if idx_f == -1:
                    date_f = fil_date # Fallback, probably won't have price
                else:
                    date_f = prices.index[idx_f]
            else:
                date_f = fil_date
            
            val_q = val_f 
            
            if ticker and ticker in prices.columns:
                try:
                    p_f = prices.at[date_f, ticker]
                    p_q = prices.at[date_q, ticker]
                    
                    if pd.notna(p_f) and pd.notna(p_q) and p_f > 0:
                        ret = p_q / p_f
                        val_q = val_f * ret
                    else:
                        # Fallback to proxy
                        # Calculate proxy ret for this specific period
                        try:
                            px_f = proxy_prices.at[date_f]
                            px_q = proxy_prices.at[date_q]
                            if px_f > 0:
                                val_q = val_f * (px_q / px_f)
                        except:
                            pass
                except:
                     pass
            else:
                 # Unmapped: Proxy
                 try:
                    px_f = proxy_prices.at[date_f]
                    px_q = proxy_prices.at[date_q]
                    if px_f > 0:
                        val_q = val_f * (px_q / px_f)
                 except:
                    pass
            
            final_rows.append({
                "Date": q_date.date(),
                "Ticker": ticker if ticker else name, # Use Name if no ticker
                "Name": name,
                "Value": val_q,
                "IsMapped": bool(ticker)
            })

    # Convert to DataFrame
    res_df = pd.DataFrame(final_rows)
    
    # Aggregate duplicate tickers (e.g. merger variations or same ticker mappings)
    # Sum 'Value' for same ('Date', 'Ticker')
    # Keep metadata from first occurrence
    if not res_df.empty:
        res_df = res_df.groupby(['Date', 'Ticker'], as_index=False).agg({
            'Value': 'sum',
            'Name': 'first',
            'IsMapped': 'first'
        })
    
    # Calculate Weights per Date
    # GroupBy Date sum
    print("Calculating final weights...")
    sums = res_df.groupby('Date')['Value'].transform('sum')
    res_df['Weight'] = res_df['Value'] / sums
    
    # ---------------------------------------------------------
    # APPLY NDX CAPPING RULES (Methodology_NDX.pdf)
    # To ensure the "Reconstructed" weights represent the Index at Rebalance.
    # 
    # Stage 1:
    # If any weight > 24%, cap all > 24 to 20%.
    #
    # Stage 2:
    # If Sum(weights > 4.5%) > 48%:
    #   Cap Aggregate to 40%.
    # ---------------------------------------------------------
    
    def apply_ndx_capping(group, is_annual=False):
        # Working with Series
        w = group['Weight'].values.copy()
        tickers = group['Ticker'].values.copy()
        
        # We need to preserve mapping to return series
        w_series = pd.Series(w, index=tickers)
        
        # Iterative solver (Methodology says "process is repeated until... meet constraints")
        for _ in range(20): 
            # Normalize first
            w_series = w_series / w_series.sum()
            
            if not is_annual:
                # --- QUARTERLY RULES ---
                
                # Stage 1
                max_w = w_series.max()
                if max_w > 0.24:
                    excess = w_series[w_series > 0.20] # Target is 20%
                    if not excess.empty:
                        surplus = (excess - 0.20).sum()
                        w_series[w_series > 0.20] = 0.20
                        
                        # Redistribute to proportional to others
                        others = w_series[w_series <= 0.20]
                        if not others.empty:
                             w_series[w_series <= 0.20] = others + (surplus * others / others.sum())
                
                # Stage 2
                large_caps = w_series[w_series > 0.045]
                agg_weight = large_caps.sum()
                
                if agg_weight > 0.48:
                    # Target aggregate is 40%
                    scale_factor = 0.40 / agg_weight
                    w_series[w_series > 0.045] = w_series[w_series > 0.045] * scale_factor
                    
                    surplus = agg_weight - 0.40
                    small_caps = w_series[w_series <= 0.045]
                    if not small_caps.empty:
                        w_series[w_series <= 0.045] = small_caps + (surplus * small_caps / small_caps.sum())
                
                # Check constraints
                valid_stage1 = w_series.max() <= 0.24
                valid_stage2 = w_series[w_series > 0.045].sum() <= 0.48
                
                if valid_stage1 and valid_stage2:
                    break
            
            else:
                # --- ANNUAL RULES (December) ---
                
                # Stage 1: Cap > 15% -> 14%
                if w_series.max() > 0.15:
                    excess = w_series[w_series > 0.14]
                    if not excess.empty:
                        surplus = (excess - 0.14).sum()
                        w_series[w_series > 0.14] = 0.14
                        
                        others = w_series[w_series <= 0.14]
                        if not others.empty:
                             w_series[w_series <= 0.14] = others + (surplus * others / others.sum())

                # Stage 2: Top 5 Aggregate > 40% -> 38.5%
                w_sorted = w_series.sort_values(ascending=False)
                top5_tickers = w_sorted.iloc[:5].index
                top5_sum = w_series[top5_tickers].sum()
                
                if top5_sum > 0.40:
                    scale_factor = 0.385 / top5_sum
                    w_series[top5_tickers] = w_series[top5_tickers] * scale_factor
                    
                    surplus = top5_sum - 0.385
                    
                    # Distribute surplus to non-Top5
                    others_tickers = w_series.index.difference(top5_tickers)
                    others = w_series[others_tickers]
                    if not others.empty:
                        w_series[others_tickers] = others + (surplus * others / others.sum())

                    # Constraint: 5th largest cap limit for others
                    # Re-evaluate 5th largest after scaling
                    w_curr_sorted = w_series.sort_values(ascending=False)
                    fifth_val = w_curr_sorted.iloc[4]
                    cap_val = min(0.044, fifth_val)
                    
                    outside_tickers = w_curr_sorted.iloc[5:].index
                    outside_excess = w_series[outside_tickers][w_series[outside_tickers] > cap_val]
                    
                    if not outside_excess.empty:
                        surplus_2 = (outside_excess - cap_val).sum()
                        w_series[outside_excess.index] = cap_val
                        
                        # Redistribute to remaining unrestricted
                        unrestricted_tickers = w_series.index.difference(top5_tickers).difference(outside_excess.index)
                        if not unrestricted_tickers.empty:
                             w_series[unrestricted_tickers] = w_series[unrestricted_tickers] + (surplus_2 * w_series[unrestricted_tickers] / w_series[unrestricted_tickers].sum())

                # Loop Check
                # If Max <= 15% and Top5 <= 40%, we correspond to the 'Use Initial Weights' logic of stage 1/2
                # Strictly speaking, we should check if we violated anything.
                # But since we force caps, iterating should converge.
                if w_series.max() <= 0.1501 and w_series.sort_values(ascending=False).iloc[:5].sum() <= 0.4001:
                    break
        
        return pd.DataFrame({'Ticker': w_series.index, 'CappedWeight': w_series.values})

    print("Applying NDX capping rules (Quarterly: 24%/4.5% | Annual: 14%/38.5%)...")
    
    # Apply per Date
    # Create temp df to merge back
    capped_list = []
    
    for dt, grp in res_df.groupby('Date'):
        # Determine if Annual Reconstitution (December)
        # dt is Timestamp
        is_annual = (dt.month == 12)
        
        capped_grp = apply_ndx_capping(grp, is_annual=is_annual)
        capped_grp['Date'] = dt
        capped_list.append(capped_grp)
        
    capped_df = pd.concat(capped_list)
    
    # Merge back to update weights
    # Note: 'res_df' has multiple rows, 'capped_df' has adjusted weights
    res_df = res_df.merge(capped_df, on=['Date', 'Ticker'], how='left')
    
    # Overwrite Weight
    res_df['Weight'] = res_df['CappedWeight']
    res_df.drop(columns=['CappedWeight'], inplace=True)
    
    # Save
    res_df.to_csv(OUTPUT_FILE, index=False)
    print(f"Saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    reconstruct()

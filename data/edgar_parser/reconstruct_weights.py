import pandas as pd
import json
import yfinance as yf
import numpy as np
import datetime
import os

# Configuration
INPUT_CSV = "nasdaq_components.csv"
MAPPING_FILE = "name_mapping.json"
OUTPUT_FILE = "nasdaq_quarterly_weights.csv"
PROXY_TICKER = "QQQ"

def load_data():
    print("Loading data...")
    df = pd.read_csv(INPUT_CSV)
    df['Date'] = pd.to_datetime(df['Date'])
    
    with open(MAPPING_FILE, "r") as f:
        mapping = json.load(f)
        
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

def get_quarter_ends(start_year=2000, end_year=2025):
    quarters = []
    current_date = datetime.date(start_year, 3, 31)
    end_date = datetime.date.today()
    
    while current_date <= end_date:
        quarters.append(pd.Timestamp(current_date))
        # Move to next quarter
        # Add 3 months approx
        year = current_date.year
        month = current_date.month + 3
        if month > 12:
            month -= 12
            year += 1
        
        # Get last day of month
        if month in [4, 6, 9, 11]: day = 30
        elif month == 2: day = 28 # Simplified, Leap years handled by Pandas if we just use MonthEnd?
        else: day = 31
        
        # Correct approach using pd.tseries.offsets
        current_date = (pd.Timestamp(current_date) + pd.tseries.offsets.QuarterEnd()).date()
        # Wait, Loop trigger.
        # Just use pd.date_range
        pass
        break

    # Easier way
    return pd.date_range(start=f'{start_year}-03-31', end=pd.Timestamp.now(), freq='Q')

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
    
    quarters = get_quarter_ends()
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
        
        valid_filings = df[df['Date'] <= q_date]
        if valid_filings.empty:
            continue
            
        latest_date = valid_filings['Date'].max()
        filing_data = valid_filings[valid_filings['Date'] == latest_date].copy()
        
        # 2. Project Value
        
        # Get prices at Filing Date and Quarter Date
        # Find closest available price index looking backwards
        idx_filing = prices.index.get_indexer([latest_date], method='pad')[0]
        idx_quarter = prices.index.get_indexer([q_date], method='pad')[0]
        
        if idx_filing == -1 or idx_quarter == -1:
            continue
            
        date_f = prices.index[idx_filing]
        date_q = prices.index[idx_quarter]
        
        # Proxy Return
        try:
            proxy_val_f = proxy_prices.loc[date_f]
            proxy_val_q = proxy_prices.loc[date_q]
            proxy_ret = proxy_val_q / proxy_val_f if proxy_val_f > 0 else 1.0
        except:
            proxy_ret = 1.0

        for _, row in filing_data.iterrows():
            name = row['Company']
            
            # Skip invalid rows (header garbage)
            try:
                val_f = float(str(row['Value']).replace(',',''))
            except:
                continue

            ticker = mapping.get(name)
            
            val_q = val_f # Default NO CHANGE (or use proxy_ret)
            
            if ticker and ticker in prices.columns:
                try:
                    p_f = prices.at[date_f, ticker]
                    p_q = prices.at[date_q, ticker]
                    
                    if pd.notna(p_f) and pd.notna(p_q) and p_f > 0:
                        ret = p_q / p_f
                        val_q = val_f * ret
                    else:
                        # Fallback to proxy
                        val_q = val_f * proxy_ret
                except:
                     val_q = val_f * proxy_ret
            else:
                 # Unmapped: Proxy
                 val_q = val_f * proxy_ret
            
            final_rows.append({
                "Date": q_date.date(),
                "Ticker": ticker if ticker else name, # Use Name if no ticker
                "Name": name,
                "Value": val_q,
                "IsMapped": bool(ticker)
            })

    # Convert to DataFrame
    res_df = pd.DataFrame(final_rows)
    
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
    
    def apply_ndx_capping(group):
        # Working with Series
        w = group['Weight'].values.copy()
        tickers = group['Ticker'].values.copy()
        
        # We need to preserve mapping to return series
        w_series = pd.Series(w, index=tickers)
        
        # Iterative solver (Methodology says "process is repeated until... meet constraints")
        for _ in range(10): 
            # Normalize first
            w_series = w_series / w_series.sum()
            
            # --- Stage 1 ---
            # "If no company’s initial weight exceeds 24%... initial weights are used"
            # "Otherwise... such that no company’s weight may exceed 20%"
            
            max_w = w_series.max()
            if max_w > 0.24:
                # Cap all > 24% to 20%? 
                # Methodology: "adjusted such that no company’s weight may exceed 20%"
                # Typically implies straightforward capping at 0.20 and redistribution.
                excess = w_series[w_series > 0.20] # Note: The constraint implies target is 20%
                if not excess.empty:
                    surplus = (excess - 0.20).sum()
                    w_series[w_series > 0.20] = 0.20
                    
                    # Redistribute to proportional to others
                    others = w_series[w_series <= 0.20]
                    if not others.empty:
                         w_series[w_series <= 0.20] = others + (surplus * others / others.sum())
            
            # Re-Check Stage 1 satisfaction? 
            # If we redistributed, something small might have grown? Unlikely to cross 24 from <20.
            
            # --- Stage 2 ---
            # "If the aggregate weight of the companies whose Stage 1 weights exceed 4.5% does not exceed 48%..."
            # "Otherwise... aggregate weight... is set to 40%."
            
            large_caps = w_series[w_series > 0.045]
            agg_weight = large_caps.sum()
            
            if agg_weight > 0.48:
                # Target aggregate is 40%
                # We need to scale down the large caps to sum to 0.40
                scale_factor = 0.40 / agg_weight
                w_series[w_series > 0.045] = w_series[w_series > 0.045] * scale_factor
                
                # The surplus (agg_weight - 0.40) goes to the small caps
                surplus = agg_weight - 0.40
                
                small_caps = w_series[w_series <= 0.045]
                if not small_caps.empty:
                    # "Companies with Stage 1 weights below 4.5% may also have their weights adjusted to preserve the initial rank order"
                    # Proportional distribution preserves rank.
                    w_series[w_series <= 0.045] = small_caps + (surplus * small_caps / small_caps.sum())
            
            # Check constraints again
            valid_stage1 = w_series.max() <= 0.24
            valid_stage2 = w_series[w_series > 0.045].sum() <= 0.48
            
            if valid_stage1 and valid_stage2:
                break
                
        return pd.DataFrame({'Ticker': w_series.index, 'CappedWeight': w_series.values})

    print("Applying NDX capping rules (24%/4.5% Mod Mkt Cap)...")
    
    # Apply per Date
    # Create temp df to merge back
    capped_list = []
    
    for dt, grp in res_df.groupby('Date'):
        capped_grp = apply_ndx_capping(grp)
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

import yfinance as yf
import pandas as pd
import os
import config

def get_price_data(tickers, start_date='2000-01-01', force_refresh=False):
    """
    Fetches price data for the specified tickers.
    Uses a local pickle cache to avoid redundant downloads.
    """
    cache_file = config.PRICE_CACHE_FILE
    
    # 1. Try Loading Cache
    if not force_refresh and os.path.exists(cache_file):
        print(f"Loading price data from cache: {cache_file} ...")
        try:
            df = pd.read_pickle(cache_file)
            
            # Check Ticker Coverage
            # Note: yfinance might drop tickers if invalid (delisted).
            # So missing tickers in cache might just be invalid tickers.
            # But if the cache has FEWER columns than requested, we might need more.
            # However, simpler logic: If we have a decent chunk, assume it's the "Master Cache".
            # Better: Check if requested "Benchmark" is present.
            
            missing = [t for t in tickers if t not in df.columns]
            
            # Heuristic: If we are missing > 10% of tickers, maybe we need to refresh.
            # Or if specific critical ones are missing.
            
            pct_missing = len(missing) / len(tickers) if tickers else 0
            if pct_missing > 0.10:
                print(f"Cache missing {len(missing)} tickers ({pct_missing:.1%} coverage). Download recommended.")
                # We could auto-download here, but let's stick to using cache unless force_refresh.
                # Actually, if we are running a new simulation with NEW tickers, we MUST download.
                print("Refreshing data to ensure coverage...")
                return download_and_cache(tickers, start_date, cache_file)
            
            # Check Date Coverage
            # If start_date is earlier than cache start?
            if pd.to_datetime(start_date) < df.index[0]:
                 print(f"Cache starts {df.index[0].date()}, need {start_date}. Refreshing...")
                 return download_and_cache(tickers, start_date, cache_file)
                 
            return df
            
        except Exception as e:
            print(f"Error loading cache: {e}. Downloading fresh...")
            
    # 2. Download Fresh
    return download_and_cache(tickers, start_date, cache_file)

def download_and_cache(tickers, start_date, cache_file):
    print(f"Downloading prices for {len(tickers)} tickers from {start_date}...")
    
    # Chunking to prevent timeouts? yfinance handles 200 okay.
    # But let's handle the case where 'tickers' has duplicates.
    unique_tickers = list(set(tickers))
    
    data = yf.download(unique_tickers, start=start_date, auto_adjust=True, progress=True)
    
    if 'Close' in data.columns:
        new_df = data['Close']
    else:
        new_df = data
        
    # Merge with existing cache to preserve other tickers
    if os.path.exists(cache_file):
        try:
            old_df = pd.read_pickle(cache_file)
            print(f"Merging with existing cache ({len(old_df.columns)} cols)...")
            
            # Align indices
            # Drop columns from old that are in new (refresh them)
            cols_to_drop = [c for c in new_df.columns if c in old_df.columns]
            if cols_to_drop:
                old_df = old_df.drop(columns=cols_to_drop)
                
            # Outer join on Date index
            df = old_df.join(new_df, how='outer').sort_index()
        except Exception as e:
            print(f"Merge failed ({e}), using fresh data only.")
            df = new_df
    else:
        df = new_df
        
    df.to_pickle(cache_file)
    print(f"Saved {len(df.columns)} tickers to cache: {cache_file}")
    return df

import yfinance as yf
import pandas as pd
import os
import config
from datetime import datetime, timedelta

# Global data source setting (can be overridden by environment variable)
DATA_SOURCE = os.environ.get('NDX_DATA_SOURCE', 'yfinance')
POLYGON_API_KEY = os.environ.get('POLYGON_API_KEY', None)

def get_price_data(tickers, start_date='2000-01-01', force_refresh=False, data_source=None, polygon_api_key=None):
    """
    Fetches price data for the specified tickers.
    Uses a local pickle cache to avoid redundant downloads.
    
    Args:
        tickers: List of ticker symbols
        start_date: Start date for data
        force_refresh: If True, bypass cache and download fresh
        data_source: 'yfinance' (default), 'polygon', or 'stooq'
        polygon_api_key: API key for Polygon.io (required if data_source='polygon')
    """
    # Determine data source
    source = data_source or DATA_SOURCE
    api_key = polygon_api_key or POLYGON_API_KEY
    
    # Use different cache files for different sources
    if source == 'polygon':
        cache_file = config.PRICE_CACHE_FILE.replace('.pkl', '_polygon.pkl')
    elif source == 'stooq':
        cache_file = config.PRICE_CACHE_FILE.replace('.pkl', '_stooq.pkl')
    else:
        cache_file = config.PRICE_CACHE_FILE
    
    # 1. Try Loading Cache
    if not force_refresh and os.path.exists(cache_file):
        print(f"Loading price data from cache: {cache_file} ...")
        try:
            df = pd.read_pickle(cache_file)
            
            missing = [t for t in tickers if t not in df.columns]
            pct_missing = len(missing) / len(tickers) if tickers else 0
            
            if pct_missing > 0.25:
                print(f"Cache missing {len(missing)} tickers ({pct_missing:.1%} coverage). Download recommended.")
                print("Refreshing data to ensure coverage...")
                return download_and_cache(tickers, start_date, cache_file, source, api_key)
            elif pct_missing > 0.10:
                print(f"Cache missing {len(missing)} tickers ({pct_missing:.1%}) — likely delisted, using cache as-is.")
            
            if pd.to_datetime(start_date) < df.index[0]:
                print(f"Cache starts {df.index[0].date()}, need {start_date}. Refreshing...")
                return download_and_cache(tickers, start_date, cache_file, source, api_key)
                 
            return df
            
        except Exception as e:
            print(f"Error loading cache: {e}. Downloading fresh...")
            
    # 2. Download Fresh
    return download_and_cache(tickers, start_date, cache_file, source, api_key)

def download_and_cache(tickers, start_date, cache_file, data_source='yfinance', polygon_api_key=None):
    """Download price data from specified source and cache it."""
    unique_tickers = list(set(tickers))
    
    if data_source == 'polygon':
        if not polygon_api_key:
            raise ValueError("Polygon API key required. Set POLYGON_API_KEY env var or pass polygon_api_key parameter.")
        new_df = download_from_polygon(unique_tickers, start_date, polygon_api_key)
    elif data_source == 'stooq':
        new_df = download_from_stooq(unique_tickers, start_date)
    else:
        # Default: Yahoo Finance with Fallback
        new_df = download_from_yfinance(unique_tickers, start_date)
        
        # --- Fallback Logic ---
        # Identify missing tickers (requested but not returned OR returned as all NaNs)
        downloaded_tickers = new_df.columns.tolist() if not new_df.empty else []
        
        # Check 1: Tickers not in columns
        missing_tickers = [t for t in unique_tickers if t not in downloaded_tickers]
        
        # Check 2: Tickers with >50% NaN (yfinance returns partial data for many delisted)
        if not new_df.empty:
            for t in list(downloaded_tickers):
                nan_pct = new_df[t].isna().mean()
                if nan_pct > 0.50:
                    missing_tickers.append(t)
                    new_df = new_df.drop(columns=[t])
        
        missing_tickers = list(set(missing_tickers))
        
        if missing_tickers:
            print(f"Primary source (yfinance) missed {len(missing_tickers)} tickers. Attempting fallback to Stooq...")
            stooq_df = download_from_stooq(missing_tickers, start_date)
            
            if not stooq_df.empty:
                print(f"Stooq recovered {len(stooq_df.columns)} tickers: {stooq_df.columns.tolist()}")
                
                # Merge Stooq results into new_df
                if new_df.empty:
                    new_df = stooq_df
                else:
                    new_df = new_df.join(stooq_df, how='outer').sort_index()
            else:
                 print("Stooq fallback yielded no data.")

            # Individual retry for tickers that failed in batch
            # (yfinance batch downloads sometimes miss active tickers)
            still_missing = [t for t in missing_tickers if t not in (stooq_df.columns.tolist() if not stooq_df.empty else [])]
            if still_missing and len(still_missing) <= 200:
                print(f"Retrying {len(still_missing)} tickers individually via yfinance...")
                import time
                recovered = {}
                for t in still_missing:
                    try:
                        single = yf.download(t, start=start_date, auto_adjust=True, progress=False)
                        if not single.empty:
                            close = single['Close'] if 'Close' in single.columns else single.iloc[:, 0]
                            if hasattr(close, 'columns'):
                                close = close.iloc[:, 0]
                            if close.notna().sum() > 50:
                                recovered[t] = close
                        time.sleep(0.1)
                    except Exception:
                        pass
                if recovered:
                    print(f"  Individual retry recovered {len(recovered)} tickers: {list(recovered.keys())[:10]}")
                    retry_df = pd.DataFrame(recovered)
                    retry_df.index = pd.to_datetime(retry_df.index)
                    if new_df.empty:
                        new_df = retry_df
                    else:
                        new_df = new_df.join(retry_df, how='outer').sort_index()

    if new_df is None or new_df.empty:
        print("ERROR: No data downloaded!")
        return pd.DataFrame()
        
    # Merge with existing cache to preserve other tickers
    if os.path.exists(cache_file):
        try:
            old_df = pd.read_pickle(cache_file)
            print(f"Merging with existing cache ({len(old_df.columns)} cols)...")
            
            cols_to_drop = [c for c in new_df.columns if c in old_df.columns]
            if cols_to_drop:
                old_df = old_df.drop(columns=cols_to_drop)
                
            df = old_df.join(new_df, how='outer').sort_index()
        except Exception as e:
            print(f"Merge failed ({e}), using fresh data only.")
            df = new_df
    else:
        df = new_df
        
    # Apply successor ticker fallback for acquired companies
    df = apply_successor_fallback(df)

    df.to_pickle(cache_file)
    print(f"Saved {len(df.columns)} tickers to cache: {cache_file}")
    return df

def apply_successor_fallback(df):
    """For delisted tickers with no data, try to fill with successor ticker data.
    Downloads successor tickers that aren't already in the cache."""
    try:
        from mapper import SUCCESSOR_TICKERS
    except ImportError:
        return df

    successors_needed = {}
    for t, succ in SUCCESSOR_TICKERS.items():
        if succ and succ != t:
            # Fill if ticker is missing OR has >50% NaN (matching yfinance threshold)
            if t not in df.columns or df[t].isna().mean() > 0.50:
                successors_needed[t] = succ

    if not successors_needed:
        return df

    # Download successor tickers that aren't in the dataframe
    missing_succs = list(set(
        succ for succ in successors_needed.values()
        if succ not in df.columns or df[succ].isna().all()
    ))
    if missing_succs:
        import time
        print(f"  Downloading {len(missing_succs)} successor tickers: {missing_succs}")
        for succ in missing_succs:
            try:
                single = yf.download(succ, start='2000-01-01', auto_adjust=True, progress=False)
                if not single.empty:
                    close = single['Close'] if 'Close' in single.columns else single.iloc[:, 0]
                    if hasattr(close, 'columns'):
                        close = close.iloc[:, 0]
                    if close.notna().sum() > 50:
                        close.index = pd.to_datetime(close.index)
                        df[succ] = close.reindex(df.index)
                        print(f"    Downloaded {succ}: {close.notna().sum()} days")
                time.sleep(0.1)
            except Exception as e:
                print(f"    Failed to download {succ}: {e}")

    filled = 0
    for orig, succ in successors_needed.items():
        if succ in df.columns and not df[succ].isna().all():
            df[orig] = df[succ]
            filled += 1

    if filled:
        print(f"  Successor fallback: filled {filled} delisted tickers with acquirer data")

    return df

def download_from_yfinance(tickers, start_date):
    """Download price data from yfinance."""
    print(f"[yfinance] Downloading prices for {len(tickers)} tickers from {start_date}...")
    
    data = yf.download(tickers, start=start_date, auto_adjust=True, progress=True)
    
    if 'Close' in data.columns:
        return data['Close']
    else:
        return data

def download_from_stooq(tickers, start_date):
    """Download price data directly from Stooq CSV endpoint (no pandas-datareader)."""
    import time
    import io
    import requests

    print(f"[Stooq Direct] Downloading prices for {len(tickers)} tickers from {start_date}...")

    all_data = {}
    failed_tickers = []

    session = requests.Session()
    session.headers.update({
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)'
    })

    start_dt = pd.to_datetime(start_date)

    for i, ticker in enumerate(tickers):
        if (i + 1) % 20 == 0:
            print(f"  Progress: {i + 1}/{len(tickers)} tickers...")

        # Stooq URL format
        stooq_ticker = f"{ticker}.US" if not ticker.startswith('^') else ticker
        url = (f"https://stooq.com/q/d/l/"
               f"?s={stooq_ticker}"
               f"&d1={start_dt.strftime('%Y%m%d')}"
               f"&d2={pd.Timestamp.now().strftime('%Y%m%d')}"
               f"&i=d")

        try:
            resp = session.get(url, timeout=15)
            if resp.status_code == 200 and 'Date' in resp.text[:50]:
                df = pd.read_csv(io.StringIO(resp.text), parse_dates=['Date'], index_col='Date')
                if not df.empty and 'Close' in df.columns:
                    series = df['Close'].sort_index()
                    if len(series) > 10:
                        all_data[ticker] = series
                    else:
                        failed_tickers.append(ticker)
                else:
                    failed_tickers.append(ticker)
            else:
                failed_tickers.append(ticker)
        except Exception:
            failed_tickers.append(ticker)

        time.sleep(0.5)  # Rate limit

    if failed_tickers:
        print(f"  Failed: {len(failed_tickers)} tickers")

    if not all_data:
        return pd.DataFrame()

    df = pd.DataFrame(all_data)
    df.index = pd.to_datetime(df.index)
    df = df.sort_index()
    print(f"  Downloaded {len(df.columns)} tickers, {len(df)} trading days")
    return df

def download_from_polygon(tickers, start_date, api_key):
    """
    Download price data from Polygon.io.
    
    Polygon.io provides more complete historical data than yfinance,
    including data for delisted tickers.
    """
    try:
        from polygon import RESTClient
    except ImportError:
        print("ERROR: polygon-api-client not installed!")
        print("Install with: pip install polygon-api-client")
        raise ImportError("Please install polygon-api-client: pip install polygon-api-client")
    
    print(f"[Polygon.io] Downloading prices for {len(tickers)} tickers from {start_date}...")
    
    client = RESTClient(api_key)
    
    # Convert dates
    start_dt = pd.to_datetime(start_date)
    end_dt = datetime.now()
    
    all_data = {}
    failed_tickers = []
    
    for i, ticker in enumerate(tickers):
        if (i + 1) % 20 == 0:
            print(f"  Progress: {i + 1}/{len(tickers)} tickers...")
            
        try:
            # Get daily aggregates
            aggs = client.get_aggs(
                ticker=ticker,
                multiplier=1,
                timespan="day",
                from_=start_dt.strftime('%Y-%m-%d'),
                to=end_dt.strftime('%Y-%m-%d'),
                adjusted=True,
                sort="asc",
                limit=50000  # Max allowed
            )
            
            if aggs:
                # Convert to DataFrame
                dates = [pd.to_datetime(a.timestamp, unit='ms') for a in aggs]
                closes = [a.close for a in aggs]
                all_data[ticker] = pd.Series(closes, index=dates, name=ticker)
            else:
                print(f"  Warning: No data for {ticker}")
                failed_tickers.append(ticker)
                
        except Exception as e:
            print(f"  Error fetching {ticker}: {e}")
            failed_tickers.append(ticker)
    
    if failed_tickers:
        print(f"  Failed tickers: {len(failed_tickers)} ({', '.join(failed_tickers[:10])}{'...' if len(failed_tickers) > 10 else ''})")
    
    if not all_data:
        print("ERROR: No data downloaded from Polygon!")
        return pd.DataFrame()
    
    # Combine into DataFrame
    df = pd.DataFrame(all_data)
    df.index = pd.to_datetime(df.index)
    df = df.sort_index()
    
    print(f"  Downloaded {len(df)} trading days for {len(df.columns)} tickers")
    print(f"  Date range: {df.index[0].date()} to {df.index[-1].date()}")
    
    return df


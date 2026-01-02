import pandas as pd
import streamlit as st
import os
import subprocess
import sys
import yfinance as yf
from app.services import testfol_api as api

@st.cache_data(show_spinner="Fetching Component Data...", ttl=3600)
def fetch_component_data(tickers, start_date, end_date):
    """
    Fetches historical data for each ticker individually via Testfol API (or local).
    Handles composite tickers like NDXMEGASIM by splicing local CSVs with live data.
    """
    combined_prices = pd.DataFrame()
    unique_tickers = list(set(tickers))
    
    for ticker in unique_tickers:
        try:
            # Parse Ticker
            base = ticker.split("?")[0]
            if base in combined_prices.columns:
                continue
                
            # SPECIAL: Load NDX Mega simulations from local CSV + Splice with QBIG
            if base in ["NDXMEGASIM", "NDXMEGA2SIM"]:
                try:
                    # 1. Load Simulation Data (dynamic path based on ticker)
                    csv_path = f"data/{base}.csv"
                    df_sim = pd.DataFrame()
                    if os.path.exists(csv_path):
                        try:
                            df_sim = pd.read_csv(csv_path)
                        except (pd.errors.ParserError, pd.errors.EmptyDataError, ValueError) as e:
                            st.warning(f"⚠️ Corruption detected in {base}.csv ({e}). Attempting auto-rebuild...")
                            rebuild_script = os.path.join("data", "ndx_simulation", "scripts", "rebuild_all.py")
                            if os.path.exists(rebuild_script):
                                try:
                                    with st.spinner("♻️ Rebuilding Simulation Data... (This takes a moment)"):
                                        subprocess.run([sys.executable, rebuild_script], check=True)
                                    st.success("Rebuild complete. Reloading data.")
                                    df_sim = pd.read_csv(csv_path) # Retry load
                                except Exception as rebuild_err:
                                    st.error(f"❌ Rebuild failed: {rebuild_err}")
                                    return pd.DataFrame()
                            else:
                                st.error(f"Cannot rebuild: Script not found at {rebuild_script}")
                                return pd.DataFrame()

                        if 'Date' in df_sim.columns:
                            df_sim['Date'] = pd.to_datetime(df_sim['Date'])
                            df_sim = df_sim.set_index('Date')
                        if 'Close' not in df_sim.columns:
                            st.warning(f"{base}.csv missing 'Close' column")
                            df_sim = pd.DataFrame()
                        else:
                            df_sim = df_sim['Close'].sort_index() # Convert to Series
                    else:
                        st.warning(f"{base} requested but {csv_path} not found.")

                    # 2. Fetch QBIG (Live Proxy) using yfinance
                    try:
                        qbig_df = yf.download("QBIG", period="max", auto_adjust=True, progress=False)
                        
                        # Handle varied yfinance return structures
                        qbig_series = pd.Series(dtype=float)
                        if not qbig_df.empty:
                            if 'Close' in qbig_df:
                                 qbig_series = qbig_df['Close']
                            elif 'Adj Close' in qbig_df:
                                 qbig_series = qbig_df['Adj Close']
                            else:
                                 qbig_series = qbig_df.iloc[:,0]
                        
                        if isinstance(qbig_series, pd.DataFrame):
                            qbig_series = qbig_series.iloc[:,0]
                            
                        qbig_series.index = pd.to_datetime(qbig_series.index)
                        qbig_series = qbig_series.sort_index()
                    except Exception as e:
                        st.warning(f"Failed to fetch QBIG data: {e}")
                        qbig_series = pd.Series(dtype=float)

                    # 3. Splice
                    if not qbig_series.empty and not df_sim.empty:
                        # Find splice point (Start of QBIG)
                        splice_date = qbig_series.index[0]
                        
                        # Get Sim Data UP TO Splice Date
                        sim_part = df_sim[df_sim.index < splice_date]
                        
                        if not sim_part.empty:
                            # Align Sim End to QBIG Start
                            sim_end_val = sim_part.iloc[-1]
                            qbig_start_val = qbig_series.iloc[0]
                            scale_factor = qbig_start_val / sim_end_val if sim_end_val != 0 else 1.0
                            
                            sim_part_scaled = sim_part * scale_factor
                            
                            combined = pd.concat([sim_part_scaled, qbig_series])
                            combined_prices[base] = combined
                        else:
                             combined_prices[base] = qbig_series
                             
                    elif not df_sim.empty:
                        combined_prices[base] = df_sim
                    elif not qbig_series.empty:
                        combined_prices[base] = qbig_series
                        
                    continue
                    
                except Exception as e:
                    st.error(f"Failed to load/splice local {base}: {e}")
                    
            # API Fetch for standard tickers
            # We can use the api.fetch_backtest directly as it has disk caching.
            # No need for the extra st.cache_data layer here since specific args change often,
            # but this function is cached with st.cache_data at the top level anyway.
            
            # 1. Try Yahoo Finance First (For Real Market Prices)
            # EXCEPTION: If the ticker implies a specific SIMULATION (e.g. VXUSSIM),
            # we want the EXTENDED history from Testfol, NOT the short real history from Yahoo (VXUS).
            # So we SKIP Yahoo for *SIM tickers (unless they are handled above like NDXMEGASIM)
            is_sim_request = base.upper().endswith("SIM") or base.upper().endswith("TR")
            
            if not is_sim_request:
                try:
                    from app.core.shadow_backtest import parse_ticker
                    # Resolve real ticker (e.g. QQQSIM -> QQQ)
                    mapped_ticker, _ = parse_ticker(base)
                    
                    # Use history() showing real price
                    yf_obj = yf.Ticker(mapped_ticker)
                    # Ensure dates are strings
                    sd_str = start_date.strftime("%Y-%m-%d") if hasattr(start_date, 'strftime') else str(start_date)
                    ed_str = end_date.strftime("%Y-%m-%d") if hasattr(end_date, 'strftime') else str(end_date)
                    
                    hist = yf_obj.history(start=sd_str, end=ed_str, auto_adjust=True)
                    if not hist.empty and 'Close' in hist.columns:
                        # Normalize Timezone to Naive to prevent mismatch with other data sources
                        if hist.index.tz is not None:
                            hist.index = hist.index.tz_localize(None)
                            
                        combined_prices[base] = hist['Close']
                        continue
                except Exception:
                    pass # Fail silently and use fallback

            # 2. Testfol API Fallback (For Synthetic/Custom Tickers)
            # Returns an Equity Curve (Backtest starting at 10,000)
            broad_start = "1900-01-01" 
            broad_end = pd.Timestamp.now().strftime("%Y-%m-%d")
            
            series, _, _ = api.fetch_backtest(
                start_date=broad_start,
                end_date=broad_end,
                start_val=10000,
                cashflow=0,
                cashfreq="Monthly",
                rolling=1,
                invest_div=True, # Total Return
                rebalance="Yearly",
                allocation={base: 100.0}
            )
            combined_prices[base] = series
            
        except Exception as e:
            st.warning(f"Failed to fetch data for {ticker}: {e}")
            
    return combined_prices

import time
@st.cache_data
def get_fed_funds_rate():
    """
    Fetches historical Fed Funds Rate (daily) from FRED or local cache.
    Returns: pd.Series with DatetimeIndex and rate as float (e.g. 5.25 for 5.25%)
    """
    data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../data")
    file_path = os.path.join(data_dir, "FEDFUNDS.csv")
    
    # Download if missing or old (>30 days)
    should_download = True
    if os.path.exists(file_path):
        mtime = os.path.getmtime(file_path)
        if (time.time() - mtime) < (30 * 86400):
            should_download = False
            
    if should_download:
        try:
            url = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=FEDFUNDS"
            # Fake User-Agent to avoid 403
            headers = {'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)'}
            import requests
            r = requests.get(url, headers=headers)
            r.raise_for_status()
            with open(file_path, "wb") as f:
                f.write(r.content)
            print("Downloaded fresh FEDFUNDS.csv")
        except Exception as e:
            print(f"Failed to download FEDFUNDS: {e}. using cached if available.")
            
    if os.path.exists(file_path):
        try:
            # FRED CSV format: observation_date, FEDFUNDS
            df = pd.read_csv(file_path, parse_dates=["observation_date"], index_col="observation_date")
            # Resample to daily (forward fill monthlies)
            full_idx = pd.date_range(start=df.index.min(), end=pd.Timestamp.today(), freq='D')
            daily_series = df['FEDFUNDS'].reindex(full_idx, method='ffill')
            return daily_series
        except Exception as e:
             st.error(f"Error reading FEDFUNDS.csv: {e}")
             return None
    return None

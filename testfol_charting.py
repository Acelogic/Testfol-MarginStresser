#!/usr/bin/env python3
# ─────────────────────────────────────────────────────────────────────────────
# testfol_charting.py
#
# New Streamlit App for Testfol Backtesting with Multi-Timeframe Candlesticks
# ─────────────────────────────────────────────────────────────────────────────

import pandas as pd
import streamlit as st
import time
import os
from app.services import fetch_backtest
from app.core import run_shadow_backtest, calculations
from app.common import utils
from app.ui import render_sidebar, render_config, render_results, asset_explorer

# ─────────────────────────────────────────────────────────────────────────────
# Caching Wrappers
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_data(show_spinner="Fetching data from Testfol API...", ttl=3600)
def cached_fetch_backtest(*args, **kwargs):
    """Cached wrapper for api.fetch_backtest"""
    return fetch_backtest(*args, **kwargs)

@st.cache_data(show_spinner="Running Shadow Backtest...", ttl=3600)
def cached_run_shadow_backtest_v2(*args, **kwargs):
    """Cached wrapper for shadow_backtest.run_shadow_backtest"""
    return run_shadow_backtest(*args, **kwargs)

@st.cache_data(show_spinner="Fetching Component Data...", ttl=3600)
def fetch_component_data(tickers, start_date, end_date):
    """
    Fetches historical data for each ticker individually via Testfol API.
    Uses universal API caching in testfol_api.py
    """
    combined_prices = pd.DataFrame()
    unique_tickers = list(set(tickers))
    
    # Pre-fetch check or parallelize? For now sequential.
    
    for ticker in unique_tickers:
        try:
            # Parse Ticker
            base = ticker.split("?")[0]
            if base in combined_prices.columns:
                continue
                
            # SPECIAL: Load NDXMEGASIM from local CSV + Splice with QBIG (Defiance Nasdaq 100 Enhanced Options & Growth ETF)
            if base == "NDXMEGASIM":
                try:
                    # 1. Load Simulation Data
                    csv_path = "data/NDXMEGASIM.csv"
                    df_sim = pd.DataFrame()
                    if os.path.exists(csv_path):
                        df_sim = pd.read_csv(csv_path)
                        if 'Date' in df_sim.columns:
                            df_sim['Date'] = pd.to_datetime(df_sim['Date'])
                            df_sim = df_sim.set_index('Date')
                        if 'Close' not in df_sim.columns:
                            st.warning("NDXMEGASIM.csv missing 'Close' column")
                            df_sim = pd.DataFrame()
                        else:
                            df_sim = df_sim['Close'].sort_index() # Convert to Series
                    else:
                        st.warning(f"NDXMEGASIM requested but {csv_path} not found.")

                    # 2. Fetch QBIG (Live Proxy)
                    # Use a cache-friendly fetch or direct download?
                    # Let's use direct download to ensure we get "Latest" data regardless of cache
                    try:
                        import yfinance as yf
                        qbig_df = yf.download("QBIG", period="max", auto_adjust=True, progress=False)
                        if 'Close' in qbig_df:
                             qbig_series = qbig_df['Close']
                        elif 'Adj Close' in qbig_df:
                             qbig_series = qbig_df['Adj Close']
                        else:
                             # Fallback for yfinance structure changes
                             qbig_series = qbig_df.iloc[:,0] if not qbig_df.empty else pd.Series()
                        
                        if isinstance(qbig_series, pd.DataFrame):
                            qbig_series = qbig_series.iloc[:,0] # Handle multi-index if single ticker
                            
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
                            # Scale Sim to match QBIG start
                            # Need Sim Price at Splice Date? (Or last available date)
                            # Actually, look for overlap?
                            # Sim ends at splice_date (approx).
                            # Let's align the END of Sim Part to the START of QBIG using last available value.
                            
                            sim_end_val = sim_part.iloc[-1]
                            qbig_start_val = qbig_series.iloc[0]
                            
                            # Scaling Factor: We heavily simulated "Growth of $100" in sim.
                            # QBIG starts at ~$20?
                            # We scale SIM DOWN/UP to match QBIG.
                            scale_factor = qbig_start_val / sim_end_val
                            
                            sim_part_scaled = sim_part * scale_factor
                            
                            # Combine
                            combined = pd.concat([sim_part_scaled, qbig_series])
                            combined_prices[base] = combined
                            # st.info(f"Spliced NDXMEGASIM with QBIG at {splice_date.date()} (Scale: {scale_factor:.4f})")
                        else:
                             # Overlap issue? Just use QBIG?
                             combined_prices[base] = qbig_series
                             
                    elif not df_sim.empty:
                        combined_prices[base] = df_sim
                    elif not qbig_series.empty:
                        combined_prices[base] = qbig_series
                        
                    continue
                    
                except Exception as e:
                    st.error(f"Failed to load/splice local NDXMEGASIM: {e}")
                    
            # Fetch Data (Universal Cache handles hits/misses)
            # We still request broad history to maximize cache utility across diff ranges?
            # Actually, with Hash Caching, exact params matter.
            # If user asks for 2020-2024, hash includes 2020-2024.
            # If next user asks for 2019-2024, it's a MISS.
            # To make cache EFFECTIVE, we should normalize the request dates here.
            # Requesting 1900-Present matches the universal cache strategy best.
            
            broad_start = "1900-01-01" 
            broad_end = pd.Timestamp.now().strftime("%Y-%m-%d")
            
            series, _, _ = cached_fetch_backtest(
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
            
            # Slice to requested range? Or handled by caller/simulation?
            # Returns are usually sliced by common index intersection in simulation.
            
            combined_prices[base] = series
            
            # Rate limit might still be needed on MISS, but API layer doesn't sleep.
            # We can sleep here just in case, or sleep in API layer?
            # If it's a CACHE HIT, loop is fast.
            # If MISS, we might spam.
            # Let's keep a small sleep just to be safe, but only if we suspect it was a hit?
            # We don't know if hit/miss here.
            # Adding small delay won't hurt much if cached (0.1s?), but 2.0s is annoying.
            # Let's sleep 0.1s. If miss, we rely on having few tickers or API handling it.
            # Rate limit is handled internally by testfol_api on Cache MISS (2.0s)
            # No sleep needed here.
            
        except Exception as e:
            st.warning(f"Failed to fetch data for {ticker}: {e}")
            
    return combined_prices

# ─────────────────────────────────────────────────────────────────────────────
# Main Layout
# ─────────────────────────────────────────────────────────────────────────────

st.set_page_config(page_title="Testfol Charting", layout="wide", page_icon="📈")

# --- Navigation ---
mode = st.sidebar.radio("Navigation", ["Simulator", "Documentation"], horizontal=True)

if mode == "Documentation":
    utils.render_documentation()
    st.stop()

# --- Sidebar ---
start_date, end_date, run_placeholder = render_sidebar()

# --- Main Area ---
config = render_config()

# --- Validation & Run ---
working_df = config['working_df']
alloc_preview = config['alloc_preview']
total_weight = config['total_weight']

if total_weight != 100:
    run_placeholder.error("Fix allocation (must be 100%)")
else:
    if run_placeholder.button("🚀 Run Backtest", type="primary", use_container_width=True):
        st.divider()
        with st.spinner("Running Simulation..."):
            try:
                # Logic for Pay Down Margin
                bt_cashflow = 0.0 if config['pay_down_margin'] else config['cashflow']
                shadow_cashflow = 0.0 if config['pay_down_margin'] else config['cashflow']
                
                sim_engine = config.get('sim_engine', 'standard')
                
                # GUARD RAIL: NDXMEGASIM requires Local/Hybrid engine
                # The public API does not know about this custom local ticker.
                has_ndxmega = any("NDXMEGASIM" in t for t in alloc_preview.keys())
                
                if has_ndxmega and sim_engine != 'hybrid':
                     st.info("💎 'NDXMEGASIM' detected. Automatically enabling Local Simulation (Hybrid Mode) since this ticker is not available via public API.")
                     sim_engine = 'hybrid'
                
                if sim_engine == 'hybrid':
                    # --- Hybrid (Local) Simulation ---
                    from app.core import calculations
                    
                    # 1. Fetch Component Data (Testfol API for extended history)
                    tickers = list(alloc_preview.keys())
                    prices_df = fetch_component_data(tickers, start_date, end_date)
                    
                    if prices_df.empty:
                        st.error("Failed to fetch price data for tickers.")
                        st.stop()
                        
                    # 2. Generate Chart using Testfol Data (Extended Simulated History)
                    _, _, _, _, _, port_series, _ = cached_run_shadow_backtest_v2(
                        allocation=alloc_preview, 
                        start_val=config['start_val'],
                        start_date=start_date,
                        end_date=end_date,
                        api_port_series=None, # Pure local
                        rebalance_freq="Custom",
                        cashflow=shadow_cashflow,
                        cashflow_freq=config['cashfreq'],
                        prices_df=prices_df,  # Use Testfol data for chart
                        rebalance_month=config.get('rebalance_month', 1),
                        rebalance_day=config.get('rebalance_day', 1),
                        custom_freq=config.get('custom_freq', 'Yearly')
                    )
                    
                    if port_series.empty:
                        st.error("Hybrid Simulation Failed (Chart Generation).")
                        st.stop()
                    
                    # 3. Generate Taxes using yFinance (Real Market Data Only)
                    trades_df, pl_by_year, composition_df, unrealized_pl_df, logs, _, twr_series = cached_run_shadow_backtest_v2(
                        allocation=alloc_preview, 
                        start_val=config['start_val'],
                        start_date=start_date,
                        end_date=end_date,
                        api_port_series=port_series, # Use Testfol chart for alignment
                        rebalance_freq="Custom",
                        cashflow=shadow_cashflow,
                        cashflow_freq=config['cashfreq'],
                        # prices_df NOT passed - forces yFinance usage for realistic taxes
                        # UNLESS it's NDXMEGASIM, which needs the local data
                        prices_df=prices_df if has_ndxmega else None,
                        rebalance_month=config.get('rebalance_month', 1),
                        rebalance_day=config.get('rebalance_day', 1),
                        custom_freq=config.get('custom_freq', 'Yearly')
                    )
                        
                    # Package results for render_results
                    stats = calculations.generate_stats(twr_series) # Local Stats (TWR based)
                    extra_data = {"rebalancing_events": []} # No native events for custom yet
                    
                    
                else:
                    # --- Standard (API) Simulation ---
                    port_series, stats, extra_data = cached_fetch_backtest(
                        start_date=start_date,
                        end_date=end_date,
                        start_val=config['start_val'],
                        cashflow=bt_cashflow, 
                        cashfreq="Monthly",
                        rolling=60, 
                        invest_div=config['invest_div'],
                        rebalance=config['rebalance'],
                        allocation=alloc_preview, 
                        return_raw=False,
                        include_raw=True
                    )
                    
                    # Run Shadow for Tax Lots ONLY (using API series for alignment)
                    if not port_series.empty:
                        trades_df, pl_by_year, composition_df, unrealized_pl_df, logs, _, twr_series = cached_run_shadow_backtest_v2(
                            allocation=alloc_preview, 
                            start_val=config['start_val'],
                            start_date=start_date,
                            end_date=end_date,
                            api_port_series=port_series,
                            rebalance_freq=config['rebalance'], 
                            cashflow=shadow_cashflow,
                            cashflow_freq=config['cashfreq']
                        )
                    else:
                        st.error("API returned empty portfolio data.")
                        st.stop()
                
                # Ranges for UI display
                sim_range_str = "N/A"
                if not port_series.empty:
                    s = port_series.index[0].strftime("%b %d, %Y")
                    e = port_series.index[-1].strftime("%b %d, %Y")
                    sim_range_str = f"{s} - {e}"

                shadow_range_str = "N/A"
                if not composition_df.empty:
                     if "Date" in composition_df.columns:
                         s_shadow = composition_df["Date"].iloc[0].strftime("%b %d, %Y")
                         e_shadow = composition_df["Date"].iloc[-1].strftime("%b %d, %Y")
                         shadow_range_str = f"{s_shadow} - {e_shadow}"
                     else:
                         shadow_range_str = "Invalid Composition"
                elif sim_engine == 'hybrid' and not prices_df.empty:
                    # Fallback if composition failed but prices existed (e.g. all filtered out?)
                    try:
                        shadow_range_str = f"{prices_df.index[0].strftime('%b %d, %Y')} - {prices_df.index[-1].strftime('%b %d, %Y')}"
                    except:
                        pass


                # Initialize results
                st.session_state.bt_results = {
                    "port_series": port_series,
                    "stats": stats,
                    "extra_data": extra_data,
                    "raw_response": extra_data.get("raw_response", {}),
                    "wmaint": config['wmaint'],
                    "start_val": config['start_val'],
                    "trades_df": trades_df,
                    "pl_by_year": pl_by_year,
                    "composition_df": composition_df,
                    "unrealized_pl_df": unrealized_pl_df,
                    "logs": logs,
                    "sim_range": sim_range_str,
                    "shadow_range": shadow_range_str,
                    "twr_series": twr_series, # Add TWR for Monte Carlo
                    "cashflow": config.get('cashflow', 0.0),
                    "cashfreq": config.get('cashfreq', 'None')
                }

                
            except Exception as e:
                st.error(f"Error running backtest: {e}")
                st.stop()
                
            # --- Benchmark Backtest ---
            # Run only if enabled and primary backtest succeeded
            bench_series = None
            bench_stats = None
            
            # Logic for Comparison Override
            # Logic for Comparison Override
            # 2024-12-17: Prioritize explicit Benchmark Tab settings over Hybrid Default
            
            # 1. Explicit Benchmark (from Tab)
            if config['bench_mode'] != "None":
                with st.spinner("Fetching Benchmark Data..."):
                    try:
                        bench_port_map = {}
                        if config['bench_mode'] == "Single Ticker":
                             if config['bench_ticker'].strip():
                                 bench_port_map = {config['bench_ticker'].strip(): 100.0}
                        elif config['bench_mode'] == "Custom Portfolio":
                             # Check if bench_edited_df is in config (from data editor)
                             edited = config.get('bench_edited_df')
                             if isinstance(edited, pd.DataFrame) and not edited.empty:
                                 b_df = edited.dropna(subset=["Ticker"])
                                 bench_port_map = {r["Ticker"]: r["Weight %"] for _,r in b_df.iterrows()}
                        
                        if bench_port_map:
                             # Validate rebalance freq for API (Custom is not allowed in API)
                             api_rebal = config['rebalance']
                             if api_rebal == "Custom":
                                 # Fallback to the custom_freq (e.g. Monthly/Yearly) or default to Yearly
                                 api_rebal = config.get('custom_freq', 'Yearly')
                             
                             # GUARD: Check for NDXMEGASIM in Benchmark
                             bench_has_ndx = any("NDXMEGASIM" in t for t in bench_port_map.keys())
                             
                             if bench_has_ndx:
                                 # Use Local Shadow Engine for Benchmark
                                 st.info("💎 'NDXMEGASIM' detected in Benchmark. Running Local Simulation.")
                                 b_tickers = list(bench_port_map.keys())
                                 # Reuse fetch_component_data to get local CSV + Splice
                                 b_prices = fetch_component_data(b_tickers, start_date, end_date)
                                 
                                 # Map rebalance freq for shadow engine
                                 shadow_rebal = "Custom" if config['rebalance'] == "Custom" else config['rebalance']
                                 
                                 _, _, _, _, _, b_port_series, b_twr_series = cached_run_shadow_backtest_v2(
                                    allocation=bench_port_map,
                                    start_val=config['start_val'],
                                    start_date=start_date,
                                    end_date=end_date,
                                    api_port_series=None,
                                    rebalance_freq=shadow_rebal,
                                    cashflow=config.get('cashflow', 0.0), # Use user cashflow
                                    cashflow_freq=config.get('cashfreq', 'Monthly'), # Use user freq
                                    prices_df=b_prices,
                                    rebalance_month=config.get('rebalance_month', 1),
                                    rebalance_day=config.get('rebalance_day', 1), 
                                    custom_freq=config.get('custom_freq', 'Yearly')
                                 )
                                 b_series = b_port_series
                                 b_stats = calculations.generate_stats(b_series)
                             else:
                                 b_series, b_stats, _ = cached_fetch_backtest(
                                    start_date=start_date,
                                    end_date=end_date,
                                    start_val=config['start_val'],
                                    cashflow=config.get('cashflow', 0.0), # Use user cashflow
                                    cashfreq=config.get('cashfreq', 'Monthly'), # Use user freq
                                    rolling=60,
                                    invest_div=True, 
                                    rebalance=api_rebal, 
                                    allocation=bench_port_map,
                                    return_raw=False
                                 )
                             bench_series = b_series
                             
                             # Rename for clear chart labels
                             if config['bench_mode'] == "Single Ticker":
                                 bench_series.name = f"Benchmark ({config.get('bench_ticker', 'Ticker').strip()})"
                             else:
                                 bench_series.name = "Benchmark (Custom)"
                             
                             # Fallback: Calculate stats locally if API didn't return them
                             from_api = True
                             if not b_stats:
                                 b_stats = calculations.generate_stats(b_series)
                                 from_api = False
                                 
                             bench_stats = b_stats
                             st.session_state.bt_results["bench_stats_from_api"] = from_api
                          
                    except Exception as e:
                        st.warning(f"Benchmark Fetch Failed: {e}")
                        bench_stats = None

            # 2. Hybrid Standard Comparison (Secondary Benchmark)
            # Independent check - can coexist with primary benchmark
            if sim_engine == 'hybrid' and config.get('compare_standard', False):
                 with st.spinner("Fetching Standard Rebalance Benchmark..."):
                    try:
                        # Use same allocation but standard Yearly rebalance
                        alloc_map = alloc_preview
                        
                        # Check for NDXMEGASIM in Comparison Allocation
                        has_ndx_comp = any("NDXMEGASIM" in t for t in alloc_map.keys())
                        
                        if has_ndx_comp:
                             # Use Local Shadow Engine
                             st.info("💎 'NDXMEGASIM' detected in Comparison. Running Local Simulation.")
                             c_tickers = list(alloc_map.keys())
                             c_prices = fetch_component_data(c_tickers, start_date, end_date)
                             
                             _, _, _, _, _, c_port_series, _ = cached_run_shadow_backtest_v2(
                                    allocation=alloc_map,
                                    start_val=config['start_val'],
                                    start_date=start_date,
                                    end_date=end_date,
                                    api_port_series=None,
                                    rebalance_freq="Yearly", # Comparison is Standard Yearly
                                    cashflow=config.get('cashflow', 0.0), 
                                    cashflow_freq=config.get('cashfreq', "Monthly"),
                                    prices_df=c_prices,
                                    rebalance_month=config.get('rebalance_month', 1),
                                    rebalance_day=config.get('rebalance_day', 1),
                                    custom_freq="Yearly"
                             )
                             c_series = c_port_series
                             c_stats = calculations.generate_stats(c_series)
                        else:
                             c_series, c_stats, _ = cached_fetch_backtest(
                                start_date=start_date,
                                end_date=end_date,
                                start_val=config['start_val'],
                                cashflow=config.get('cashflow', 0.0), # Use user defined cashflow
                                cashfreq=config.get('cashfreq', "Monthly"),
                                rolling=60,
                                invest_div=config['invest_div'],
                                rebalance="Yearly", # Force Yearly Standard
                                allocation=alloc_map,
                                return_raw=False
                            )
                        c_series.name = "Standard (Yearly)"
                        st.session_state.bt_results["comparison_series"] = c_series
                        st.session_state.bt_results["comparison_stats"] = c_stats
                        
                    except Exception as e:
                        st.warning(f"Comparison Benchmark failed: {e}")
            
            # Add benchmark to results
            st.session_state.bt_results["bench_series"] = bench_series
            st.session_state.bt_results["bench_stats"] = bench_stats

    # Check if we have results to display
    if "bt_results" in st.session_state:
        render_results(st.session_state.bt_results, config)

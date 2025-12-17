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
from app.ui import render_sidebar, render_config, render_results

# ─────────────────────────────────────────────────────────────────────────────
# Caching Wrappers
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_data(show_spinner="Fetching data from Testfol API...", ttl=3600)
def cached_fetch_backtest(*args, **kwargs):
    """Cached wrapper for api.fetch_backtest"""
    return fetch_backtest(*args, **kwargs)

@st.cache_data(show_spinner="Running Shadow Backtest...", ttl=3600)
def cached_run_shadow_backtest(*args, **kwargs):
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
                
                if sim_engine == 'hybrid':
                    # --- Hybrid (Local) Simulation ---
                    from app.core import calculations
                    
                    # 1. Fetch Component Data
                    tickers = list(alloc_preview.keys())
                    prices_df = fetch_component_data(tickers, start_date, end_date)
                    
                    if prices_df.empty:
                        st.error("Failed to fetch price data for tickers.")
                        st.stop()
                        
                    # 2. Run Local Simulation
                    trades_df, pl_by_year, composition_df, unrealized_pl_df, logs, port_series = cached_run_shadow_backtest(
                        allocation=alloc_preview, 
                        start_val=config['start_val'],
                        start_date=start_date,
                        end_date=end_date,
                        api_port_series=None, # Pure local
                        rebalance_freq="Custom",
                        cashflow=shadow_cashflow,
                        cashflow_freq=config['cashfreq'],
                        prices_df=prices_df,
                        rebalance_month=config.get('rebalance_month', 1),
                        rebalance_day=config.get('rebalance_day', 1)
                    )
                    
                    if port_series.empty:
                        st.error("Hybrid Simulation Failed.")
                        with st.expander("Error Logs", expanded=True):
                            for log in logs:
                                st.write(log)
                        st.stop()
                        
                        st.stop()
                        
                    # Package results for render_results
                    stats = calculations.generate_stats(port_series) # Local Stats
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
                        trades_df, pl_by_year, composition_df, unrealized_pl_df, logs, _ = cached_run_shadow_backtest(
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
                    "logs": logs
                }
                
            except Exception as e:
                st.error(f"Error running backtest: {e}")
                st.stop()
                
            # --- Benchmark Backtest ---
            # Run only if enabled and primary backtest succeeded
            bench_series = None
            bench_stats = None
            
            # Logic for Comparison Override
            run_standard_bench = False
            if sim_engine == 'hybrid' and config.get('compare_standard', False):
                run_standard_bench = True
            
            if run_standard_bench:
                 with st.spinner("Fetching Standard Rebalance Benchmark..."):
                    try:
                        # Use same allocation but standard Yearly rebalance
                        alloc_map = alloc_preview
                        
                        b_series, b_stats, _ = cached_fetch_backtest(
                            start_date=start_date,
                            end_date=end_date,
                            start_val=config['start_val'],
                            cashflow=0.0, # Pure performance comparison
                            cashfreq="Monthly",
                            rolling=60,
                            invest_div=config['invest_div'],
                            rebalance="Yearly", # Force Yearly Standard
                            allocation=alloc_map,
                            return_raw=False
                        )
                        bench_series = b_series
                        bench_series.name = "Standard (Yearly)"
                        
                        # Capture benchmark stats - this was missing!
                        from_api = True
                        if not b_stats:
                            b_stats = calculations.generate_stats(b_series)
                            from_api = False
                            
                        bench_stats = b_stats
                        st.session_state.bt_results["bench_stats_from_api"] = from_api
                        
                        config['bench_mode'] = "Standard Rebalance" # Override for UI label
                        
                    except Exception as e:
                        st.warning(f"Comparison Benchmark failed: {e}")
            
            elif config['bench_mode'] != "None":
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
                             b_series, b_stats, _ = cached_fetch_backtest(
                                start_date=start_date,
                                end_date=end_date,
                                start_val=config['start_val'],
                                cashflow=0.0, # Pure performance
                                cashfreq="Monthly",
                                rolling=60,
                                invest_div=True, 
                                rebalance=config['rebalance'], 
                                allocation=bench_port_map,
                                return_raw=False
                             )
                             bench_series = b_series
                             
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
            
            # Add benchmark to results
            st.session_state.bt_results["bench_series"] = bench_series
            st.session_state.bt_results["bench_stats"] = bench_stats

    # Check if we have results to display
    if "bt_results" in st.session_state:
        render_results(st.session_state.bt_results, config)

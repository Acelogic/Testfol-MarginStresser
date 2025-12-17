#!/usr/bin/env python3
# ─────────────────────────────────────────────────────────────────────────────
# testfol_charting.py
#
# New Streamlit App for Testfol Backtesting with Multi-Timeframe Candlesticks
# ─────────────────────────────────────────────────────────────────────────────

import pandas as pd
import streamlit as st
from app.services import fetch_backtest
from app.core import run_shadow_backtest
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
        with st.spinner("Fetching Backtest Data..."):
            try:
                # Logic for Pay Down Margin
                bt_cashflow = 0.0 if config['pay_down_margin'] else config['cashflow']
                shadow_cashflow = 0.0 if config['pay_down_margin'] else config['cashflow']

                # 1. Fetch Standard Backtest (Total Return)
                port_series, stats, extra_data = cached_fetch_backtest(
                    start_date=start_date,
                    end_date=end_date,
                    start_val=config['start_val'],
                    cashflow=bt_cashflow, 
                    cashfreq="Monthly",
                    rolling=60, # Assuming 60 from original code
                    invest_div=config['invest_div'],
                    rebalance=config['rebalance'],
                    allocation=alloc_preview, 
                    return_raw=False,
                    include_raw=True
                )
                
                # 2. Run Shadow Backtest (Tax Lots & Realized Gains)
                if not port_series.empty:
                    trades_df, pl_by_year, composition_df, unrealized_pl_df, logs = cached_run_shadow_backtest(
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
                
                # Initialize results with API data
                st.session_state.bt_results = {
                    "port_series": port_series,
                    "stats": stats,
                    "extra_data": extra_data,
                    "raw_response": extra_data.get("raw_response", {}),
                    "wmaint": config['wmaint'],
                    "start_val": config['start_val'],
                    # Shadow Backtest Results
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
                             bench_series.name = "Benchmark"
                             
                    except Exception as e:
                        st.warning(f"Benchmark failed: {e}")

                # Add benchmark to results
                st.session_state.bt_results["bench_series"] = bench_series

    # Check if we have results to display
    if "bt_results" in st.session_state:
        render_results(st.session_state.bt_results, config)

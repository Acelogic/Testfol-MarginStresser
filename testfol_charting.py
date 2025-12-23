#!/usr/bin/env python3
# ─────────────────────────────────────────────────────────────────────────────
# testfol_charting.py
#
# New Streamlit App for Testfol Backtesting with Multi-Portfolio Support
# ─────────────────────────────────────────────────────────────────────────────

import pandas as pd
import streamlit as st
import time
import os
from app.services import fetch_backtest
from app.core import run_shadow_backtest, calculations
from app.common import utils
from app.ui import render_sidebar, render_config, render_results, asset_explorer, charts

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
    Fetches historical data for each ticker individually via Testfol API (or local).
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
                        df_sim = pd.read_csv(csv_path)
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
                        import yfinance as yf
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
            combined_prices[base] = series
            
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
# Note: With multi-portfolio, we validate per portfolio inside the loop or pre-check.
# We trust configuration.py to manage weight validation visually.

if run_placeholder.button("🚀 Run Backtest", type="primary", use_container_width=True):
    st.divider()
    with st.spinner("Running Simulations..."):
        try:
            results_list = []
            
            # Default to current config as single portfolio if 'portfolios' missing (Legacy Support)
            portfolios = config.get('portfolios', [])
            if not portfolios:
                # Construct pseudo-portfolio from legacy config keys
                portfolios = [{
                    "id": "legacy",
                    "name": "Portfolio 1",
                    "alloc_df": config.get('edited_df', pd.DataFrame([{"Ticker":"SPY", "Weight %":100, "Maint %":25}])), 
                    "rebalance": {
                        "mode": "Custom" if config.get('sim_engine') == 'hybrid' else "Standard",
                        "freq": config.get('custom_freq', 'Yearly'),
                        "month": config.get('rebalance_month', 1),
                        "day": config.get('rebalance_day', 1),
                        "compare_std": config.get('compare_standard', False)
                    },
                    "cashflow": {
                        "start_val": config.get('start_val', 10000),
                        "amount": config.get('cashflow', 0),
                        "freq": config.get('cashfreq', 'Monthly'),
                        "invest_div": config.get('invest_div', True), 
                        "pay_down_margin": config.get('pay_down_margin', False)
                    }
                }]
            
            bench_series_list = []

            for p in portfolios:
                # Prepare params
                if 'alloc_df' in p and not p['alloc_df'].empty:
                     alloc_map = dict(zip(p['alloc_df']['Ticker'], p['alloc_df']['Weight %']))
                     
                     # Calculate Weighted Maintenance for this portfolio
                     # Need to normalize weights first
                     total_w = sum(alloc_map.values())
                     d_maint = config.get('default_maint', 25.0)
                     
                     current_wmaint = 0.0
                     if total_w > 0:
                         for idx, row in p['alloc_df'].iterrows():
                             w = row['Weight %']
                             m = row.get('Maint %', d_maint)
                             current_wmaint += (w/100) * (m/100)
                     else:
                         current_wmaint = d_maint / 100.0
                else:
                     continue

                # Determine Engine
                has_ndxmega = any(("NDXMEGASIM" in t or "NDXMEGA2SIM" in t) for t in alloc_map.keys())
                use_local_engine = has_ndxmega
                
                # Extract Settings (Global Cashflow)
                gcf = config.get('global_cashflow', {})
                pay_down = gcf.get('pay_down_margin', False)
                bt_cashflow = 0.0 if pay_down else gcf.get('amount', 0.0)
                shadow_cashflow = 0.0 if pay_down else gcf.get('amount', 0.0)
                start_val = gcf.get('start_val', 10000.0)
                cf_freq = gcf.get('freq', 'Monthly')
                invest_div = gcf.get('invest_div', True)


                # Rebalance
                reb = p.get('rebalance', {})
                r_mode = reb.get('mode', 'Standard')
                r_freq = reb.get('freq', 'Yearly')
                
                port_series = pd.Series(dtype=float)
                stats = {}
                trades_df = pd.DataFrame()
                
                if not use_local_engine:
                    # --- API Path ---
                    
                    # Calculate Offsets
                    calc_rebal_offset = 0
                    
                    if r_mode == "Custom":
                        r_month = reb.get('month', 1)
                        r_day = reb.get('day', 1)
                        
                        try:
                            if r_freq == "Yearly":
                                end_of_year = pd.Timestamp("2024-12-31")
                                target_date = pd.Timestamp(f"2024-{r_month}-{r_day}")
                                days_remaining = (end_of_year - target_date).days
                                calc_rebal_offset = int(days_remaining * (252.0 / 366.0))
                                calc_rebal_offset = max(0, calc_rebal_offset)
                            else:
                                days_remaining = 31 - r_day
                                calc_rebal_offset = int(days_remaining * (21.0 / 31.0))
                        except Exception as e:
                            # st.warning(f"Offset Error: {e}")
                            calc_rebal_offset = 0
                    
                    # Fetch
                    port_series, stats_api, extra_data = cached_fetch_backtest(
                        start_date=start_date,
                        end_date=end_date,
                        start_val=start_val,
                        cashflow=bt_cashflow, 
                        cashfreq=cf_freq,
                        rolling=60, 
                        invest_div=invest_div,
                        rebalance=r_freq,
                        rebalance_offset=calc_rebal_offset,
                        allocation=alloc_map, 
                        return_raw=False,
                        include_raw=True
                    )
                    stats = stats_api
                    
                    # Run Shadow (Tax Only) - Using Global Tax Config for now
                    if not port_series.empty:
                        trades_df, pl_by_year, composition_df, unrealized_pl_df, logs, _, twr_series = cached_run_shadow_backtest_v2(
                            allocation=alloc_map, 
                            start_val=start_val,
                            start_date=start_date,
                            end_date=end_date,
                            api_port_series=port_series,
                            rebalance_freq="Custom", # Force logic to not skip
                            cashflow=shadow_cashflow,
                            cashflow_freq=cf_freq,
                            invest_dividends=invest_div,
                            pay_down_margin=pay_down,
                            tax_config=config, 
                            custom_rebal_config=reb if r_mode == "Custom" else {}
                        )
                else:
                    # --- Pure Local Path (NDXMEGASIM) ---
                     tickers = list(alloc_map.keys())
                     prices_df = fetch_component_data(tickers, start_date, end_date)

                     trades_df, pl_by_year, composition_df, unrealized_pl_df, logs, port_series, twr_series = cached_run_shadow_backtest_v2(
                        allocation=alloc_map, 
                        start_val=start_val,
                        start_date=start_date,
                        end_date=end_date,
                        api_port_series=None,
                        rebalance_freq=reb.get('freq', 'Yearly'),
                        cashflow=shadow_cashflow,
                        cashflow_freq=cf_freq,
                        invest_dividends=invest_div,
                        pay_down_margin=pay_down,
                        tax_config=config,
                        custom_rebal_config=reb if r_mode == "Custom" else {},
                        prices_df=prices_df,
                        rebalance_month=reb.get('month', 1),
                        rebalance_day=reb.get('day', 1),
                        custom_freq=reb.get('freq', 'Yearly')
                    )

                     # Generate Stats locally
                     if not port_series.empty:
                        stats = calculations.generate_stats(twr_series if twr_series is not None else port_series)
                     else:
                        stats = {}

                # Add result
                # Add result
                results_list.append({
                    "name": p.get('name', 'Portfolio'),
                    "series": port_series,
                    "port_series": port_series, # Alias for results.py
                    "stats": stats,
                    "trades": trades_df,
                    "trades_df": trades_df, # Alias for results.py
                    "pl_by_year": pl_by_year,
                    "unrealized_pl_df": unrealized_pl_df,
                    "logs": logs if 'logs' in locals() else [],
                    "composition": composition_df if 'composition_df' in locals() else pd.DataFrame(),
                    "composition_df": composition_df if 'composition_df' in locals() else pd.DataFrame(),
                    "raw_response": extra_data if 'extra_data' in locals() else {},
                    "start_val": start_val,
                    "twr_series": twr_series if 'twr_series' in locals() else None,
                    "sim_range": f"{start_date} to {end_date}",
                    "shadow_range": f"{start_date} to {end_date}",
                    "wmaint": current_wmaint
                })
                
                # --- Comparisons (Vs Standard) ---
                if reb.get("compare_std", False) and r_mode == "Custom": 
                    # Run a Standard Version of this same portfolio
                    try:
                        std_series, std_stats, _ = cached_fetch_backtest(
                            start_date=start_date,
                            end_date=end_date,
                            start_val=start_val,
                            cashflow=bt_cashflow, 
                            cashfreq=cf_freq,
                            rolling=60, 
                            invest_div=invest_div,
                            rebalance="Yearly", # Force Yearly Standard
                            allocation=alloc_map, 
                            return_raw=False
                        )
                        std_series.name = f"{p.get('name')} (Standard)"
                        bench_series_list.append(std_series)
                    except Exception as e:
                        print(f"Comparison failed: {e}")

            # Store for Rendering
            st.session_state.results_list = results_list
            st.session_state.bench_series_list = bench_series_list
            
        except Exception as e:
            st.error(f"Error running backtest: {e}")
            import traceback
            st.code(traceback.format_exc())
            # Don't stop here, let it try to render what it has or nothing

# --- Render Results (Outside Button Logic to Persist) ---
if "results_list" in st.session_state and st.session_state.results_list:
    results_list = st.session_state.results_list
    bench_series_list = st.session_state.get("bench_series_list", [])

    st.divider()
    
    # --- Render Main Chart ---
    charts.render_multi_portfolio_chart(results_list, benchmarks=bench_series_list, log_scale=config.get('log_scale', True))
    
    # --- Render Detailed Results (synced with config tab) ---
    st.divider()
    
    # Use the active tab index from configuration tabs
    active_idx = st.session_state.get('active_tab_idx', 0)
    # Validate index
    if active_idx >= len(results_list):
        active_idx = 0
        
    res = results_list[active_idx]
    st.markdown(f"### 📋 {res['name']} Details")
    render_results(res, config, portfolio_name=res['name'])

#!/usr/bin/env python3
# ─────────────────────────────────────────────────────────────────────────────
# testfol_charting.py
#
# New Streamlit App for Testfol Backtesting with Multi-Portfolio Support
# ─────────────────────────────────────────────────────────────────────────────

import pandas as pd
import streamlit as st
import time
from app.services import fetch_backtest
from app.services.data_service import fetch_component_data
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
start_date, end_date, bearer_token, run_placeholder = render_sidebar()

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
                
                # Initialize variables that may not be set in all code paths
                extra_data = {}
                df_rets = None
                
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
                        include_raw=True,
                        bearer_token=bearer_token
                    )
                    stats = stats_api
                    print(f"DEBUG APP: Tickers={alloc_map.keys()} | API Stats CAGR={stats.get('cagr')}")

                    # --- Extract TWR Series from API for Accurate Smart Stats ---
                    # We prefer API TWR (Official) over Local Shadow TWR (Conservative)
                    api_twr_series = None
                    if extra_data and 'daily_returns' in extra_data:
                        d_rets = extra_data['daily_returns']
                        # Format: [[date_str, pct, val], ...]
                        if d_rets:
                            try:
                                # Convert to DataFrame
                                df_rets = pd.DataFrame(d_rets, columns=['Date', 'Pct', 'Val'])
                                df_rets['Date'] = pd.to_datetime(df_rets['Date'])
                                df_rets = df_rets.set_index('Date').sort_index()
                                
                                # Calculate Factor (1 + r)
                                # Pct is percentage (e.g. -3.035)
                                df_rets['Factor'] = 1 + (df_rets['Pct'] / 100.0)
                                
                                # Cumulative Product to get TWR Index
                                # Start at 1.0 (insert initial)
                                start_dt = df_rets.index[0] - pd.Timedelta(days=1) # Aproximation
                                
                                # Let's just use cumprod directly
                                twr_series_vals = df_rets['Factor'].cumprod()
                                
                                # Prepend 1.0 at start date (if possible, or just use first val)
                                # Actually, cumprod starts at (1+r1).
                                # We want a series representing Unit Value.
                                # Normalized to 1.0 at start of data.
                                # We can just pass this series. Our charts.py logic rebases it anyway (twr / twr[0]).
                                api_twr_series = twr_series_vals
                                api_twr_series.name = "TWR (API)"
                            except Exception as e:
                                print(f"Failed to build API TWR: {e}")
                    
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

                        # Override TWR if API provided it (More accurate than Shadow)
                        if api_twr_series is not None:
                            twr_series = api_twr_series
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
                port_name = p.get('name', 'Portfolio')
                results_list.append({
                    'name': port_name,
                    'series': port_series,
                    'port_series': port_series, # Alias for results.py
                    'stats': stats,
                    'twr_series': twr_series if 'twr_series' in locals() else None,
                    'daily_returns_df': df_rets,  # Now always defined (may be None)
                    'is_local': use_local_engine,
                    "trades": trades_df,
                    "trades_df": trades_df, # Alias for results.py
                    "pl_by_year": pl_by_year,
                    "unrealized_pl_df": unrealized_pl_df,
                    "logs": logs if 'logs' in locals() else [],
                    "composition": composition_df if 'composition_df' in locals() else pd.DataFrame(),
                    "composition_df": composition_df if 'composition_df' in locals() else pd.DataFrame(),
                    "raw_response": extra_data if 'extra_data' in locals() else {},
                    "start_val": start_val,
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
                            return_raw=False,
                            bearer_token=bearer_token
                        )
                        std_series.name = f"{p.get('name')} (Standard)"
                        bench_series_list.append(std_series)
                    except Exception as e:
                        print(f"Failed standard comparison: {e}")

            # --- Pass 2: Auto-Align Start Dates (Re-Simulate from Common Start) ---
            # User Request: "if we have a common date out of that range we should ignore the global and just use the common as the start date"
            # To fix "End Value" discrepancies, we must start fresh at common date with full capital.
            
            # 1. Determine Common Start
            start_dates = []
            for res in results_list:
                if res.get('series') is not None and not res['series'].empty:
                    start_dates.append(res['series'].index.min())
                    
                    
            if bench_series_list:
                for b in bench_series_list:
                    if b is not None and not b.empty:
                        start_dates.append(b.index.min())
                            
            common_start = max(start_dates) if start_dates else None
            
            # 2. Re-Simulate if needed
            if common_start:
                global_start_val = config.get('global_cashflow', {}).get('start_val', 10000.0)
                
                for i, res in enumerate(results_list):
                    series = res.get('series')
                    if series is None or series.empty: continue
                    
                    # If this portfolio started significantly earlier than common_start
                    if series.index[0] < common_start:
                        # We need to re-simulate starting from common_start
                        # Method: Use TWR series to project value from scratch
                        
                        twr = res.get('twr_series')
                        if twr is not None and not twr.empty:
                            # Slice TWR to start at common_start
                            twr_slice = twr[twr.index >= common_start]
                            
                            if not twr_slice.empty:
                                # Normalize TWR to start at 1.0 at common_start
                                scale_factor = twr_slice / twr_slice.iloc[0]
                                new_series = scale_factor * global_start_val
                                
                                # Replace in results
                                res['series'] = new_series
                                res['port_series'] = new_series # Fix KeyError: Sync Alias
                                
                                # Preserve original API stats before recalculating
                                res['original_api_stats'] = res.get('stats', {})
                                res['stats'] = calculations.generate_stats(new_series)
                                res['stats_source'] = 'rebased'  # Flag for UI disambiguation
                                
                                # Update description
                                # st.toast(f"Re-aligned {res['name']} to start {common_start.date()}")
                                
            # -------------------------------------------------------------------
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
    
    # --- Render Main Chart (Raw Data, No Transformations) ---
    charts.render_multi_portfolio_chart(
        results_list, 
        benchmarks=bench_series_list, 
        log_scale=config.get('log_scale', True)
    )
    
    # --- Render Detailed Results (synced with config tab) ---
    st.divider()
    
    # Use the active tab index from configuration tabs
    active_idx = st.session_state.get('active_tab_idx', 0)
    # Validate index
    if active_idx >= len(results_list):
        active_idx = 0
    
    # Calculate Common Start Date (for syncing results with chart)
    start_dates = []
    for r in results_list:
        if not r['series'].empty:
            start_dates.append(r['series'].index[0])
    common_start = max(start_dates) if start_dates else None
        
    res = results_list[active_idx]
    st.markdown(f"### 📋 {res['name']} Details")
    render_results(res, config, portfolio_name=res['name'], clip_start_date=common_start)

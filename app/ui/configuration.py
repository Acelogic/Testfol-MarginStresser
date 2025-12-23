import streamlit as st
import pandas as pd
from app.common import utils
from app.services import testfol_api as api
from . import asset_explorer

def render():
    """Renders the configuration tabs and returns a config dictionary."""
    
    st.subheader("Strategy Configuration")
    
    tab_port, tab_margin, tab_bench, tab_asset, tab_settings = st.tabs(["💼 Portfolio", "🏦 Margin & Financing", "📊 Benchmark", "🧩 Asset Explorer", "⚙️ Settings"])
    
    config = {}

    with tab_port:
        # --- Top Settings Grid ---
        c_set1, c_set2, c_set3 = st.columns(3)
        
        with c_set1:
            st.markdown("##### 💰 Capital & Cashflow")
            config['start_val'] = utils.num_input("Start Value ($)", "start_val", 10000.0, 1000.0, on_change=utils.sync_equity)
            config['cashflow'] = utils.num_input("Cashflow ($)", "cashflow", 0.0, 100.0)
            
            sc1, sc2 = st.columns(2)
            with sc1:
                config['cashfreq'] = st.selectbox("Freq", ["Monthly", "Quarterly", "Yearly"], index=0, label_visibility="collapsed")
            with sc2:
                 config['invest_div'] = st.checkbox("Re-invest Divs", value=True)

        with c_set2:
            st.markdown("##### 📅 Rebalancing")
            sim_mode = st.radio(
                "Mode",
                ["Standard", "Custom"],
                index=0,
                horizontal=True,
                label_visibility="collapsed",
                help="**Standard**: End of Period (API).\n**Custom**: Specific Date (Hybrid)."
            )
            config['sim_engine'] = "hybrid" if "Custom" in sim_mode else "standard"
            
            if config['sim_engine'] == "hybrid":
                # Layout for Custom Date
                rc1, rc2 = st.columns([1, 1])
                with rc1:
                    config['custom_freq'] = st.selectbox("Frequency", ["Yearly", "Quarterly", "Monthly"], index=0)
                with rc2:
                    if config['custom_freq'] == "Yearly":
                        config['rebalance_month'] = st.selectbox("Month", range(1, 13), format_func=lambda x: pd.to_datetime(f"2024-{x}-1").strftime("%b"), index=0)
                    else:
                        config['rebalance_month'] = 1
                
                config['rebalance_day'] = st.number_input("Day of Month", 1, 31, 15)
                config['rebalance'] = "Custom"
                config['compare_standard'] = st.checkbox("Vs Standard", value=True)
            else:
                config['rebalance'] = st.selectbox("Freq", ["Yearly", "Quarterly", "Monthly"], index=0, label_visibility="collapsed")
                config['rebalance_month'] = 1
                config['rebalance_day'] = 1
                config['compare_standard'] = False
                
            config['pay_down_margin'] = st.checkbox("Pay Down Margin", value=False)

        with c_set3:
            st.markdown("##### 🏛️ Tax Settings")
            config['use_std_deduction'] = st.checkbox("Standard Deduction", value=True)
            
            tax_method_selection = st.selectbox(
                "Method",
                ["Historical Smart", "Historical Max Rate", "2025 Fixed Brackets"],
                index=0,
                help="**Historical Smart**: Uses actual tax brackets and inclusion rates from each specific year.\n**Historical Max**: Applies the highest historical capital gains rate for that year.\n**2025 Fixed**: Applies today's (2025) tax brackets to all past years."
            )
            
            with st.expander("Details", expanded=False):
                config['filing_status'] = st.selectbox("Status", ["Single", "Married Joint", "Head of Household"], index=0)
                config['state_tax_rate'] = st.number_input("State Tax %", 0.0, 20.0, 0.0, 0.1) / 100.0
                config['other_income'] = st.number_input("Other Income", 0.0, 10000000.0, 100000.0, 5000.0)

            if "Smart" in tax_method_selection:
                config['tax_method'] = "smart"
            elif "Max" in tax_method_selection:
                config['tax_method'] = "historical_max"
            else:
                config['tax_method'] = "2025_fixed"

        st.divider()
        
        # --- Allocation Table (Full Width) ---
        st.markdown("##### 🥧 Asset Allocation")
        
        _default = [
            {"Ticker":"AAPL?L=2","Weight %":7.5,"Maint %":50},
            {"Ticker":"MSFT?L=2","Weight %":7.5,"Maint %":50},
            {"Ticker":"AVGO?L=2","Weight %":7.5,"Maint %":50},
            {"Ticker":"AMZN?L=2","Weight %":7.5,"Maint %":50},
            {"Ticker":"META?L=2","Weight %":7.5,"Maint %":50},
            {"Ticker":"NVDA?L=2","Weight %":7.5,"Maint %":50},
            {"Ticker":"GOOGL?L=2","Weight %":7.5,"Maint %":50},
            {"Ticker":"TSLA?L=2","Weight %":7.5,"Maint %":50},
            {"Ticker":"GLD","Weight %":20,"Maint %":25},
            {"Ticker":"VXUS","Weight %":15,"Maint %":25},
            {"Ticker":"TQQQ","Weight %":5,"Maint %":75},
        ]

        if "alloc_df" not in st.session_state:
            st.session_state.alloc_df = pd.DataFrame(_default)

        # Use full container width
        edited_df = st.data_editor(
            st.session_state.alloc_df,
            key=f"alloc_editor_{st.session_state.get('reload_counter', 0)}",
            num_rows="dynamic",
            column_order=["Ticker", "Weight %", "Maint %"],
            column_config={
                "Weight %": st.column_config.NumberColumn(
                    min_value=0.0, max_value=100.0, step=0.01, format="%.2f"
                ),
                "Maint %": st.column_config.NumberColumn(
                    min_value=0.0, max_value=100.0, step=0.1, format="%.1f"
                ),
            },
            use_container_width=True
        )
        
        # Store edited data for save functionality
        st.session_state.current_edited_df = edited_df
        config['edited_df'] = edited_df

    with tab_margin:
        # Move Tax Simulation to top to control state of other inputs
        tax_sim_mode = st.radio(
            "Tax Payment Simulation",
            ["None (Gross)", "Pay from Cash", "Pay with Margin"],
            index=0,
            horizontal=True, # Make it horizontal to save space at top
            help="**None (Gross)**: Show raw pre-tax returns.\n**Pay from Cash**: Simulate selling shares to pay taxes (reduces equity).\n**Pay with Margin**: Simulate borrowing to pay taxes (increases loan)."
        )
        
        # Map selection to flags
        config['pay_tax_margin'] = (tax_sim_mode == "Pay with Margin")
        config['pay_tax_cash'] = (tax_sim_mode == "Pay from Cash")
        
        # Disable margin inputs if "Pay from Cash" is selected (per user request)
        # This implies a "Cash Only" mindset for this mode
        margin_disabled = config['pay_tax_cash']

        c1, c2 = st.columns(2)
        
        with c1:
            st.markdown("##### Loan Configuration")
            config['starting_loan'] = utils.num_input(
                "Starting Loan ($)", "starting_loan", 0.0, 100.0,
                on_change=utils.sync_equity,
                disabled=margin_disabled
            )
            config['equity_init'] = utils.num_input(
                "Initial Equity %", "equity_init", 100.0, 1.0,
                on_change=utils.sync_loan,
                disabled=margin_disabled
            )
            
            # Calculate leverage for display (handle zero division)
            if config['start_val'] != config['starting_loan']:
                current_lev = config['start_val'] / (config['start_val'] - config['starting_loan'])
            else:
                current_lev = 0.0
                
            st.caption(f"Current Leverage: **{current_lev:.2f}x**")
            
        with c2:
            st.markdown("##### Rates & Maintenance")
            config['rate_annual'] = utils.num_input("Interest % per year", "rate_annual", 8.0, 0.5, disabled=margin_disabled)
            config['draw_monthly'] = utils.num_input("Monthly Draw ($)", "draw_monthly", 0.0, 100.0) # Draws allowed in cash mode? Assuming yes.
            config['default_maint'] = utils.num_input("Default Maint %", "default_maint", 25.0, 1.0, disabled=margin_disabled)
            
            # Portfolio Margin Toggle
            config['pm_enabled'] = st.checkbox("Enable Portfolio Margin (PM)", value=False, help="Enforces $100k Minimum Equity requirement.", disabled=margin_disabled)


    with tab_bench:
        st.markdown("##### Benchmark Configuration")
        st.info("Compare your strategy against a benchmark (Gross Total Return).")
        
        config['bench_mode'] = st.radio("Benchmark Mode", ["None", "Single Ticker", "Custom Portfolio"], horizontal=True)
        
        if config['bench_mode'] == "Single Ticker":
            config['bench_ticker'] = st.text_input("Benchmark Ticker", "SPY", help="Enter a single ticker symbol (e.g. SPY, VTI, QQQ).")
            st.caption("Standard 100% allocation to this ticker.")
            
        elif config['bench_mode'] == "Custom Portfolio":
            st.markdown("Define Benchmark Allocation:")
            
            if "bench_alloc_df" not in st.session_state:
                st.session_state.bench_alloc_df = pd.DataFrame([{"Ticker":"SPY","Weight %":60}, {"Ticker":"O","Weight %":40}])
                
            config['bench_edited_df'] = st.data_editor(
                st.session_state.bench_alloc_df,
                num_rows="dynamic",
                use_container_width=True,
                key="bench_editor"
            )

    with tab_asset:
        # Asset Explorer is self-contained.
        # It doesn't modify the simulation config, just visualizes data.
        asset_explorer.render_asset_explorer()

    with tab_settings:
        c1, c2 = st.columns(2)
        with c1:
            config['chart_style'] = st.selectbox(
                "Chart Style",
                ["Classic (Combined)", "Classic (Dashboard)", "Candlestick"],
                index=0
            )
            config['timeframe'] = st.selectbox(
                "Chart Timeframe",
                ["1D", "1W", "1M", "3M", "1Y"],
                index=2
            )
            config['log_scale'] = st.checkbox("Logarithmic Scale", value=True)
            
            config['log_opts'] = {}
            if config['chart_style'] == "Classic (Dashboard)":
                st.markdown("**Dashboard Log Scales:**")
                config['log_opts']['portfolio'] = st.checkbox("Portfolio Chart", value=False, key="log_portfolio")
                config['log_opts']['leverage'] = st.checkbox("Leverage Chart", value=False, key="log_leverage")
                config['log_opts']['margin'] = st.checkbox("Margin Debt Chart", value=False, key="log_margin")
        with c2:
            config['show_range_slider'] = st.checkbox("Show Range Slider", value=True)
            config['show_volume'] = st.checkbox("Show Range/Volume Panel", value=True)
            
            st.markdown("---")
            st.markdown("**Cache Management**")
            if st.button("🗑️ Clear API Cache", help="Remove cached API responses to force fresh data fetches"):
                import shutil
                import os
                cache_dir = "data/api_cache"
                if os.path.exists(cache_dir):
                    shutil.rmtree(cache_dir)
                    os.makedirs(cache_dir, exist_ok=True)
                    st.success("Cache cleared!")
                else:
                    st.info("Cache is already empty.")
            
    # --- Validation ---
    # Handle case where data editor is empty or hasn't populated yet (e.g., after loading)
    edited_df = config.get('edited_df')
    working_df = pd.DataFrame()
    try:
        if isinstance(edited_df, pd.DataFrame) and not edited_df.empty and "Ticker" in edited_df.columns:
            working_df = edited_df.dropna(subset=["Ticker"]).loc[lambda df: df["Ticker"].str.strip() != ""]
        else:
            # Use session state data if editor data isn't ready
            if "alloc_df" in st.session_state and not st.session_state.alloc_df.empty:
                working_df = st.session_state.alloc_df.dropna(subset=["Ticker"]).loc[lambda df: df["Ticker"].str.strip() != ""]
    except (KeyError, AttributeError):
        pass # Fallback

    # Ensure working_df has the required columns before passing to API
    if working_df.empty or "Ticker" not in working_df.columns:
        # Create empty but valid DataFrame with required columns
        working_df = pd.DataFrame(columns=["Ticker", "Weight %", "Maint %"])

    alloc_preview, maint_preview = api.table_to_dicts(working_df)
    total_weight = sum(alloc_preview.values())
    
    # Calculate wmaint for display
    # Use default_maint from config
    d_maint = config.get('default_maint', 25.0)
    wmaint = sum(
        (wt/100) * (maint_preview.get(t.split("?")[0], d_maint)/100)
        for t, wt in alloc_preview.items()
    )

    with tab_port:
        st.markdown("---")
        c1, c2 = st.columns(2)
        c1.metric("Total Allocation", f"{total_weight:.2f}%", delta=None if total_weight == 100 else "Must be 100%", delta_color="off" if total_weight == 100 else "inverse")
        c2.metric("Weighted Maint Req", f"{wmaint*100:.2f}%")
        
    config['working_df'] = working_df
    config['alloc_preview'] = alloc_preview
    config['maint_preview'] = maint_preview
    config['total_weight'] = total_weight
    config['wmaint'] = wmaint
    
    return config

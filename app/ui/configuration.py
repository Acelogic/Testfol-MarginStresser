import streamlit as st
import pandas as pd
from app.common import utils
from app.services import testfol_api as api
from . import asset_explorer
from . import ndx_scanner

def render():
    """Renders the configuration tabs and returns a config dictionary."""
    
    st.subheader("Strategy Configuration")
    
    tab_port, tab_margin, tab_asset, tab_ndx, tab_settings = st.tabs(["💼 Portfolio", "🏦 Margin & Financing", "🧩 Asset Explorer", "📊 NDX Scanner", "⚙️ Settings"])
    
    
    config = {}

    # Per-portfolio widget keys — stable across tab switches, popped on switch
    _PF_KEYS = [
        "p_name", "p_rmode", "p_rfreq", "p_rmon", "p_rday", "p_cmp",
        "p_rthresh", "p_rfreq_tc", "p_rthresh_tc", "p_rfreq_std",
        "p_editor",
    ]

    def _pop_pf_keys():
        for k in _PF_KEYS:
            st.session_state.pop(k, None)

    def _portfolio_fragment():
        # --- initialize state ---
        if "portfolios" not in st.session_state:
            _default_alloc = pd.DataFrame([
                {"Ticker": "NDXMEGASIM?L=2", "Weight %": 60.0, "Maint %": 50.0},
                {"Ticker": "GLDSIM", "Weight %": 20.0, "Maint %": 25.0},
                {"Ticker": "VXUSSIM", "Weight %": 15.0, "Maint %": 25.0},
                {"Ticker": "QQQSIM?L=3", "Weight %": 5.0, "Maint %": 75.0}
            ])
            st.session_state.portfolios = [{
                "id": "p1",
                "name": "NDXMEGASPLIT",
                "alloc_df": _default_alloc,
                "rebalance": {
                    "mode": "Custom", "freq": "Yearly", "month": 1, "day": 1, "compare_std": False, "threshold_pct": 5.0
                },
                "cashflow": {
                     "start_val": 10000.0, "amount": 0.0, "freq": "Monthly", "invest_div": True, "pay_down_margin": False
                }
            }]

        if "global_cashflow" not in st.session_state:
            st.session_state.global_cashflow = {
                "start_val": 10000.0, "amount": 0.0,
                "freq": "Monthly", "invest_div": True, "pay_down_margin": False
            }

        if "active_tab_idx" not in st.session_state:
            st.session_state.active_tab_idx = 0

        # Sync active portfolio name from the stable text_input key
        if "p_name" in st.session_state:
            _aidx = st.session_state.active_tab_idx
            if _aidx < len(st.session_state.portfolios):
                st.session_state.portfolios[_aidx]["name"] = st.session_state["p_name"]

        portfolio_names = [port["name"] for port in st.session_state.portfolios]

        # Build unique display names for segmented control (handles duplicate names)
        display_names = list(portfolio_names)
        _seen = {}
        for i, name in enumerate(display_names):
            count = _seen.get(name, 0) + 1
            _seen[name] = count
            if count > 1:
                display_names[i] = f"{name} ({count})"

        # Clamp active_tab_idx
        idx = min(st.session_state.active_tab_idx, len(display_names) - 1)
        if idx < 0:
            idx = 0
        st.session_state.active_tab_idx = idx

        # --- Render Global Section ---
        st.markdown("##### 💰 Global Capital & Cashflow")
        gc1, gc2, gc3, gc4, gc5 = st.columns([2, 2, 2, 1.5, 1.5])
        with gc1:
            st.session_state.global_cashflow["start_val"] = utils.num_input("Start Value ($)", "g_start", st.session_state.global_cashflow["start_val"], 1000.0)
        with gc2:
            st.session_state.global_cashflow["amount"] = utils.num_input("Cashflow ($)", "g_cf", st.session_state.global_cashflow["amount"], 100.0)
        with gc3:
             st.session_state.global_cashflow["freq"] = st.selectbox("Freq", ["Monthly", "Quarterly", "Yearly"], index=["Monthly", "Quarterly", "Yearly"].index(st.session_state.global_cashflow["freq"]), key="g_freq")
        with gc4:
             st.markdown("<br>", unsafe_allow_html=True)
             st.session_state.global_cashflow["invest_div"] = st.checkbox("Re-invest Divs", st.session_state.global_cashflow["invest_div"], key="g_div")
        with gc5:
             st.markdown("<br>", unsafe_allow_html=True)
             st.session_state.global_cashflow["pay_down_margin"] = st.checkbox("Pay Down Margin", st.session_state.global_cashflow["pay_down_margin"], key="g_paydown")

        st.divider()

        # --- Portfolio Management Toolbar ---
        c_tool1, c_tool2, c_tool3 = st.columns([1, 2, 1])

        with c_tool1:
            if st.button("➕ Add Empty", use_container_width=True):
                if len(st.session_state.portfolios) < 5:
                    import uuid
                    new_id = f"p_{uuid.uuid4().hex[:8]}"
                    st.session_state.portfolios.append({
                        "id": new_id,
                        "name": f"Portfolio {len(st.session_state.portfolios) + 1}",
                        "alloc_df": pd.DataFrame([{"Ticker":"SPY", "Weight %": 100, "Maint %": 25}]),
                        "rebalance": {"mode": "Standard", "freq": "Yearly", "month": 1, "day": 1, "compare_std": False, "threshold_pct": 5.0}
                    })
                    st.session_state.active_tab_idx = len(st.session_state.portfolios) - 1
                    st.session_state.pop("portfolio_selector", None)
                    _pop_pf_keys()
                    st.rerun()
                else:
                    st.warning("Max 5 portfolios.")

        import json
        import os
        try:
            base_dir = os.path.dirname(os.path.abspath(__file__))
            preset_path = os.path.join(base_dir, "../../data/presets.json")
            preset_names = []
            presets = []
            if os.path.exists(preset_path):
                with open(preset_path, "r") as f:
                    presets = json.load(f)
                preset_names = [p["name"] for p in presets]

            with c_tool2:
                selected_preset = st.selectbox(
                    "Select Preset",
                    preset_names if preset_names else ["No Presets"],
                    key="preset_selector",
                    label_visibility="collapsed"
                )

            with c_tool3:
                if st.button("⬇️ Load Preset", use_container_width=True):
                    if preset_names and selected_preset and selected_preset != "No Presets":
                        if len(st.session_state.portfolios) < 5:
                            p_data = next(p for p in presets if p["name"] == selected_preset)
                            import uuid
                            new_id = f"p_pre_{uuid.uuid4().hex[:8]}"
                            reb = p_data.get("rebalance", {})
                            month_map = {"Jan":1, "Feb":2, "Mar":3, "Apr":4, "May":5, "Jun":6, "Jul":7, "Aug":8, "Sep":9, "Oct":10, "Nov":11, "Dec":12}
                            r_month = month_map.get(reb.get("month_str", "Jan"), 1)
                            st.session_state.portfolios.append({
                                "id": new_id,
                                "name": p_data["name"],
                                "alloc_df": pd.DataFrame(p_data["allocation"]),
                                "rebalance": {
                                    "mode": reb.get("mode", "Standard"),
                                    "freq": reb.get("freq", "Yearly"),
                                    "month": r_month,
                                    "day": reb.get("day", 1),
                                    "compare_std": False,
                                    "threshold_pct": reb.get("threshold_pct", 5.0),
                                }
                            })
                            st.session_state.active_tab_idx = len(st.session_state.portfolios) - 1
                            st.session_state.pop("portfolio_selector", None)
                            _pop_pf_keys()
                            st.rerun()
                        else:
                            st.warning("Max 5 portfolios.")
        except Exception as e:
            st.error(f"Error: {e}")

        st.divider()

        # --- Portfolio Selector (outside fragment — tab switch = full rerun) ---
        def _on_tab_change():
            # Save current portfolio's name before switching
            old_idx = st.session_state.active_tab_idx
            if "p_name" in st.session_state and old_idx < len(st.session_state.portfolios):
                st.session_state.portfolios[old_idx]["name"] = st.session_state["p_name"]
            # Resolve new tab
            sel = st.session_state.portfolio_selector
            if sel in display_names:
                st.session_state.active_tab_idx = display_names.index(sel)
            # Pop per-portfolio keys so widgets reinitialize from new portfolio data
            _pop_pf_keys()

        # Ensure selector is valid before rendering
        _sel = st.session_state.get("portfolio_selector")
        if _sel is None or _sel not in display_names:
            st.session_state.portfolio_selector = display_names[idx]

        st.segmented_control(
            "Portfolio",
            display_names,
            key="portfolio_selector",
            on_change=_on_tab_change,
        )

        # --- Per-portfolio content (inside fragment — stable keys, fast reruns) ---
        @st.fragment
        def _portfolio_content():
            idx = st.session_state.active_tab_idx
            p = st.session_state.portfolios[idx]

            with st.container():
                c_name, c_save, c_del = st.columns([6, 1, 1])
                with c_name:
                    p["name"] = st.text_input("Portfolio Name", p["name"], key="p_name", label_visibility="collapsed")

                with c_save:
                    if st.button("💾", key="p_save", help="Save as new Preset", use_container_width=True):
                        alloc_list = p["alloc_df"].to_dict("records")
                        preset_data = {
                            "name": p["name"],
                            "allocation": alloc_list,
                            "rebalance": {
                                "mode": p["rebalance"]["mode"],
                                "freq": p["rebalance"]["freq"],
                                "day": p["rebalance"]["day"]
                            }
                        }
                        month_map_inv = {1:"Jan", 2:"Feb", 3:"Mar", 4:"Apr", 5:"May", 6:"Jun", 7:"Jul", 8:"Aug", 9:"Sep", 10:"Oct", 11:"Nov", 12:"Dec"}
                        preset_data["rebalance"]["month_str"] = month_map_inv.get(p["rebalance"].get("month", 1), "Jan")
                        utils.save_preset(preset_data)
                        st.success(f"Saved!")
                        st.rerun(scope="app")

                with c_del:
                    if st.button("🗑️", key="p_del", help="Delete Portfolio", use_container_width=True):
                        if len(st.session_state.portfolios) > 1:
                            st.session_state.portfolios.pop(idx)
                            st.session_state.active_tab_idx = max(0, idx-1)
                            st.session_state.pop("portfolio_selector", None)
                            _pop_pf_keys()
                            st.rerun(scope="app")
                        else:
                            st.warning("Last portfolio")

            # --- Rebalancing Strategy ---
            with st.expander("📅 Rebalancing Strategy", expanded=False):
                mode_options = ["Standard", "Custom", "Threshold", "Threshold+Calendar"]
                try:
                    mode_idx = mode_options.index(p["rebalance"]["mode"])
                except (ValueError, KeyError):
                    mode_idx = 0
                r_mode = st.radio("Mode", mode_options, index=mode_idx, key="p_rmode", horizontal=True, label_visibility="collapsed")
                p["rebalance"]["mode"] = r_mode

                c_r1, c_r2, c_r3, c_r4 = st.columns(4)

                if r_mode == "Custom":
                    with c_r1:
                        freq_opts = ["Yearly", "Quarterly", "Monthly"]
                        try: f_idx = freq_opts.index(p["rebalance"]["freq"])
                        except (ValueError, KeyError): f_idx = 0
                        p["rebalance"]["freq"] = st.selectbox("Frequency", freq_opts, index=f_idx, key="p_rfreq")

                    with c_r2:
                        if p["rebalance"]["freq"] == "Yearly":
                            p["rebalance"]["month"] = st.selectbox("Rebalance Month", range(1, 13), index=p["rebalance"]["month"]-1, format_func=lambda x: pd.to_datetime(f"2024-{x}-1").strftime("%b"), key="p_rmon")
                        else:
                            p["rebalance"]["month"] = 1
                            st.markdown("")

                    with c_r3:
                        p["rebalance"]["day"] = st.number_input("Day of Month", 1, 31, p["rebalance"]["day"], key="p_rday")

                    with c_r4:
                        st.markdown("<br>", unsafe_allow_html=True)
                        p["rebalance"]["compare_std"] = st.checkbox("Compare vs Standard", p["rebalance"]["compare_std"], key="p_cmp")

                elif r_mode == "Threshold":
                    with c_r1:
                        p["rebalance"]["threshold_pct"] = st.number_input(
                            "Drift Threshold (%)", 1.0, 50.0,
                            float(p["rebalance"].get("threshold_pct", 5.0)),
                            step=1.0, key="p_rthresh",
                            help="Rebalance when any position drifts more than X pp from target. Checked daily."
                        )
                    p["rebalance"]["freq"] = "Yearly"
                    p["rebalance"]["month"] = 1
                    p["rebalance"]["day"] = 1

                elif r_mode == "Threshold+Calendar":
                    with c_r1:
                        freq_opts = ["Yearly", "Quarterly", "Monthly"]
                        try: f_idx = freq_opts.index(p["rebalance"]["freq"])
                        except (ValueError, KeyError): f_idx = 0
                        p["rebalance"]["freq"] = st.selectbox("Check Frequency", freq_opts, index=f_idx, key="p_rfreq_tc")
                    with c_r2:
                        p["rebalance"]["threshold_pct"] = st.number_input(
                            "Drift Threshold (%)", 1.0, 50.0,
                            float(p["rebalance"].get("threshold_pct", 5.0)),
                            step=1.0, key="p_rthresh_tc",
                            help="Only rebalance at scheduled check dates if drift exceeds threshold."
                        )
                    p["rebalance"]["month"] = 1
                    p["rebalance"]["day"] = 1

                else: # Standard Mode
                    with c_r1:
                        p["rebalance"]["freq"] = st.selectbox("Frequency", ["Yearly", "Quarterly", "Monthly"], index=["Yearly", "Quarterly", "Monthly"].index(p["rebalance"]["freq"]), key="p_rfreq_std")
                    p["rebalance"]["month"] = 1
                    p["rebalance"]["day"] = 1

            st.markdown("##### 🥧 Asset Allocation")
            new_alloc_df = st.data_editor(
                p["alloc_df"],
                key="p_editor",
                num_rows="dynamic",
                column_order=["Ticker", "Weight %", "Maint %"],
                column_config={
                    "Weight %": st.column_config.NumberColumn(min_value=0.0, max_value=100.0, step=0.01, format="%.2f"),
                    "Maint %": st.column_config.NumberColumn(min_value=0.0, max_value=100.0, step=0.1, format="%.1f"),
                },
                use_container_width=True
            )

            if not new_alloc_df.equals(p["alloc_df"]):
                p["alloc_df"] = new_alloc_df
                st.rerun(scope="fragment")

            # Validation & Metrics
            try:
                p_alloc_preview, p_maint_preview = api.table_to_dicts(p["alloc_df"])
                p_total_weight = sum(p_alloc_preview.values())
                d_maint = config.get('default_maint', 25.0)
                p_wmaint = sum(
                    (wt/100) * (p_maint_preview.get(t.split("?")[0], d_maint)/100)
                    for t, wt in p_alloc_preview.items()
                )
            except Exception:
                p_total_weight = 0.0
                p_wmaint = 0.0

            st.markdown("---")
            mc1, mc2 = st.columns(2)
            mc1.metric("Total Allocation", f"{p_total_weight:.2f}%", delta=None if abs(p_total_weight - 100) < 0.01 else "Must be 100%", delta_color="off" if abs(p_total_weight - 100) < 0.01 else "inverse")
            mc2.metric("Weighted Maint Req", f"{p_wmaint*100:.2f}%")

        _portfolio_content()
        
    with tab_port:
        _portfolio_fragment()
        config['portfolios'] = st.session_state.portfolios
        config['global_cashflow'] = st.session_state.get('global_cashflow', {
            "start_val": 10000.0, "amount": 0.0, "freq": "Monthly", "invest_div": True, "pay_down_margin": False
        })

    @st.fragment
    def _margin_fragment():
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
        
        with st.expander("Tax Configuration", expanded=True):
            c_tax1, c_tax2, c_tax3 = st.columns(3)
            with c_tax1:
                config['other_income'] = utils.num_input("Annual Income ($)", "other_income", 100000.0, 5000.0)
            with c_tax2:
                config['filing_status'] = st.selectbox("Filing Status", ["Single", "Married Filing Jointly", "Head of Household", "Married Filing Separately"], index=0, key="filing_status")
            with c_tax3:
                config['state_tax_rate'] = utils.num_input("State Tax Rate (%)", "state_tax_rate", 0.0, 0.1)

            st.caption("Federal tax brackets are automatically applied based on income & filing status.")
            
            tax_method_selection = st.radio(
                "Tax Calculation Method",
                ["Smart (Historical Brackets)", "Max (Top Rate)", "Fixed (2025 Rates)"],
                index=0,
                horizontal=True,
                help="Smart: Uses historical inclusion rates. Max: Flat historical max rate. Fixed: Modern 0/15/20% for all years."
            )
            
            if "Smart" in tax_method_selection:
                 config['tax_method'] = "historical_smart"
            elif "Max" in tax_method_selection:
                 config['tax_method'] = "historical_max_rate"
            else:
                 config['tax_method'] = "2025_fixed"
        
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
            curr_start_val = config.get('global_cashflow', {}).get('start_val', 10000.0)
            if curr_start_val != config['starting_loan']:
                current_lev = curr_start_val / (curr_start_val - config['starting_loan'])
            else:
                current_lev = 0.0
                
            st.caption(f"Current Leverage: **{current_lev:.2f}x**")
            
        with c2:
            st.markdown("##### Rates & Maintenance")

            
            margin_mode = st.selectbox("Margin Rate Model", ["Fixed", "Variable (Fed + Spread)", "Tiered (Blended)"], 
                                      index=2,
                                      help="**Fixed**: Constant annual rate.\n**Variable**: Fed Funds Rate (Daily) + User Spread.\n**Tiered**: Blended rate based on loan size (Base + Tiered Spread).",
                                      disabled=margin_disabled)
            
            margin_config = {}
            if margin_mode == "Fixed":
                margin_config = {
                    "type": "Fixed",
                    "rate_pct": utils.num_input("Annual Interest %", "rate_annual", 8.0, 0.5, disabled=margin_disabled)
                }
            elif margin_mode == "Variable (Fed + Spread)":
                from app.services import data_service
                fed_series = data_service.get_fed_funds_rate()
                
                # Show Preview
                curr_rate = fed_series.iloc[-1] if fed_series is not None and not fed_series.empty else 0.0
                st.caption(f"Current Base: **{curr_rate:.2f}%** (Fed Effective)")
                
                spread = utils.num_input("Spread over Base %", "spread_pct", 1.5, 0.1, disabled=margin_disabled)
                
                margin_config = {
                    "type": "Variable",
                    "base_series": fed_series,
                    "spread_pct": spread
                }
            else: # Tiered
                from app.services import data_service
                fed_series = data_service.get_fed_funds_rate()

                curr_rate = fed_series.iloc[-1] if fed_series is not None and not fed_series.empty else 0.0
                st.caption(f"Current Base: **{curr_rate:.2f}%**")
                
                st.markdown("**IBKR Pro Spreads (Blended)**")
                # IBKR Pro Tiers: 0-100k, 100k-1M, 1M-50M, >50M
                c_t1, c_t2 = st.columns(2)
                with c_t1:
                    t1_spread = st.number_input("Tier 1 (<100k) %", value=1.5, step=0.1, key="t1_spread")
                    t3_spread = st.number_input("Tier 3 (1M-50M) %", value=0.75, step=0.05, key="t3_spread")
                with c_t2:
                    t2_spread = st.number_input("Tier 2 (100k-1M) %", value=1.0, step=0.1, key="t2_spread")
                    t4_spread = st.number_input("Tier 4 (>50M) %", value=0.5, step=0.05, key="t4_spread")
                
                tiers = [
                    (0, t1_spread),
                    (100000, t2_spread),
                    (1000000, t3_spread),
                    (50000000, t4_spread)
                ]
                
                margin_config = {
                    "type": "Tiered",
                    "base_series": fed_series,
                    "tiers": tiers
                }
            
            config['rate_annual'] = margin_config
            
            config['draw_monthly'] = utils.num_input("Monthly Draw ($)", "draw_monthly", 0.0, 100.0)
            config['default_maint'] = utils.num_input("Default Maint %", "default_maint", 25.0, 1.0, disabled=margin_disabled)
            
            # Portfolio Margin Toggle
            config['pm_enabled'] = st.checkbox("Enable Portfolio Margin (PM)", value=False, help="Enforces $100k Minimum Equity requirement.", disabled=margin_disabled)

    with tab_margin:
        _margin_fragment()

    with tab_asset:
        # Asset Explorer is self-contained.
        # It doesn't modify the simulation config, just visualizes data.
        asset_explorer.render_asset_explorer()

    with tab_ndx:
        # NDX-100 Moving Average Scanner
        ndx_scanner.render_ndx_scanner()

    @st.fragment
    def _settings_fragment():
        c1, c2 = st.columns(2)
        with c1:
            config['chart_style'] = st.selectbox(
                "Chart Style",
                ["Classic (Combined)", "Candlestick"],
                index=0
            )
            config['timeframe'] = st.selectbox(
                "Chart Timeframe",
                ["1D", "1W", "1M", "3M", "1Y"],
                index=2
            )
            config['log_scale'] = st.checkbox("Logarithmic Scale", value=True)
        with c2:
            config['show_range_slider'] = st.checkbox("Show Range Slider", value=True)
            config['show_volume'] = st.checkbox("Show Range/Volume Panel", value=True)
            
            st.markdown("---")
            st.markdown("**Cache Management**")
            if st.button("🗑️ Clear All Caches", help="Purge all local data including API responses and downloaded ETF filings."):
                import shutil
                import os
                
                # 1. API Cache
                api_cache_dir = "data/api_cache"
                cleared = False
                if os.path.exists(api_cache_dir):
                    shutil.rmtree(api_cache_dir)
                    os.makedirs(api_cache_dir, exist_ok=True)
                    cleared = True
                    
                # 2. ETF Filings Cache
                etf_cache_dir = "data/etf_xray/cache/etf_filings"
                if os.path.exists(etf_cache_dir):
                    shutil.rmtree(etf_cache_dir)
                    os.makedirs(etf_cache_dir, exist_ok=True)
                    cleared = True
                
                # 3. Streamlit Data Cache (NDX Scanner, etc.)
                st.cache_data.clear()
                cleared = True
                
                if cleared:
                    st.success("All caches cleared!")
                else:
                    st.info("Caches are already empty.")

    with tab_settings:
        _settings_fragment()

    return config

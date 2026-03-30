import streamlit as st
import datetime as dt
import pandas as pd
from app.common import utils

def render():
    """Renders the sidebar and returns global settings."""
    
    # Initialize reload counter if needed (global state)
    if "reload_counter" not in st.session_state:
        st.session_state.reload_counter = 0

    with st.sidebar:
        st.title("📈 Testfol Charting")

        with st.expander("📋 Changelog (v3.6.0)", expanded=False):
            st.markdown("""
**v3.6.0** — Fresh Start Returns & Rebalance Timing Fix
- **Fresh Start** yearly column: per-year backtests for drift-free annual returns
- **Fresh Start toggle**: switch entire Returns Analysis to use fresh-start data
- Stitched fresh-start series for quarterly, monthly, daily & drawdown breakdowns
- **Rebalance timing fix**: Custom mode now correctly triggers on target date (was using end-of-period)
- Leveraged presets switched from Standard to Custom (Jan 1) rebalancing
- Single-ticker presets (QLD, QQUP) set to no rebalancing
- "None" rebalance mode added to UI

**v3.5.0** — Drawdowns Tab & Corrections Analysis
- New **Drawdowns** tab in Returns Analysis
- Corrections >5% with SPY comparison, severity filter
- 70+ market event labels (2000–2026)
- Sortable duration columns

**v3.4.0** — NDX Simulation Accuracy
- Official Nasdaq membership auditing
- Survivorship bias dampening
- Price cache improvements

**v3.3.0** — Multi-Provider Price Data
- Polygon.io → yfinance automatic failover
- Component performance chart
- ER-aware presets (NDXMEGASPLIT w/ ERs)

**v3.2.0** — Margin & Tax Overhaul
- Historical smart tax rates (2013–2023)
- Variable Fed Funds margin interest
- Draw start date & retirement income
- 164 regression tests

**v3.1.0** — Portfolio Margin & State Taxes
- Dynamic PM comparison & buy restrictions
- State tax library (all 50 states)
- Rolling metrics & risk charts

**v3.0.0** — Architecture Refactor
- Split into `app/` package structure
- FastAPI backend with REST endpoints
- Shadow backtest engine (FIFO tax lots)
- Disk-based HMAC cache

**v2.5.0** — Technical Analysis
- 200DMA, 150MA, Munger 200WMA
- Weinstein Stage Analysis
- NDX-100 MA scanner
- Trader's Cheat Sheet

**v2.0.0** — Returns & Monte Carlo
- Seasonal summary, heatmaps, distributions
- Annual/Quarterly/Monthly/Daily returns
- Monte Carlo simulation
- Benchmark comparisons

**v1.5.0** — Backtesting Engine
- Shadow yfinance backtester
- Custom rebalancing logic
- Tax calculations (federal + GLD)
- NDXMEGA simulated tickers

**v1.0.0** — Initial Release
- Margin stress testing
- Log scaling, chart metrics
- Testfol API integration
""")

        st.markdown("---")
        
        st.header("Global Settings")
        today = dt.date.today()

        # Check if dates were set programmatically (e.g., from scanner)
        default_start = st.session_state.get("_set_start_date", dt.date(2012, 1, 1))
        default_end = st.session_state.get("_set_end_date", today)

        start_date = st.date_input("Start Date", value=default_start, min_value=dt.date(1884, 1, 1), max_value=today)
        end_date = st.date_input("End Date", value=default_end, min_value=dt.date(1884, 1, 1), max_value=today)

        # Clear the programmatic date flags after use
        if "_set_start_date" in st.session_state:
            del st.session_state._set_start_date
        if "_set_end_date" in st.session_state:
            del st.session_state._set_end_date
        
        
        run_placeholder = st.empty()
        st.info("Configure your strategy, then click Run.")

        # --- Portfolio Switcher (quick access without scrolling) ---
        if "portfolios" in st.session_state and len(st.session_state.portfolios) > 1:
            st.markdown("---")
            portfolios = st.session_state.portfolios
            port_names = [p["name"] for p in portfolios]
            # Dedup display names
            display = list(port_names)
            _seen = {}
            for i, n in enumerate(display):
                c = _seen.get(n, 0) + 1
                _seen[n] = c
                if c > 1:
                    display[i] = f"{n} ({c})"

            active_idx = min(st.session_state.get("active_tab_idx", 0), len(display) - 1)

            def _on_sidebar_switch():
                sel = st.session_state.get("sidebar_portfolio")
                if sel in display:
                    new_idx = display.index(sel)
                    st.session_state.active_tab_idx = new_idx
                    st.session_state["portfolio_selector"] = display[new_idx]
                    # Pop stable keys + set p_name
                    for k in ["p_rmode", "p_rfreq", "p_rmon", "p_rday",
                               "p_cmp", "p_rthresh", "p_rfreq_tc", "p_rthresh_tc",
                               "p_rfreq_std", "p_editor"]:
                        st.session_state.pop(k, None)
                    st.session_state["p_name"] = portfolios[new_idx]["name"]

            st.session_state["sidebar_portfolio"] = display[active_idx]
            st.radio(
                "Portfolio",
                display,
                key="sidebar_portfolio",
                on_change=_on_sidebar_switch,
                label_visibility="collapsed",
            )

        import os
        from app.services.testfol_auth import get_token as _get_auth_token, login_with_credentials, get_user_email
        _has_env_auth = bool(os.environ.get("TESTFOL_EMAIL") or os.environ.get("TESTFOL_API_KEY"))
        _is_logged_in = bool(_get_auth_token())
        with st.expander("API Settings", expanded=_has_env_auth or _is_logged_in):
             # Testfol login
             if not _is_logged_in:
                 st.markdown("**Testfol Login**")
                 _tf_email = st.text_input("Email", key="_tf_email", label_visibility="collapsed", placeholder="Email")
                 _tf_pass = st.text_input("Password", type="password", key="_tf_pass", label_visibility="collapsed", placeholder="Password")
                 if st.button("Sign In", use_container_width=True):
                     if _tf_email and _tf_pass:
                         try:
                             login_with_credentials(_tf_email, _tf_pass)
                             st.rerun()
                         except Exception as e:
                             st.error(f"Login failed: {e}")
                     else:
                         st.warning("Enter email and password")
                 st.divider()

             bearer_token = st.text_input("Bearer Token (Override)", type="password", help="Manual token — overrides auto-login if set.")
             st.session_state._bearer_token = bearer_token
             _token = bearer_token or _get_auth_token()
             if _token:
                 _email = get_user_email()
                 st.caption(f"🟢 {_email}" if _email else "🟢 Authenticated (Pro API)")
             else:
                 st.caption("🔴 No token — limited to 10 tickers")
        
    return start_date, end_date, bearer_token, run_placeholder


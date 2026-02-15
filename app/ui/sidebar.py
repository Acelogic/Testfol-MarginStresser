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

        with st.expander("API Settings"):
             bearer_token = st.text_input("Bearer Token (Optional)", type="password", help="Overrides TESTFOL_API_KEY env var if set.")
             # Store in session state for use by other components (e.g., NDX Scanner)
             st.session_state._bearer_token = bearer_token
        
    return start_date, end_date, bearer_token, run_placeholder


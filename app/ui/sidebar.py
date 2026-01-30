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
        
        with st.expander("API Settings"):
             bearer_token = st.text_input("Bearer Token (Optional)", type="password", help="Overrides TESTFOL_API_KEY env var if set.")
             # Store in session state for use by other components (e.g., NDX Scanner)
             st.session_state._bearer_token = bearer_token
        
    return start_date, end_date, bearer_token, run_placeholder


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
        start_date = st.date_input("Start Date", value=dt.date(2012,1,1), min_value=dt.date(1884, 1, 1), max_value=today)
        end_date = st.date_input("End Date", value=today, min_value=dt.date(1884, 1, 1), max_value=today)
        
        
        run_placeholder = st.empty()
        st.info("Configure your strategy, then click Run.")
        
    return start_date, end_date, run_placeholder


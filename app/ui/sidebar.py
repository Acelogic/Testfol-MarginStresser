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
        start_date = st.date_input("Start Date", value=dt.date(2012,1,1), min_value=dt.date(1884, 1, 1))
        end_date = st.date_input("End Date", value=dt.date.today(), min_value=dt.date(1884, 1, 1))
        
        st.markdown("---")
        
        st.header("Saved Portfolios")
        saved_ports = utils.load_saved_portfolios()
        selected_port = st.selectbox("Select Portfolio", [""] + list(saved_ports.keys()))
        
        c_s1, c_s2, c_s3 = st.columns(3)
        
        if c_s1.button("Load"):
            if selected_port and selected_port in saved_ports:
                cfg = saved_ports[selected_port]
                st.session_state.alloc_df = pd.DataFrame(cfg.get("alloc", []))
                st.session_state.start_val = cfg.get("start_val", 10000.0)
                st.session_state.cashflow = cfg.get("cashflow", 0.0)
                st.session_state.starting_loan = cfg.get("starting_loan", 0.0)
                st.session_state.equity_init = cfg.get("equity_init", 100.0)
                st.session_state.rate_annual = cfg.get("rate_annual", 8.0)
                st.session_state.draw_monthly = cfg.get("draw_monthly", 0.0)
                st.session_state.default_maint = cfg.get("default_maint", 25.0)
                # Increment reload counter to force data editor refresh
                st.session_state.reload_counter += 1
                st.rerun()
                
        new_port_name = st.text_input("New Portfolio Name")
        if c_s2.button("Save"):
            if new_port_name:
                # Get the current edited data
                alloc_data = st.session_state.get("current_edited_df")
                if alloc_data is None or alloc_data.empty:
                    alloc_data = st.session_state.get("alloc_df")
                    
                if isinstance(alloc_data, pd.DataFrame) and not alloc_data.empty:
                    alloc_records = alloc_data.to_dict("records")
                else:
                    alloc_records = []

                current_config = {
                    "alloc": alloc_records,
                    "start_val": st.session_state.get("start_val", 10000.0),
                    "cashflow": st.session_state.get("cashflow", 0.0),
                    "starting_loan": st.session_state.get("starting_loan", 0.0),
                    "equity_init": st.session_state.get("equity_init", 100.0),
                    "rate_annual": st.session_state.get("rate_annual", 8.0),
                    "draw_monthly": st.session_state.get("draw_monthly", 0.0),
                    "default_maint": st.session_state.get("default_maint", 25.0)
                }
                utils.save_portfolio_to_disk(new_port_name, current_config)
                st.success(f"Saved {new_port_name}!")
                st.rerun()
                
        if c_s3.button("Delete"):
            if selected_port:
                utils.delete_portfolio_from_disk(selected_port)
                st.rerun()
        
        run_placeholder = st.empty()
        st.info("Configure your strategy, then click Run.")
        
    return start_date, end_date, run_placeholder


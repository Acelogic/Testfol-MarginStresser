"""Debug tab rendering."""
from __future__ import annotations

import json

import streamlit as st


def render_debug_tab(
    tab,
    logs: list,
    raw_response: dict,
    portfolio_name: str,
) -> None:
    with tab:
        st.markdown("### Shadow Backtest Logs")
        if logs:
            st.code("\n".join(logs), language="text")
        else:
            st.info("No logs available.")

        st.markdown("### Raw API Response")
        st.json(raw_response)

        st.subheader("Debug Info")

        st.divider()
        st.json(logs)
        st.write("Raw API Response (First 5 items):")
        st.write(str(raw_response)[:1000])

        json_str = json.dumps(raw_response, indent=2)
        st.download_button(
            label="Download Raw Response",
            data=json_str,
            file_name=f"testfol_api_response_{portfolio_name}.json",
            mime="application/json",
            key=f"dl_json_{portfolio_name}"
        )

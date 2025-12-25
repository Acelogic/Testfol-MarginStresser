"""
X-Ray View
UI components for displaying expanded portfolio holdings.
"""

import streamlit as st
import plotly.graph_objects as go
import pandas as pd
import sys
import os

# Ensure we can import from app/services
sys.path.append(os.path.join(os.getcwd(), "app/services"))
try:
    from xray_engine import compute_xray
except ImportError:
    sys.path.append(os.path.join(os.path.dirname(__file__), "../services"))
    from xray_engine import compute_xray

def render_xray(portfolio_dict, portfolio_name="Portfolio"):
    """
    portfolio_dict: {ticker: weight, ...}
    portfolio_name: Name of the portfolio being analyzed
    """
    if not portfolio_dict:
        st.info("Add assets to your portfolio to see the X-Ray view.")
        return

    st.subheader(f"🔍 Portfolio X-Ray: {portfolio_name}")
    from datetime import datetime
    year = datetime.now().year
    st.caption(f"Looking through ETFs to see your true underlying holdings (Holdings as of {year}).")

    with st.spinner("Calculating X-Ray expansion..."):
        df = compute_xray(portfolio_dict)

    if df.empty:
        st.warning("Could not expand holdings. Please check your internet connection or ticker symbols.")
        return

    # Show summary metrics
    unique_holdings = len(df)
    total_weight = df['Weight'].sum()
    positive_weight = df[df['Weight'] > 0]['Weight'].sum()
    negative_weight = df[df['Weight'] < 0]['Weight'].sum()
    
    col_m1, col_m2, col_m3 = st.columns(3)
    col_m1.metric("Unique Holdings", f"{unique_holdings:,}")
    col_m2.metric("Gross Exposure", f"{positive_weight:.1%}")
    if negative_weight < -0.001:
        col_m3.metric("Margin/Short", f"{negative_weight:.1%}")
    else:
        col_m3.metric("Net Exposure", f"{total_weight:.1%}")


    # Donut Chart
    col1, col2 = st.columns([1, 1])
    
    with col1:
        top_n = 50
        # Only show positive weights in the donut chart (Pie charts don't like negative values)
        df_plot = df[df['Weight'] > 0].head(top_n).copy()
        
        if len(df[df['Weight'] > 0]) > top_n:
            other_weight = df[df['Weight'] > 0].iloc[top_n:]['Weight'].sum()
            remaining_count = len(df[df['Weight'] > 0]) - top_n
            other_df = pd.DataFrame([{
                'Name': f'Other ({remaining_count} smaller holdings)',
                'Weight': other_weight,
                'Ticker': '',
                'Source': 'Multiple'
            }])
            df_plot = pd.concat([df_plot, other_df], ignore_index=True)

        fig = go.Figure(data=[go.Pie(
            labels=df_plot['Name'],
            values=df_plot['Weight'],
            hole=.6,
            textinfo='percent',
            hoverinfo='label+percent+value',
            marker=dict(colors=None)
        )])
        
        fig.update_layout(
            showlegend=False,
            margin=dict(t=0, b=0, l=0, r=0),
            height=350,
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.write("### Top Holdings")
        # Format weight as percentage
        df_display = df.copy()
        df_display['WeightPct'] = df_display['Weight'].apply(lambda x: f"{x:.2%}")
        
        # Display top 10 in a clean list
        for _, row in df.head(10).iterrows():
            st.write(f"**{row['Weight']:.2%}** {row['Name']} (`{row['Source']}`)")

    # Full Table
    with st.expander("Show Full Holdings Breakdown", expanded=False):
        st.dataframe(
            df[['Name', 'Ticker', 'Weight', 'Source']].style.format({
                'Weight': '{:.6%}'
            }),
            use_container_width=True,
            hide_index=True
        )

if __name__ == "__main__":
    # Mock testing
    render_xray({"QQQ": 0.5, "AAPL": 0.5}, "Test Portfolio")

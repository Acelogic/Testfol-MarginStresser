import plotly.graph_objects as go
import plotly.express as px
import streamlit as st
import pandas as pd
import numpy as np
import datetime as dt
from app.core import tax_library
import os
from app.common.utils import color_return
from plotly.subplots import make_subplots


from app.core import calculations

# --- Multi-Portfolio Chart ---
def render_multi_portfolio_chart(results_list, benchmarks=[], log_scale=True):
    """
    Renders a performance chart for multiple portfolios.
    Clips all series to common start date and rebases to $10k for fair comparison.
    """
    st.markdown("### Multi-Portfolio Performance Comparison")

    fig = go.Figure()

    # Colors for portfolios
    colors = ['#2E86C1', '#28B463', '#D35400', '#884EA0', '#F1C40F', '#1F618D', '#148F77', '#B03A2E']

    # Determine Common Start Date (latest start among all portfolios)
    start_dates = []
    for res in results_list:
        series = res.get('series')
        if series is not None and not series.empty:
            start_dates.append(series.index.min())
    if benchmarks:
        for bench in benchmarks:
            if bench is not None and not bench.empty:
                start_dates.append(bench.index.min())
    
    common_start = max(start_dates) if start_dates else None
    
    # Use the first portfolio's start_val from user's Global Capital config
    rebase_target = 10000.0
    if results_list:
        rebase_target = results_list[0].get('start_val', 10000.0)
    
    if common_start:
        st.caption(f"ℹ️ Chart aligned to common start date: **{common_start.date()}**. All values rebased to ${rebase_target:,.0f}.")

    # Add Portfolios (clipped to common start, rebased to $10k)
    for i, res in enumerate(results_list):
        name = res['name']
        series = res.get('series')
        stats = res.get('stats', {})
        
        if series is None or series.empty:
            continue
        
        # Clip to common start date
        if common_start:
            series = series[series.index >= common_start]
            if series.empty: continue
            
        color = colors[i % len(colors)]
        
        # Use stats directly - they're already accurate for the displayed period
        # For 'refetched' portfolios: stats came from fresh API call with common_start
        # For 'rebased' portfolios: stats were calculated locally from TWR (only option)
        # For non-rebased portfolios: stats are original API stats for that period
        cagr = stats.get('cagr', 0.0)
        max_dd = stats.get('max_drawdown', 0.0)
        
        label = f"{name} (CAGR: {cagr:.2f}%, DD: {max_dd:.2f}%)"
        
        # Rebase to $10k at common start for fair visual comparison
        original_values = series.values
        plot_values = series.values
        if not series.empty and series.iloc[0] != 0:
            plot_values = (series.values / series.iloc[0]) * rebase_target
        
        fig.add_trace(go.Scatter(
            x=series.index, 
            y=plot_values, 
            mode='lines', 
            name=label,
            line=dict(color=color, width=2),
            customdata=original_values,  # Store actual values for reference
            hovertemplate="<b>%{fullData.name}</b>: $%{y:,.0f}<extra></extra>"
        ))

    # Add Benchmarks (clipped to common start, rebased to $10k)
    if benchmarks:
        for i, bench in enumerate(benchmarks):
            if bench is None or bench.empty: continue
            
            # Clip to common start date
            if common_start:
                bench = bench[bench.index >= common_start]
                if bench.empty: continue
            
            b_name = bench.name if hasattr(bench, 'name') and bench.name else f"Benchmark {i+1}"
            b_stats = calculations.generate_stats(bench)
            b_cagr = b_stats.get('cagr', 0.0)
            b_mdd = b_stats.get('max_drawdown', 0.0)
            b_label = f"{b_name} (CAGR: {b_cagr:.2f}%, DD: {b_mdd:.2f}%)"
            
            # Rebase to $10k
            b_original = bench.values
            b_plot = bench.values
            if not bench.empty and bench.iloc[0] != 0:
                b_plot = (bench.values / bench.iloc[0]) * rebase_target
            
            fig.add_trace(go.Scatter(
                x=bench.index,
                y=b_plot,
                mode='lines',
                name=b_label,
                line=dict(color='#BDC3C7', width=1.5, dash='dash'),
                customdata=b_original,
                hovertemplate="<b>%{fullData.name}</b>: $%{y:,.0f}<extra></extra>"
            ))
            
    fig.update_layout(
        template="plotly_dark",
        xaxis_title="Date",
        xaxis_hoverformat="%b %d, %Y",
        yaxis_title="Portfolio Value ($)",
        yaxis_type="log" if log_scale else "linear",
        yaxis_tickprefix="$",
        yaxis_tickformat="s", # Uses SI prefixes (k, M, G)
        height=500,
        hovermode="x unified",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5
        ),
        margin=dict(l=40, r=40, t=60, b=40)
    )

    st.plotly_chart(fig, use_container_width=True)

    # --- Drawdown Chart ---
    st.markdown("### Drawdowns")
    fig_dd = go.Figure()

    # 1. Portfolios Drawdown (clipped to common start)
    for i, res in enumerate(results_list):
        series = res.get('series')
        if series is None or series.empty: continue
        
        # Clip to common start date
        if common_start:
            series = series[series.index >= common_start]
            if series.empty: continue
            
        # Drawdown Calc
        series = series.astype(float)
        running_max = series.cummax()
        drawdown = (series / running_max) - 1.0
        
        name = res['name']
        color = colors[i % len(colors)]
        
        fig_dd.add_trace(go.Scatter(
            x=drawdown.index, 
            y=drawdown, 
            mode='lines', 
            name=name,
            line=dict(color=color, width=1),
            fill='tozeroy', 
            hovertemplate="<b>%{fullData.name}</b>: %{y:.2%}<extra></extra>"
        ))

    # 2. Benchmarks Drawdown (clipped to common start)
    if benchmarks:
        for i, bench in enumerate(benchmarks):
            if bench is None or bench.empty: continue
            
            # Clip to common start date
            if common_start:
                bench = bench[bench.index >= common_start]
                if bench.empty: continue
            
            bench = bench.astype(float)
            running_max = bench.cummax()
            dd = (bench / running_max) - 1.0
            
            b_name = bench.name if hasattr(bench, 'name') and bench.name else f"Benchmark {i+1}"
            
            fig_dd.add_trace(go.Scatter(
                x=dd.index, 
                y=dd,
                mode='lines',
                name=b_name,
                line=dict(color='#BDC3C7', width=1.0, dash='dash'),
                hovertemplate="<b>%{fullData.name}</b>: %{y:.2%}<extra></extra>"
            ))
            
    fig_dd.update_layout(
        template="plotly_dark",
        xaxis_title="Date",
        xaxis_hoverformat="%b %d, %Y",
        yaxis_title="Drawdown (%)",
        yaxis_tickformat=".0%",
        height=350,
        hovermode="x unified",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5
        ),
        margin=dict(l=40, r=40, t=30, b=40)
    )
        
    st.plotly_chart(fig_dd, use_container_width=True)

    # Comparison Table (Stats are already accurate - refetched or calculated in main app)
    if results_list:
        st.markdown("### Statistics")
        
        stats_data = []
        for res in results_list:
            series = res.get('series')
            stats = res.get('stats', {})
            if series is None or series.empty:
                continue
            
            # Clip to common start (same as chart) for End Balance calculation
            if common_start:
                series = series[series.index >= common_start]
                if series.empty: continue
            
            # Use stats directly - they're already accurate for the displayed period
            cagr_raw = stats.get('cagr', 0)
            cagr_display = cagr_raw * 100 if abs(cagr_raw) <= 1 else cagr_raw
            
            # Handle volatility/std - API may return 'std' or 'volatility'
            vol_raw = stats.get('std', stats.get('volatility', 0))
            vol_display = vol_raw * 100 if abs(vol_raw) <= 1 else vol_raw
            
            row = {
                "Name": res['name'],
                "CAGR": f"{cagr_display:.2f}%",
                "Stdev": f"{vol_display:.2f}%",
                "Sharpe": f"{stats.get('sharpe', 0):.2f}",
                "Max DD": f"{stats.get('max_drawdown', 0):.2f}%",
                "End Balance": f"${series.iloc[-1]:,.2f}" if not series.empty else "$0"
            }
            stats_data.append(row)
            
        st.dataframe(pd.DataFrame(stats_data), use_container_width=True, hide_index=True)

@st.cache_data(show_spinner=False)
def render_classic_chart(port_series, final_adj_series, loan_series, 
                        equity_pct_series, usage_series, 
                        series_opts, log_scale,
                        bench_series=None, comparison_series=None, effective_rate_series=None):
    """
    Renders the classic line chart with toggleable traces.
    """
    fig = go.Figure()
    
    # 1. Total Portfolio Value
    if "Portfolio" in series_opts:
        fig.add_trace(go.Scatter(
            x=port_series.index, y=port_series,
            name="Portfolio Value (Gross)",
            line=dict(color='#2E86C1', width=2),
            hovertemplate="Portfolio: $%{y:,.0f}<extra></extra>"
        ))
        
    # 2. Net Equity
    if "Equity" in series_opts:
        # If paying taxes from cash, final_adj_series is Net Equity
        # If paying with margin, final_adj_series is also Net Equity (simulated)
        fig.add_trace(go.Scatter(
            x=final_adj_series.index, y=final_adj_series,
            name="Net Equity",
            line=dict(color='#28B463', width=2),
            fill='tozeroy',
            fillcolor='rgba(40, 180, 99, 0.1)',
            hovertemplate="Net Equity: $%{y:,.0f}<extra></extra>"
        ))

    # 3. Margin Loan
    if "Loan" in series_opts:
        fig.add_trace(go.Scatter(
            x=loan_series.index, y=loan_series,
            name="Margin Loan",
            line=dict(color='#E74C3C', width=1.5, dash='dot'),
            hovertemplate="Loan: $%{y:,.0f}<extra></extra>"
        ))
        
    # 4. Benchmarks
    if bench_series is not None and not bench_series.empty:
        fig.add_trace(go.Scatter(
            x=bench_series.index, y=bench_series,
            name=bench_series.name or "Benchmark",
            line=dict(color='#F1C40F', width=1.5, dash='dash'),
            hovertemplate="%{link_text}: $%{y:,.0f}<extra></extra>".replace("%{link_text}", bench_series.name or "Benchmark")
        ))
        
    if comparison_series is not None and not comparison_series.empty:
        fig.add_trace(go.Scatter(
            x=comparison_series.index, y=comparison_series,
            name=comparison_series.name or "Comparison",
            line=dict(color='#9B59B6', width=1.5, dash='dash'),
            hovertemplate="%{link_text}: $%{y:,.0f}<extra></extra>".replace("%{link_text}", comparison_series.name or "Comparison")
        ))
        
    # Secondary Axis for Percentages
    if "Margin usage %" in series_opts or "Equity %" in series_opts:
        fig.update_layout(yaxis2=dict(
            title="Percentage (%)",
            overlaying="y",
            side="right",
            range=[0, max(usage_series.max()*1.2 if not usage_series.empty else 1, 1.5)] # Some headroom
        ))

    if "Margin usage %" in series_opts:
        fig.add_trace(go.Scatter(
            x=usage_series.index, y=usage_series,
            name="Margin Usage %",
            line=dict(color='#FFD700', width=1),
            yaxis="y2",
            hovertemplate="Usage: %{y:.1%}<extra></extra>"
        ))
        
    if "Equity %" in series_opts:
        fig.add_trace(go.Scatter(
            x=equity_pct_series.index, y=equity_pct_series,
            name="Equity %",
            line=dict(color='#148F77', width=1),
            yaxis="y2",
            hovertemplate="Equity %: %{y:.1%}<extra></extra>"
        ))

    # Margin Call Threshold Line (100% Usage)
    if "Margin usage %" in series_opts:
        fig.add_trace(go.Scatter(
            x=[port_series.index[0], port_series.index[-1]], 
            y=[1.0, 1.0],
            name="Margin Call Threshold",
            line=dict(color='#FF0000', width=1.5, dash='dash'), # Red dashed line
            yaxis="y2",
            mode="lines",
            hoverinfo="skip" # Don't clutter hover
        ))
        
    if effective_rate_series is not None and not effective_rate_series.empty:
        # Check if rate is in % (e.g. 5.0) or decimal (0.05). API returns %.
        # Convert to decimal for consistent Y-axis with other percentages
        rate_decimal = effective_rate_series / 100.0
        fig.add_trace(go.Scatter(
            x=effective_rate_series.index, y=rate_decimal,
            name="Margin Rate %",
            line=dict(color='#FF00FF', width=1, dash='dot'), # Magenta
            yaxis="y2",

            hovertemplate="Rate: %{y:.2%}<extra></extra>" 
        ))

    fig.update_layout(
        template="plotly_dark",
        title="Portfolio History",
        xaxis_title="Date",
        xaxis_hoverformat="%b %d, %Y",
        yaxis_title="Value ($)",
        yaxis_type="log" if log_scale else "linear",
        height=600,
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    
    st.plotly_chart(fig, use_container_width=True)

@st.cache_data(show_spinner=False)
def render_dashboard_view(port, equity, loan, equity_pct, usage_pct, maint_pct, stats, log_opts, bench_series=None, comparison_series=None, start_val=10000, rate_annual=8.0):
    """Render dashboard-style separate charts"""
    
    # Get log scale settings
    log_portfolio = log_opts.get("portfolio", False)
    log_leverage = log_opts.get("leverage", False)
    log_margin = log_opts.get("margin", False)
    
    # Configure dark theme
    dark_theme = {
        "template": "plotly_dark",
        "paper_bgcolor": "rgba(30,35,45,1)",
        "plot_bgcolor": "rgba(30,35,45,1)",
        "font": {"color": "#E0E0E0"},
        "xaxis": {
            "gridcolor": "rgba(255,255,255,0.1)",
            "zerolinecolor": "rgba(255,255,255,0.2)"
        },
        "yaxis": {
            "gridcolor": "rgba(255,255,255,0.1)",
            "zerolinecolor": "rgba(255,255,255,0.2)"
        }
    }
    
    # Row 1: Portfolio Value Chart
    st.markdown("### Portfolio Value Over Time")
    fig1 = go.Figure()
    
    # Calculate leveraged portfolio (simulated with margin)
    leveraged_mult = 1 / (start_val / 100) if start_val < 100 else 1
    leveraged_port = port * leveraged_mult
    
    fig1.add_trace(go.Scatter(
        x=port.index, y=leveraged_port,
        name=f"Margin Portfolio ({leveraged_mult:.1f}x Leveraged)",
        line=dict(color="#4A90E2", width=2),
        hovertemplate="Leveraged: $%{y:,.0f}<extra></extra>"
    ))
    fig1.add_trace(go.Scatter(
        x=port.index, y=port,
        name="Margin Portfolio (Unleveraged)",
        line=dict(color="#1DB954", width=2),
        hovertemplate="Unleveraged: $%{y:,.0f}<extra></extra>"
    ))

    # Add Benchmark Trace if available (Moved to end)
    if bench_series is not None:
         bench_name = bench_series.name if hasattr(bench_series, 'name') and bench_series.name else "Benchmark (Gross)"
         fig1.add_trace(go.Scatter(
             x=bench_series.index, y=bench_series,
             name=bench_name,
             line=dict(color="#FFD700", width=2, dash="dash"),
             hovertemplate="%{link_text}: $%{y:,.0f}<extra></extra>".replace("%{link_text}", bench_name)
         ))
         
    # Add Comparison Trace if available
    if comparison_series is not None:
         comp_name = comparison_series.name if hasattr(comparison_series, 'name') and comparison_series.name else "Standard Rebalance"
         fig1.add_trace(go.Scatter(
             x=comparison_series.index, y=comparison_series,
             name=comp_name,
             line=dict(color="#00FFFF", width=2, dash="dot"),
             hovertemplate="%{link_text}: $%{y:,.0f}<extra></extra>".replace("%{link_text}", comp_name)
         ))
    
    fig1.update_layout(
        **dark_theme,
        height=400,
        xaxis_title="Month",
        xaxis_hoverformat="%b %d, %Y",
        yaxis_title="Portfolio Value ($)",
        yaxis_type="log" if log_portfolio else "linear",
        legend=dict(x=0.5, y=1.02, xanchor="center", orientation="h"),
        hovermode="x unified"
    )
    st.plotly_chart(fig1, use_container_width=True)
    
    # Row 2: Two columns for Leverage and Margin Debt
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Leverage Over Time")
        fig2 = go.Figure()
        
        # Calculate leverage metrics
        current_leverage = port / equity
        target_leverage = 1 / (start_val / 100) if start_val < 100 else 1
        max_allowed = 1 / (1 - maint_pct)
        
        fig2.add_trace(go.Scatter(
            x=port.index, y=current_leverage,
            name="Current Leverage",
            line=dict(color="#1DB954", width=2),
            hovertemplate="Current: %{y:.2f}x<extra></extra>"
        ))
        fig2.add_trace(go.Scatter(
            x=port.index, y=[target_leverage]*len(port),
            name="Target Leverage",
            line=dict(color="#FFD700", width=2, dash="dot"),
            hovertemplate="Target: %{y:.2f}x<extra></extra>"
        ))
        fig2.add_trace(go.Scatter(
            x=port.index, y=[max_allowed]*len(port),
            name="Max Allowed",
            line=dict(color="#FF6B6B", width=2, dash="dash"),
            hovertemplate="Max: %{y:.2f}x<extra></extra>"
        ))
        
        fig2.update_layout(
            **dark_theme,
            height=350,
            xaxis_title="Date",
            xaxis_hoverformat="%b %d, %Y",
            yaxis_title="Leverage Ratio",
            yaxis_type="log" if log_leverage else "linear",
            legend=dict(x=0.5, y=1.02, xanchor="center", orientation="h"),
            hovermode="x unified"
        )
        st.plotly_chart(fig2, use_container_width=True)
    
    with col2:
        st.markdown("### Margin Debt Evolution")
        fig3 = go.Figure()
        
        # Add area chart for margin debt
        fig3.add_trace(go.Scatter(
            x=loan.index, y=loan,
            name="Margin Debt",
            fill='tozeroy',
            line=dict(color="#FF6B6B", width=2),
            fillcolor="rgba(255,107,107,0.3)",
            hovertemplate="Debt: $%{y:,.0f}<extra></extra>"
        ))
        
        # Portfolio value line
        fig3.add_trace(go.Scatter(
            x=port.index, y=port,
            name="Portfolio Value",
            line=dict(color="#4A90E2", width=2),
            yaxis="y",
            hovertemplate="Portfolio: $%{y:,.0f}<extra></extra>"
        ))
        
        # Net liquidating value
        fig3.add_trace(go.Scatter(
            x=equity.index, y=equity,
            name="Net Liquidating Value",
            line=dict(color="#1DB954", width=2),
            yaxis="y",
            hovertemplate="Net Liq: $%{y:,.0f}<extra></extra>"
        ))
        
        # Monthly interest on secondary axis
        monthly_interest = loan * (rate_annual / 100 / 12)
        fig3.add_trace(go.Scatter(
            x=loan.index, y=monthly_interest,
            name="Monthly Interest",
            line=dict(color="#FFD700", width=1),
            yaxis="y2"
        ))
        
        fig3.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(30,35,45,1)",
            plot_bgcolor="rgba(30,35,45,1)",
            font={"color": "#E0E0E0"},
            height=350,
            xaxis=dict(
                title="Date",
                hoverformat="%b %d, %Y",
                gridcolor="rgba(255,255,255,0.1)",
                zerolinecolor="rgba(255,255,255,0.2)"
            ),
            yaxis=dict(
                title="Value ($)", 
                side="left",
                type="log" if log_margin else "linear",
                gridcolor="rgba(255,255,255,0.1)",
                zerolinecolor="rgba(255,255,255,0.2)"
            ),
            yaxis2=dict(
                title="Monthly Interest ($)", 
                overlaying="y", 
                side="right",
                type="log" if log_margin else "linear",
                gridcolor="rgba(255,255,255,0.1)",
                zerolinecolor="rgba(255,255,255,0.2)"
            ),
            legend=dict(x=0.5, y=1.02, xanchor="center", orientation="h"),
            hovermode="x unified"
        )
        st.plotly_chart(fig3, use_container_width=True)
    
    # Row 3: Final Margin Status - Enhanced display
    st.markdown("### Final Margin Status")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # Margin Utilization gauge
        fig4 = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=usage_pct.iloc[-1] * 100,
            title={'text': "Margin Utilization"},
            domain={'x': [0, 1], 'y': [0, 1]},
            gauge={
                'axis': {'range': [None, 100], 'tickwidth': 1},
                'bar': {'color': "#FFD700"},
                'steps': [
                    {'range': [0, 50], 'color': "rgba(0,255,0,0.3)"},
                    {'range': [50, 80], 'color': "rgba(255,255,0,0.3)"},
                    {'range': [80, 100], 'color': "rgba(255,0,0,0.3)"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 100
                }
            },
            number={'suffix': "%", 'font': {'size': 40}},
            delta={'reference': 50, 'increasing': {'color': "red"}}
        ))
        fig4.update_layout(
            **dark_theme,
            height=300,
            margin=dict(l=20, r=20, t=40, b=20)
        )
        st.plotly_chart(fig4, use_container_width=True)
        st.markdown(f"<p style='text-align: center; color: #888;'>{'Moderate Risk' if usage_pct.iloc[-1] < 0.8 else 'High Risk'}</p>", unsafe_allow_html=True)
    
    with col2:
        # Leverage gauge
        final_leverage = (port.iloc[-1] / equity.iloc[-1])
        fig5 = go.Figure(go.Indicator(
            mode="gauge+number",
            value=final_leverage,
            title={'text': "Leverage"},
            domain={'x': [0, 1], 'y': [0, 1]},
            gauge={
                'axis': {'range': [1, max_allowed], 'tickwidth': 1},
                'bar': {'color': "#4A90E2"},
                'steps': [
                    {'range': [1, 2], 'color': "rgba(0,255,0,0.3)"},
                    {'range': [2, 3], 'color': "rgba(255,255,0,0.3)"},
                    {'range': [3, max_allowed], 'color': "rgba(255,0,0,0.3)"}
                ],
            },
            number={'suffix': "x", 'font': {'size': 40}}
        ))
        fig5.update_layout(
            **dark_theme,
            height=300,
            margin=dict(l=20, r=20, t=40, b=20)
        )
        st.plotly_chart(fig5, use_container_width=True)
        st.markdown(f"<p style='text-align: center; color: #888;'>Peak {max_allowed:.2f}x</p>", unsafe_allow_html=True)
    
    with col3:
        # Available Margin display
        available_margin = port.iloc[-1] * (1 - maint_pct) - loan.iloc[-1]
        st.markdown(
            f"""
            <div style='background-color: rgba(30,35,45,1); padding: 20px; border-radius: 10px; text-align: center;'>
                <h4 style='color: #888; margin-bottom: 10px;'>Available Margin</h4>
                <h1 style='color: #1DB954; margin: 10px 0;'>${available_margin:,.2f}</h1>
                <p style='color: #888; margin-top: 10px;'>Additional Borrowing Power</p>
            </div>
            """,
            unsafe_allow_html=True
        )

@st.cache_data(show_spinner=False)
def render_candlestick_chart(ohlc_df, equity_series, loan_series, usage_series, equity_pct_series, timeframe, log_scale, show_range_slider=True, show_volume=True, bench_series=None, comparison_series=None):
    title_map = {
        "1D": "Daily",
        "1W": "Weekly",
        "1M": "Monthly",
        "3M": "Quarterly",
        "1Y": "Yearly"
    }
    
    # Calculate indicators
    # Simple Moving Averages
    sma_20 = ohlc_df['Close'].rolling(window=min(20, len(ohlc_df))).mean()
    sma_50 = ohlc_df['Close'].rolling(window=min(50, len(ohlc_df))).mean()
    sma_200 = ohlc_df['Close'].rolling(window=min(200, len(ohlc_df))).mean()
    
    # Volume proxy (range of price movement)
    volume_proxy = (ohlc_df['High'] - ohlc_df['Low']).fillna(0)
    

    if show_volume:
        fig = make_subplots(
            rows=2, cols=1,
            row_heights=[0.7, 0.3],
            vertical_spacing=0.03,
            subplot_titles=('', 'Range (Volatility)'),
            shared_xaxes=True
        )
        main_row = 1
    else:
        fig = make_subplots(rows=1, cols=1)
        main_row = 1
    
    # Prepare Hover Text with % change
    hover_text = []
    for i, (d, o, h, l, c) in enumerate(zip(ohlc_df.index, ohlc_df["Open"], ohlc_df["High"], ohlc_df["Low"], ohlc_df["Close"])):
        # Calculate period-over-period return (matching the returns table)
        if i > 0:
            prev_close = ohlc_df["Close"].iloc[i-1]
            pct_change = ((c - prev_close) / prev_close * 100) if prev_close != 0 else 0
        else:
            # For first period, use open to close
            pct_change = ((c - o) / o * 100) if o != 0 else 0
        
        change_sign = "+" if pct_change >= 0 else ""
        hover_text.append(
            f"Date: {d:%b %d, %Y}<br>"
            f"O: {o:,.2f}<br>"
            f"H: {h:,.2f}<br>"
            f"L: {l:,.2f}<br>"
            f"C: {c:,.2f}<br>"
            f"Change: {change_sign}{pct_change:.2f}%"
        )

    # Main candlestick chart (TradingView colors)
    fig.add_trace(go.Candlestick(
        x=ohlc_df.index,
        open=ohlc_df["Open"],
        high=ohlc_df["High"],
        low=ohlc_df["Low"],
        close=ohlc_df["Close"],
        name="Portfolio",
        text=hover_text,
        hoverinfo="text",
        increasing_line_color='#26a69a',  # TradingView green
        decreasing_line_color='#ef5350',  # TradingView red
        increasing_fillcolor='#26a69a',
        decreasing_fillcolor='#ef5350',
    ), row=1, col=1)

    # Add Benchmark Trace if available
    if bench_series is not None:
        bench_name = bench_series.name if hasattr(bench_series, 'name') and bench_series.name else "Benchmark (Gross)"
        fig.add_trace(go.Scatter(
            x=bench_series.index, y=bench_series,
            mode='lines',
            name=bench_name,
            line=dict(color='#FFD700', width=2, dash='dash'),
            hovertemplate="Bench: $%{y:,.0f}<extra></extra>"
        ), row=1, col=1)

    # Add Comparison Trace if available
    if comparison_series is not None:
        comp_name = comparison_series.name if hasattr(comparison_series, 'name') and comparison_series.name else "Standard Rebalance"
        fig.add_trace(go.Scatter(
            x=comparison_series.index, y=comparison_series,
            mode='lines',
            name=comp_name,
            line=dict(color='#00FFFF', width=2, dash='dot'), # Cyan Dot
            hovertemplate="Comp: $%{y:,.0f}<extra></extra>"
        ), row=1, col=1)

    # Add Moving Averages
    fig.add_trace(go.Scatter(
        x=ohlc_df.index,
        y=sma_20,
        name="SMA 20",
        line=dict(color="#2962FF", width=1.5),
        hovertemplate="SMA 20: $%{y:,.0f}<extra></extra>"
    ), row=1, col=1)
    
    fig.add_trace(go.Scatter(
        x=ohlc_df.index,
        y=sma_50,
        name="SMA 50",
        line=dict(color="#FF6D00", width=1.5),
        hovertemplate="SMA 50: $%{y:,.0f}<extra></extra>"
    ), row=1, col=1)
    
    fig.add_trace(go.Scatter(
        x=ohlc_df.index,
        y=sma_200,
        name="SMA 200",
        line=dict(color="#9C27B0", width=2),
        hovertemplate="SMA 200: $%{y:,.0f}<extra></extra>"
    ), row=1, col=1)

    # Equity Line
    fig.add_trace(go.Scatter(
        x=equity_series.index,
        y=equity_series,
        name="Equity",
        line=dict(color="#00E676", width=2, dash="dot"),
        hovertemplate="Equity: $%{y:,.0f}<extra></extra>"
    ), row=1, col=1)

    # Loan Line
    fig.add_trace(go.Scatter(
        x=loan_series.index,
        y=loan_series,
        name="Loan",
        line=dict(color="#FF5252", width=1, dash="dot"),
        hovertemplate="Loan: $%{y:,.0f}<extra></extra>"
    ), row=1, col=1)

    # Add Benchmark Trace if available (Moved to end)
    if bench_series is not None:
        bench_name = bench_series.name if hasattr(bench_series, 'name') and bench_series.name else "Benchmark (Gross)"
        fig.add_trace(go.Scatter(
            x=bench_series.index, y=bench_series,
            mode='lines',
            name=bench_name,
            line=dict(color='#FFD700', width=2, dash='dash'),
            hovertemplate="Bench: $%{y:,.0f}<extra></extra>"
        ), row=1, col=1)

    # Add Comparison Trace if available
    if comparison_series is not None:
        comp_name = comparison_series.name if hasattr(comparison_series, 'name') and comparison_series.name else "Standard Rebalance"
        fig.add_trace(go.Scatter(
            x=comparison_series.index, y=comparison_series,
            mode='lines',
            name=comp_name,
            line=dict(color='#00FFFF', width=2, dash='dot'), # Cyan Dot
            hovertemplate="Comp: $%{y:,.0f}<extra></extra>"
        ), row=1, col=1)

    # Volume bars (using range as proxy) - only if enabled
    if show_volume:
        colors = ['#26a69a' if ohlc_df['Close'].iloc[i] >= ohlc_df['Open'].iloc[i] else '#ef5350' 
                  for i in range(len(ohlc_df))]
        
        fig.add_trace(go.Bar(
            x=ohlc_df.index,
            y=volume_proxy,
            name="Range",
            marker_color=colors,
            opacity=0.5,
            hovertemplate="Range: $%{y:,.0f}<extra></extra>"
        ), row=2, col=1)

    # TradingView-style layout
    fig.update_layout(
        title=dict(
            text=f"Portfolio Performance ({title_map.get(timeframe, timeframe)})",
            font=dict(size=20, color='#d1d4dc')
        ),
        template="plotly_dark",
        paper_bgcolor='#131722',  # TradingView dark background
        plot_bgcolor='#131722',
        font=dict(color='#d1d4dc', family='Trebuchet MS'),
        xaxis_rangeslider_visible=show_range_slider,
        xaxis_rangeslider_thickness=0.05,
        height=800,
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01,
            bgcolor='rgba(19, 23, 34, 0.8)',
            bordercolor='#2a2e39',
            borderwidth=1
        ),
        margin=dict(l=60, r=60, t=80, b=80),
        dragmode="zoom",
        hovermode="x unified"
    )
    
    # Set default visible range to last 90 days for daily charts (bigger candles)
    if timeframe == "1D" and len(ohlc_df) > 90:
        # Show last ~3 months by default for readable candles
        fig.update_xaxes(range=[ohlc_df.index[-90], ohlc_df.index[-1]], row=1, col=1)

    # Update axes styling (TradingView style)
    if show_volume:
        fig.update_xaxes(
            showgrid=True,
            gridwidth=1,
            gridcolor='#2a2e39',
            showline=True,
            linewidth=1,
            linecolor='#2a2e39',
            fixedrange=False,
            row=2, col=1
        )
    
    fig.update_yaxes(
        showgrid=True,
        gridwidth=1,
        gridcolor='#2a2e39',
        showline=True,
        linewidth=1,
        linecolor='#2a2e39',
        side='right',  # TradingView has y-axis on right
        type="log" if log_scale else "linear",
        fixedrange=False,  # Enable y-axis zooming
        row=1, col=1
    )
    
    if show_volume:
        fig.update_yaxes(
            showgrid=False,
            side='right',
            fixedrange=False,  # Enable y-axis zooming
            row=2, col=1
        )

    # Increase candlestick line width
    fig.update_traces(
        increasing_line_width=2,
        decreasing_line_width=2,
        selector=dict(type='candlestick')
    )
    
    # Enable better interactions
    config = {
        'displayModeBar': True,
        'displaylogo': False,
        'modeBarButtonsToAdd': ['drawline', 'drawopenpath', 'eraseshape'],
        'modeBarButtonsToRemove': ['lasso2d', 'select2d'],
        'scrollZoom': True
    }
    
    st.plotly_chart(fig, use_container_width=True, config=config)

    # TradingView-style OHLC Table
    with st.expander("📊 OHLC Table View", expanded=False):
        # Prepare table data
        table_data = ohlc_df.copy()
        table_data['Date'] = table_data.index.strftime('%Y-%m-%d')
        table_data['Change %'] = ((table_data['Close'] - table_data['Open']) / table_data['Open'] * 100).round(2)
        table_data['Range'] = table_data['High'] - table_data['Low']
        
        # Reorder columns
        display_df = table_data[['Date', 'Open', 'High', 'Low', 'Close', 'Change %', 'Range']].copy()
        display_df = display_df.sort_index(ascending=False)  # Most recent first
        
        # Format currency columns
        for col in ['Open', 'High', 'Low', 'Close', 'Range']:
            display_df[col] = display_df[col].apply(lambda x: f"${x:,.2f}")
        
        display_df['Change %'] = display_df['Change %'].apply(lambda x: f"{x:+.2f}%")
        
        st.dataframe(
            display_df,
            use_container_width=True,
            hide_index=True,
            height=400
        )

    fig2 = go.Figure()

    fig2.add_trace(go.Scatter(
        x=usage_series.index,
        y=usage_series * 100,
        name="Margin Usage %",
        line=dict(color="#FFD700", width=2),
        fill='tozeroy',
        fillcolor="rgba(255, 215, 0, 0.1)",
        hovertemplate="%{y:.2f}%<extra></extra>"
    ))

    fig2.add_trace(go.Scatter(
        x=equity_pct_series.index,
        y=equity_pct_series * 100,
        name="Equity %",
        line=dict(color="#1DB954", width=2, dash="dash"),
        hovertemplate="%{y:.2f}%<extra></extra>"
    ))

    fig2.add_hline(y=100, line_dash="dot", line_color="red", annotation_text="Margin Call (100%)")

    fig2.update_layout(
        title=dict(text="Margin Risk Metrics", font=dict(size=16)),
        yaxis_title="Percentage (%)",
        xaxis_title="Date",
        template="plotly_dark",
        height=300,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=50, r=50, t=60, b=50)
    )
    st.plotly_chart(fig2, use_container_width=True)
def render_ma_analysis_tab(port_series, portfolio_name, unique_id, window=200, show_stage_analysis=True):
    """
    Renders the Moving Average Analysis tab content.
    """
    st.subheader(f"{portfolio_name} {window}-Day Moving Average Analysis")
    
    # Controls
    c_ctrl1, c_ctrl2 = st.columns(2)
    # Ensure keys are unique per window AND per portfolio instance
    key_suffix = f"{unique_id}_{window}" if unique_id else f"{window}"
    
    merge_tol = c_ctrl1.slider(
        "Merge Events Tolerance (Days)", 
        min_value=0, max_value=30, value=14, step=1,
        key=f"ma_merge_{key_suffix}",
        help=f"**Merge Tolerance**: Ignores short recoveries. If the price recovers above {window}MA for fewer than X days before dropping again, it is considered a single continuous 'Under' event. Useful for filtering out fake breakouts."
    )
    min_days = c_ctrl2.slider(
        "Signal Filter (Min Days)", 
        min_value=0, max_value=90, value=14, step=1,
        key=f"ma_min_{key_suffix}",
        help=f"**Signal Filter**: Excludes short-lived drops below the {window}MA (noise). Events shorter than X days will be hidden from the analysis table and statistics."
    )

    # Calculate Stats (Reactive)
    # Use the generalized analyze_ma function
    dma_series, events_df = calculations.analyze_ma(port_series, window=window, tolerance_days=merge_tol)
    
    # Calculate Stage (New)
    stage_series, slope_series, _ = calculations.analyze_stage(port_series, ma_window=window)
    
    if dma_series is None or dma_series.dropna().empty: 
            st.info(f"Insufficient data to calculate {window}MA (need >{window} days).")
            return

    # Just in case events_df is empty but we have DMA
    filtered_events = pd.DataFrame()  # Initialize for chart use
    if events_df.empty:
        st.info(f"Price has never been below {window}MA in this period.")
    else:
        # Apply Min Days Filter for Display/Stats
        filtered_events = events_df[events_df["Duration (Days)"] >= min_days]
    
    # Chart
    fig = go.Figure()
    
    # Base Price (Blue)
    fig.add_trace(go.Scatter(
        x=port_series.index, y=port_series,
        name="Price",
        line=dict(color='#2E86C1', width=2),
        hovertemplate="Price: $%{y:,.0f}<extra></extra>"
    ))
    
    # Price Below MA (Red Overlay)
    price_below = port_series.copy()
    # Mask values where Price >= DMA (keep only Below)
    price_below[port_series >= dma_series] = None 
    
    fig.add_trace(go.Scatter(
        x=price_below.index, y=price_below,
        name=f"Below {window}MA",
        line=dict(color='#FFD700', width=2),
        hovertemplate="Price: $%{y:,.0f}<extra></extra>",
        showlegend=False # Cleaner legend
    ))

    fig.add_trace(go.Scatter(
        x=dma_series.index, y=dma_series,
        name=f"{window}MA",
        line=dict(color='#E74C3C', width=1.5),
        hovertemplate=f"{window}MA: $%{{y:,.0f}}<extra></extra>"
    ))
    
    # Add Peak markers - Show ALL from events_df, grey out non-filtered
    if not events_df.empty and "Peak Date" in events_df.columns:
        # Get filtered peak dates for comparison
        filtered_peak_dates = set(filtered_events["Peak Date"].dropna()) if not filtered_events.empty else set()
        
        # Get all peaks from raw events
        peak_data = events_df[["Peak Date", "Bottom to Peak (%)"]].dropna(subset=["Peak Date"])
        if not peak_data.empty:
            for _, row in peak_data.iterrows():
                d = row["Peak Date"]
                rally = row["Bottom to Peak (%)"]
                if d in port_series.index:
                    is_filtered = d in filtered_peak_dates
                    rally_val = rally if pd.notna(rally) else 0
                    filtered_label = "" if is_filtered else " (filtered)"
                    fig.add_trace(go.Scatter(
                        x=[d], 
                        y=[port_series.loc[d]],
                        mode='markers',
                        name="Peak" if is_filtered else "Peak (Filtered)",
                        legendgroup="peak" if is_filtered else "peak_filtered",
                        showlegend=False,
                        visible=True if is_filtered else 'legendonly',
                        marker=dict(
                            symbol='diamond',
                            size=10 if is_filtered else 7,
                            color='#00CC96' if is_filtered else 'rgba(100, 100, 100, 0.5)',
                            line=dict(width=1, color='white' if is_filtered else 'grey')
                        ),
                        hovertemplate=f"Peak: $%{{y:,.0f}} (+{rally_val:.1f}%)<br>%{{x|%b %d, %Y}}{filtered_label}<extra></extra>"
                    ))
        
        # Add legend entries with legendgroup for toggle
        fig.add_trace(go.Scatter(x=[None], y=[None], mode='markers', name="Peak",
            legendgroup="peak",
            marker=dict(symbol='diamond', size=10, color='#00CC96', line=dict(width=1, color='white'))))
        fig.add_trace(go.Scatter(x=[None], y=[None], mode='markers', name="Peak (Filtered)",
            legendgroup="peak_filtered", visible='legendonly',
            marker=dict(symbol='diamond', size=7, color='rgba(100, 100, 100, 0.5)', line=dict(width=1, color='grey'))))
    
    # Add Bottom markers - Show ALL from events_df, grey out non-filtered
    if not events_df.empty and "Bottom Date" in events_df.columns:
        # Get filtered bottom dates for comparison
        filtered_bottom_dates = set(filtered_events["Bottom Date"].dropna()) if not filtered_events.empty else set()
        
        # Get all bottoms from raw events
        bottom_data = events_df[["Bottom Date", "Max Depth (%)"]].dropna(subset=["Bottom Date"])
        if not bottom_data.empty:
            for _, row in bottom_data.iterrows():
                d = row["Bottom Date"]
                depth = row["Max Depth (%)"]
                if d in port_series.index:
                    is_filtered = d in filtered_bottom_dates
                    depth_val = depth if pd.notna(depth) else 0
                    filtered_label = "" if is_filtered else " (filtered)"
                    fig.add_trace(go.Scatter(
                        x=[d], 
                        y=[port_series.loc[d]],
                        mode='markers',
                        name="Bottom" if is_filtered else "Bottom (Filtered)",
                        legendgroup="bottom" if is_filtered else "bottom_filtered",
                        showlegend=False,
                        visible=True if is_filtered else 'legendonly',
                        marker=dict(
                            symbol='triangle-down',
                            size=10 if is_filtered else 7,
                            color='#EF553B' if is_filtered else 'rgba(100, 100, 100, 0.5)',
                            line=dict(width=1, color='white' if is_filtered else 'grey')
                        ),
                        hovertemplate=f"Bottom: $%{{y:,.0f}} ({depth_val:.1f}%)<br>%{{x|%b %d, %Y}}{filtered_label}<extra></extra>"
                    ))
        
        # Add legend entries with legendgroup for toggle
        fig.add_trace(go.Scatter(x=[None], y=[None], mode='markers', name="Bottom",
            legendgroup="bottom",
            marker=dict(symbol='triangle-down', size=10, color='#EF553B', line=dict(width=1, color='white'))))
        fig.add_trace(go.Scatter(x=[None], y=[None], mode='markers', name="Bottom (Filtered)",
            legendgroup="bottom_filtered", visible='legendonly',
            marker=dict(symbol='triangle-down', size=7, color='rgba(100, 100, 100, 0.5)', line=dict(width=1, color='grey'))))
    fig.update_layout(
        title=f"Price vs {window}MA",
        template="plotly_dark",
        height=500,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        xaxis_title="Date",
        xaxis=dict(range=[port_series.index[0], port_series.index[-1]]),  # Force full date range
        yaxis_title="Price ($)",
        yaxis_type="log",
        hovermode="x unified"
    )
    st.plotly_chart(fig, use_container_width=True, key=f"ma_chart_{key_suffix}")

    if not events_df.empty:
        # Summary Metrics (Based on FILTERED events)
        # Note: If filtered_events is empty, we handle that
        
        total_days = (port_series.index[-1] - port_series.index[0]).days
        if not filtered_events.empty: # Use filtered_events here to respect the variable (was hardcoded)
            days_under = filtered_events["Duration (Days)"].sum()
            pct_under = (days_under / total_days) * 100 if total_days > 0 else 0
            
            longest_event_idx = filtered_events["Duration (Days)"].idxmax()
            longest_event = filtered_events.loc[longest_event_idx]
            
            l_dur = longest_event['Duration (Days)']
            l_depth = longest_event['Max Depth (%)']
            
            # Calculate median Recovery Days (Start to Recovered)
            if "Recovery Days" in filtered_events.columns:
                recovered_events = filtered_events[filtered_events["Recovery Days"].notna()]
                if not recovered_events.empty:
                    median_recovery = recovered_events["Recovery Days"].median()
                else:
                    median_recovery = None
            else:
                median_recovery = None

            # Calculate Median Rally Days (Bottom -> Peak)
            if "Days Bottom to Peak" in filtered_events.columns:
                rally_events = filtered_events[filtered_events["Days Bottom to Peak"].notna()]
                if not rally_events.empty:
                    median_rally_days = rally_events["Days Bottom to Peak"].median()
                else:
                   median_rally_days = None
            else:
                median_rally_days = None

            # Calculate Median Rally % (Bottom -> Peak)
            if "Bottom to Peak (%)" in filtered_events.columns:
                rally_pct_events = filtered_events[filtered_events["Bottom to Peak (%)"].notna()]
                if not rally_pct_events.empty:
                    median_rally_pct = rally_pct_events["Bottom to Peak (%)"].median()
                else:
                    median_rally_pct = None
            else:
                median_rally_pct = None
            
            # Calculate median max depth and range
            median_depth = filtered_events["Max Depth (%)"].median()
            min_depth = filtered_events["Max Depth (%)"].min()  # Most negative = deepest
            max_depth = filtered_events["Max Depth (%)"].max()  # Least negative = shallowest
            total_breaches = len(filtered_events)

            # Calculate median Days to ATH (only for events that reached ATH)
            if "Days to ATH" in filtered_events.columns:
                ath_events = filtered_events[filtered_events["Days to ATH"].notna()]
                if not ath_events.empty:
                    median_days_to_ath = ath_events["Days to ATH"].median()
                else:
                    median_days_to_ath = None
            else:
                median_days_to_ath = None
        else:
            days_under = 0
            pct_under = 0
            l_dur = 0
            l_depth = 0
            median_recovery = None
            median_rally_days = None
            median_rally_pct = None
            median_depth = None
            min_depth = None
            max_depth = None
            total_breaches = 0
            median_days_to_ath = None
        
        # Check current status and calculate depth metrics
        last_price = port_series.iloc[-1]
        last_dma = dma_series.iloc[-1]
        current_depth = None
        current_depth_rank = None
        is_below = False
        status_text = ""
        status_delta = ""

        if events_df.empty:
            if last_price >= last_dma:
                status_text = "🟢 Above"
                status_delta = f"Never below {window}MA"
            else:
                status_text = "🔴 Below"
                status_delta = f"First breach of {window}MA"
                is_below = True
        elif events_df.iloc[-1]["Status"] == "Ongoing":
            last_event = events_df.iloc[-1]
            status_text = "🔴 Below"
            status_delta = f"{last_event['Duration (Days)']}d under {window}MA"
            is_below = True
            # Calculate current depth
            start_date = last_event["Start Date"]
            event_prices = port_series[start_date:]
            if not event_prices.empty:
                start_price_val = event_prices.iloc[0]
                min_price = event_prices.min()
                current_depth = ((min_price - start_price_val) / start_price_val) * 100
                # Calculate depth rank
                if total_breaches > 0 and min_depth is not None:
                    all_depths = filtered_events["Max Depth (%)"].dropna().tolist()
                    sorted_depths = sorted(all_depths)
                    current_depth_rank = 1
                    for d in sorted_depths:
                        if current_depth <= d:
                            break
                        current_depth_rank += 1
        elif last_price < last_dma:
            status_text = "🔴 Below"
            status_delta = f"Just crossed {window}MA"
            is_below = True
        else:
            last_event = events_df.iloc[-1]
            last_end = last_event["End Date"]
            if pd.notna(last_end):
                days_above = (port_series.index[-1] - last_end).days
                status_text = "🟢 Above"
                status_delta = f"{days_above}d over {window}MA"
            else:
                status_text = "🟢 Above"
                status_delta = f"{window}MA"

        # Calculate recovery stats from similar depths (for below MA state)
        recovery_rate_similar = None
        num_similar = 0
        num_recovered = 0
        med_recovery_similar = None
        max_recovery_similar = None

        if is_below and current_depth is not None and total_breaches >= 1:
            similar_or_deeper = filtered_events[filtered_events["Max Depth (%)"] <= current_depth]
            num_similar = len(similar_or_deeper)
            if num_similar > 0:
                recovered = similar_or_deeper[similar_or_deeper["Status"] == "Recovered"]
                num_recovered = len(recovered)
                recovery_rate_similar = (num_recovered / num_similar) * 100
                if num_recovered > 0 and "Recovery Days" in recovered.columns:
                    recovery_days = recovered["Recovery Days"].dropna()
                    if not recovery_days.empty:
                        med_recovery_similar = recovery_days.median()
                        max_recovery_similar = recovery_days.max()

        # Row 1: Current State (4 cols)
        r1c1, r1c2, r1c3, r1c4 = st.columns(4)
        r1c1.metric("Status", status_text, status_delta)

        if current_depth is not None:
            rank_text = f"Rank #{current_depth_rank}/{total_breaches}" if current_depth_rank else ""
            r1c2.metric("Current Depth", f"{current_depth:.1f}%", rank_text)
        elif median_depth is not None:
            depth_range = f"Range: {min_depth:.0f}% to {max_depth:.0f}%" if min_depth is not None else ""
            r1c2.metric("Med. Depth", f"{median_depth:.1f}%", depth_range)

        r1c3.metric("Time Under", f"{pct_under:.1f}%", f"{days_under}d total")
        r1c4.metric("Longest", f"{l_dur:.0f}d", f"Depth: {l_depth:.1f}%")

        # Row 2: Recovery Outlook (4 cols)
        r2c1, r2c2, r2c3, r2c4 = st.columns(4)

        if is_below and recovery_rate_similar is not None:
            # Show recovery outlook for current situation
            r2c1.metric("Recovery Rate", f"{recovery_rate_similar:.0f}%", f"{num_recovered}/{num_similar} similar")
            if med_recovery_similar is not None:
                r2c2.metric("Similar Recovery", f"{med_recovery_similar:.0f}d", f"Max: {max_recovery_similar:.0f}d")
            elif median_recovery is not None:
                r2c2.metric("Med. Recovery", f"{median_recovery:.0f}d", "Start→Even")
            if median_rally_pct is not None and median_rally_days is not None:
                r2c3.metric("Med. Rally", f"{median_rally_pct:.1f}%", f"{median_rally_days:.0f}d Bottom→Peak")
            elif median_rally_pct is not None:
                r2c3.metric("Rally Gain", f"{median_rally_pct:.1f}%", "Bottom→Peak")
            if median_days_to_ath is not None:
                r2c4.metric("To ATH", f"{median_days_to_ath:.0f}d", "Cross→New High")
        else:
            # Show general historical stats
            if median_recovery is not None:
                r2c1.metric("Med. Recovery", f"{median_recovery:.0f}d", "Start→Even")
            if median_rally_pct is not None and median_rally_days is not None:
                r2c2.metric("Med. Rally", f"{median_rally_pct:.1f}%", f"{median_rally_days:.0f}d Bottom→Peak")
            elif median_rally_days is not None:
                r2c2.metric("Med. Rally", f"{median_rally_days:.0f}d", "Bottom→Peak")
            if median_days_to_ath is not None:
                r2c3.metric("To ATH", f"{median_days_to_ath:.0f}d", "Cross→New High")
            r2c4.metric("# Breaches", f"{total_breaches}", "Historical events")

        # Stage Analysis Display
        if show_stage_analysis and stage_series is not None and not stage_series.empty:
            current_stage = stage_series.iloc[-1]
            current_slope = slope_series.iloc[-1]
            
            # Determine color/icon for Stage
            # Stage 1 (Basing) = cautiously bullish (coming out of decline)
            # Stage 3 (Topping) = cautiously bearish (coming off advance)
            if "Stage 2" in current_stage:
                s_color = "🟢"
            elif "Stage 4" in current_stage:
                s_color = "🔴"
            elif "Stage 1" in current_stage and "1/3" not in current_stage:
                s_color = "🟡"  # Basing - neutral but post-decline
            elif "Stage 3" in current_stage:
                s_color = "🟠"  # Topping - warning after advance
            else:
                s_color = "⚪"  # Indeterminate
            
            # Trend Text
            if current_slope > 0.001: trend_txt = "Rising ↗️"
            elif current_slope < -0.001: trend_txt = "Falling ↘️"
            else: trend_txt = "Flat ➡️"
            
            st.markdown("---")
            sc1, sc2 = st.columns(2)
            sc1.metric("Weinstein Stage Est.", f"{s_color} {current_stage}")
            sc2.metric(f"{window}MA Trend", trend_txt, f"Slope: {current_slope:.2%}")

            with st.expander("ℹ️ About Weinstein Market Stages"):
                st.markdown("""
                **Stan Weinstein's 4 Stages** *(from "Secrets for Profiting in Bull and Bear Markets")*:
                
                | Stage | Name | MA Trend | Price vs MA | Implication |
                |-------|------|----------|-------------|-------------|
                | 🟡 **1** | **Basing** | Flat (after falling) | Near/around | Accumulation. Bottoming process. |
                | 🟢 **2** | **Advancing** | Rising | Above | Bull market. Strong uptrend. |
                | 🟠 **3** | **Topping** | Flat (after rising) | Near/around | Distribution. Trend exhaustion. |
                | 🔴 **4** | **Declining** | Falling | Below | Bear market. Strong downtrend. |
                
                **Sub-Phases:**
                -   **Stage 2 (Correction):** Price dips below *rising* MA. Often a buying dip if trend intact.
                -   **Stage 4 (Bear Rally):** Price pops above *falling* MA. Often a "bull trap" or selling opportunity.
                
                **This Implementation:**
                -   Uses **{window}-day MA** (Weinstein used 30-week/~150-day on weekly charts).
                -   **Adaptive threshold**: "Flat" is relative to the asset's recent volatility.
                -   **Stage 1 vs 3**: Distinguished by *prior trend* (1 follows decline, 3 follows advance).
                -   **5-day smoothing**: Reduces daily noise/whipsaw.
                """.replace("{window}", str(window)))
        
        # Events Table
        st.subheader(f"Periods Under {window}MA")
        
        with st.expander("ℹ️ Understanding the Metrics"):
            st.markdown(f"""
**Row 1 - Current State:**
| Metric | Meaning |
|--------|---------|
| **Status** | Current position: 🟢 Above or 🔴 Below the {window}MA, with duration |
| **Current Depth** | *(When below MA)* Current drawdown from breach start, with rank among all historical breaches (1 = deepest) |
| **Med. Depth** | *(When above MA)* Typical (median) drawdown, with historical range |
| **Time Under** | Total % of the period spent below the MA, with total days |
| **Longest** | The single longest breach, with its max depth |

**Row 2 - Recovery Outlook** *(when below MA)*:
| Metric | Meaning |
|--------|---------|
| **Recovery Rate** | % of historical breaches at similar depth that recovered, with count |
| **Similar Recovery** | Median recovery time from breaches of similar or greater depth, with max |
| **Med. Rally** | Typical rally gain from bottom to peak, with duration |
| **To ATH** | Typical days from MA crossover to new all-time high |

**Row 2 - Historical Stats** *(when above MA)*:
| Metric | Meaning |
|--------|---------|
| **Med. Recovery** | Typical days from breach start to breakeven (price back to start) |
| **Med. Rally** | Typical rally gain from bottom to peak, with duration |
| **To ATH** | Typical days from MA crossover to new all-time high |
| **# Breaches** | Total historical breach events |

**Table Columns:**
| Column | Meaning |
|--------|---------|
| **Start / End** | When the price dropped below / recovered above the {window}MA |
| **Days Under MA** | Total calendar days spent below the MA |
| **Max Depth** | Price drawdown from event start to the lowest point (`(Bottom - Start) / Start`) |
| **Breakeven Gain** | % gain from the **bottom** to the **start price** (breakeven). Shows how much the price rallied just to get back to even. |
| **Post-MA Rally** | % gain from **recovery date** (MA crossover) to the subsequent peak |
| **Post-MA Rally Days** | Days from **recovery date** to the subsequent peak (Duration of post-recovery rally) |
| **Price Recovery Days** | Calendar days from **event start** to first date price recovers to **start price** (breakeven). For ongoing events, shows days elapsed so far. |
| **True Recovery** | Calendar days from **event start** until **BOTH** price ≥ start price **AND** price > MA. This is when you're truly recovered—at breakeven and back in an uptrend. |
| **Entry MA** | The MA value when price first dropped below it (your "entry" into the drawdown) |
| **Exit MA** | The MA value when price recovered above it (your "exit" from the drawdown) |
| **MA Δ%** | How much the MA changed during the event: `(Exit MA - Entry MA) / Entry MA`. Negative = MA fell (making it easier to cross back above even if price hasn't fully recovered) |
| **Rally Days** | Calendar days from **lowest price** to **subsequent peak** (Duration of the rally) |
| **Full Rally %** | % gain from the **lowest price** to the subsequent peak (full rebound) |
| **Days to ATH** | Days from **MA crossover** until price makes a **new all-time high** (vs pre-drawdown ATH) |
| **Status** | `Recovered` = crossed back above MA, `Ongoing` = still below (shown with 🟠 highlight) |
| **Pattern** | Recovery shape classification (see below) |

**What is "Peak"?**
> The **subsequent peak** is the highest price reached between the MA recovery date and either:
> - The start of the *next* drawdown event (next time price drops below MA), or
> - The end of the data (if no subsequent drawdown occurred).
>
> This represents the **local high** during the rally—not necessarily a new all-time high. 
> To see when the price made a *new ATH*, check the **Days to ATH** column.

**Recovery Patterns** *(classified using median thresholds from this dataset)*:

| Pattern | Criteria | What It Means |
|---------|----------|---------------|
| ⚡ **V-Shape** | Fast (≤ median days) + Strong (≥ median %) | **Best case.** Sharp selloff met with aggressive buying. Market quickly finds a floor and rockets higher. Often seen after panic selling or capitulation events. |
| 📈 **Grind** | Slow (> median days) + Strong (> median %) | **Patience rewarded.** Base-building recovery that eventually delivers strong returns. Requires holding through volatility but ends well. |
| 🐌 **Choppy** | Slow (> median days) + Weak (< median %) | **Frustrating.** Extended period of sideways action with minimal payoff. May indicate structural weakness or regime change. |
| 📉 **Weak** | Fast (≤ median days) + Weak (< median %) | **Dead cat bounce.** Quick but shallow recovery that doesn't recoup losses. Often followed by more downside. |

*Thresholds are relative to this asset's history—what's "fast" for bonds differs from stocks.*
            """)
        
        display_df = filtered_events.copy() # Use filtered_events here
        if not display_df.empty:
            # Formatting Dates
            display_df["Start"] = display_df["Start Date"].dt.date
            display_df["End"] = display_df["End Date"].apply(lambda x: x.date() if pd.notna(x) else "Ongoing")
            if "Peak Date" in display_df.columns:
                display_df["Peak Date"] = display_df["Peak Date"].apply(lambda x: x.date() if pd.notna(x) else "-")
            
            # Add Recovery Pattern to Status for recovered events
            # Calculate BEFORE renaming columns to access raw fields
            def classify_recovery(row):
                status = row.get("Status", "")
                if "Recovered" not in str(status):
                    return status
                
                days = row.get("Days Bottom to Peak") # Use Rally signal duration
                pct = row.get("Bottom to Peak (%)")
                
                if pd.isna(days) or pd.isna(pct):
                    return status
                
                # Use median thresholds for classification
                # Note: These medians should be calculated on the whole set
                median_days = filtered_events["Days Bottom to Peak"].median()
                median_pct = filtered_events["Bottom to Peak (%)"].median()
                
                short_days = days <= median_days
                high_pct = pct >= median_pct
                
                if short_days and high_pct:
                    pattern = "⚡ V-Shape"
                elif not short_days and high_pct:
                    pattern = "📈 Grind"
                elif not short_days and not high_pct:
                    pattern = "🐌 Choppy"
                else:  # short_days and low_pct
                    pattern = "📉 Weak"
                
                return pattern
            
            display_df["Pattern"] = display_df.apply(classify_recovery, axis=1)

            # Selection & Renaming for cleaner UI
            cols_map = {
                "Start": "Start",
                "End": "End",
                "Duration (Days)": "Days Under MA",
                "Max Depth (%)": "Max Depth",
                "Bottom to Recovery (%)": "Breakeven Gain",  # Actual rally from bottom to breakeven
                "Subsequent Peak (%)": "Post-MA Rally",
                "Post-MA Rally Days": "Post-MA Rally Days",
                "Bottom to Peak (%)": "Full Rally %",
                "Days Bottom to Peak": "Rally Days",
                "Recovery Days": "Price Recovery Days",
                "Days to ATH": "Days to ATH",
                "Status": "Status",
                "Pattern": "Pattern",
                # MA Context for accuracy (MA is a moving target)
                "Entry MA": "Entry MA",
                "Exit MA": "Exit MA",
                "MA Change (%)": "MA Δ%",
                "True Recovery Days": "True Recovery"
            }

            # Ensure columns exist before selecting
            final_cols = [c for c in cols_map.keys() if c in display_df.columns or c in ["Start", "End"]]

            display_df = display_df[final_cols].rename(columns=cols_map)
            display_df = display_df.sort_values("Start", ascending=False)

            # Add emoji indicators to Status for visual highlighting (since column_config doesn't support row styling)
            display_df["Status"] = display_df["Status"].apply(
                lambda x: f"🟡 {x}" if "Ongoing" in str(x) or "Current" in str(x) else x
            )

            # Toggle for MA context columns
            show_ma_context = st.checkbox(
                "Show MA Context",
                value=False,
                help="Show Entry MA, Exit MA, MA Δ%, and True Recovery columns. These help explain why 'Days Under MA' can differ from 'Price Recovery Days' since the MA moves during the drawdown.",
                key=f"ma_context_{unique_id}_{window}"
            )

            # Reorder columns - Main metrics first, then optionally MA context columns
            if show_ma_context:
                final_display_cols = [
                    "Start", "End", "Days Under MA", "Max Depth", "Breakeven Gain",
                    "Price Recovery Days", "True Recovery",  # True Recovery = Price + MA recovered
                    "Entry MA", "Exit MA", "MA Δ%",  # MA Context
                    "Full Rally %", "Rally Days", "Post-MA Rally", "Post-MA Rally Days",
                    "Days to ATH", "Status", "Pattern"
                ]
            else:
                final_display_cols = [
                    "Start", "End", "Days Under MA", "Max Depth", "Breakeven Gain",
                    "Price Recovery Days",
                    "Full Rally %", "Rally Days", "Post-MA Rally", "Post-MA Rally Days",
                    "Days to ATH", "Status", "Pattern"
                ]
            final_display_cols = [c for c in final_display_cols if c in display_df.columns]
            display_df = display_df[final_display_cols]

            # Column tooltips for hover explanations
            column_config = {
                "Start": st.column_config.DateColumn("Start", help="Date when price dropped below the MA", format="YYYY-MM-DD"),
                "End": st.column_config.DateColumn("End", help="Date when price recovered above the MA (or 'Ongoing')", format="YYYY-MM-DD"),
                "Days Under MA": st.column_config.NumberColumn("Days Under MA", help="Total calendar days spent below the MA", format="%.0f"),
                "Max Depth": st.column_config.NumberColumn("Max Depth", help="Price drawdown from event start to the lowest point", format="%.2f%%"),
                "Breakeven Gain": st.column_config.NumberColumn("Breakeven Gain", help="% gain needed from the bottom to get back to start price (breakeven)", format="%.1f%%"),
                "Price Recovery Days": st.column_config.NumberColumn("Price Recovery Days", help="Calendar days from event start to breakeven. For ongoing events, shows days elapsed so far.", format="%.0f"),
                "True Recovery": st.column_config.NumberColumn("True Recovery", help="Days until BOTH price ≥ start price AND price > MA (truly recovered: at breakeven + in uptrend)", format="%.0f"),
                "Entry MA": st.column_config.NumberColumn("Entry MA", help="MA value when price first dropped below it", format="$%.2f"),
                "Exit MA": st.column_config.NumberColumn("Exit MA", help="MA value when price recovered above it", format="$%.2f"),
                "MA Δ%": st.column_config.NumberColumn("MA Δ%", help="How much MA changed during event. Negative = MA fell, making crossover easier even if price hasn't fully recovered.", format="%+.1f%%"),
                "Full Rally %": st.column_config.NumberColumn("Full Rally %", help="% gain from the lowest price to the subsequent peak (or current price for ongoing)", format="%.1f%%"),
                "Rally Days": st.column_config.NumberColumn("Rally Days", help="Calendar days from lowest price to subsequent peak", format="%.0f"),
                "Post-MA Rally": st.column_config.NumberColumn("Post-MA Rally", help="% gain from MA crossover to the subsequent peak", format="%.1f%%"),
                "Post-MA Rally Days": st.column_config.NumberColumn("Post-MA Rally Days", help="Days from MA crossover to the subsequent peak", format="%.0f"),
                "Days to ATH": st.column_config.NumberColumn("Days to ATH", help="Days from MA crossover until price makes a new all-time high", format="%.0f"),
                "Status": st.column_config.TextColumn("Status", help="🟡 Ongoing = still below MA, Recovered = crossed back above MA"),
                "Pattern": st.column_config.TextColumn("Pattern", help="Recovery shape classification based on duration and rally strength"),
            }

            st.dataframe(
                display_df,
                column_config=column_config,
                use_container_width=True,
                hide_index=True
            )

            # Entry Strategy Comparison Table
            st.subheader(f"{portfolio_name} Entry Strategy Comparison (vs SPYSIM)")

            with st.expander("ℹ️ Understanding the Metrics"):
                st.markdown("""
**Strategy:** Buy at the **maximum depth** (lowest point) during each MA breach, sell when price recovers above the MA. Compare returns to buying SPY at the same time.

**Summary Metrics:**
| Metric | Meaning |
|--------|---------|
| **Total Events** | Number of completed (recovered) breach events analyzed |
| **Win Rate** | % of events where buying at max-depth beat buying SPY |
| **Avg Alpha** | Average outperformance vs SPY across all events |
| **Median Alpha** | Typical (median) outperformance vs SPY |

**Table Columns:**
| Column | Meaning |
|--------|---------|
| **Start / End** | When price dropped below / recovered above the MA |
| **Days/Weeks** | Duration of the breach event |
| **Max Depth** | Maximum drawdown from breach start price during this event |
| **Depth Rank** | Depth rank among all breaches (1 = deepest in history) |
| **Return** | Portfolio return from max-depth entry to recovery |
| **SPY Return** | SPYSIM return for the same window (max-depth to recovery) |
| **Alpha** | Outperformance vs SPY (Return - SPY Return). Green = beat SPY, Red = underperformed |
                """)

            comparison_df = calculations.compare_breach_events(
                port_series,
                window=window,
                tolerance_days=merge_tol
            )

            # Also get ongoing event if exists (from the full events_df)
            _, all_events_df = calculations.analyze_ma(port_series, window=window, tolerance_days=merge_tol)
            ongoing_event = None
            if not all_events_df.empty:
                last_event = all_events_df.iloc[-1]
                if last_event["Status"] == "Ongoing":
                    ongoing_event = last_event

            # Summary Statistics Row (only from recovered events)
            if not comparison_df.empty or ongoing_event is not None:
                total_recovered = len(comparison_df) if not comparison_df.empty else 0

                # Extract alpha column with NA handling
                maxdepth_alpha = comparison_df["Max-Depth Entry Alpha (%)"].dropna() if not comparison_df.empty else pd.Series(dtype=float)

                # Calculate statistics
                maxdepth_wins = (maxdepth_alpha > 0).sum() if len(maxdepth_alpha) > 0 else 0
                maxdepth_win_rate = (maxdepth_wins / len(maxdepth_alpha) * 100) if len(maxdepth_alpha) > 0 else 0
                maxdepth_avg = maxdepth_alpha.mean() if len(maxdepth_alpha) > 0 else 0
                maxdepth_median = maxdepth_alpha.median() if len(maxdepth_alpha) > 0 else 0

                # Display metrics row
                m1, m2, m3, m4, m5 = st.columns(5)
                m1.metric("Total Events", total_recovered + (1 if ongoing_event is not None else 0))
                m2.metric("Win Rate", f"{maxdepth_win_rate:.1f}%", help="% of recovered events where max-depth entry beat SPY")
                m3.metric("Avg Alpha", f"{maxdepth_avg:+.1f}%", help="Average outperformance vs SPY (recovered events)")
                m4.metric("Median Alpha", f"{maxdepth_median:+.1f}%", help="Median outperformance vs SPY (recovered events)")
                m5.metric("Events Analyzed", f"{len(maxdepth_alpha)}/{total_recovered}" + (" +1 ongoing" if ongoing_event is not None else ""))

                st.markdown("---")

                # Prepare display DataFrame
                comp_display = comparison_df.copy() if not comparison_df.empty else pd.DataFrame()

                # Add ongoing event to display if exists
                if ongoing_event is not None:
                    ongoing_row = {
                        "Start Date": ongoing_event["Start Date"],
                        "End Date": pd.NaT,
                        "Duration (Days)": ongoing_event["Duration (Days)"],
                        "Max Depth (%)": ongoing_event["Max Depth (%)"],
                        "Max-Depth Entry Return (%)": None,
                        "SPYSIM Max-Depth Return (%)": None,
                        "Max-Depth Entry Alpha (%)": None,
                        "Status": "🟠 Ongoing"
                    }
                    ongoing_df = pd.DataFrame([ongoing_row])
                    comp_display = pd.concat([ongoing_df, comp_display], ignore_index=True)

                if not comp_display.empty:
                    # Convert date columns to .date for cleaner display
                    comp_display["Start"] = comp_display["Start Date"].dt.date
                    comp_display["End"] = comp_display["End Date"].apply(lambda x: x.date() if pd.notna(x) else "Ongoing")
                    comp_display["Duration"] = comp_display["Duration (Days)"]

                    # Calculate Depth Rank across ALL events including ongoing (1 = deepest)
                    if "Max Depth (%)" in comp_display.columns:
                        comp_display["Depth Rank"] = comp_display["Max Depth (%)"].rank(method='min').astype(int)

                    # Select and order columns for display (max-depth entry only)
                    display_cols = [
                        "Start", "End", "Duration", "Max Depth (%)", "Depth Rank",
                        "Max-Depth Entry Return (%)", "SPYSIM Max-Depth Return (%)",
                        "Max-Depth Entry Alpha (%)"
                    ]
                    comp_display = comp_display[[c for c in display_cols if c in comp_display.columns]]

                    # Sort by Start descending (most recent first)
                    comp_display = comp_display.sort_values("Start", ascending=False)

                    # Color styling function for alpha columns
                    def color_alpha(val):
                        if pd.isna(val):
                            return ''
                        color = '#00CC96' if val >= 0 else '#EF553B'
                        return f'color: {color}'

                    # Apply styling to alpha column
                    alpha_cols_present = [c for c in ["Max-Depth Entry Alpha (%)"] if c in comp_display.columns]
                    styled_df = comp_display.style.map(color_alpha, subset=alpha_cols_present)

                    # Column config with tooltips
                    comp_column_config = {
                        "Start": st.column_config.DateColumn("Start", help="Date when price dropped below the MA", format="YYYY-MM-DD"),
                        "End": st.column_config.TextColumn("End", help="Date when price recovered above the MA (or 'Ongoing')"),
                        "Duration": st.column_config.NumberColumn("Days", help="Total calendar days of the breach event", format="%.0f"),
                        "Max Depth (%)": st.column_config.NumberColumn("Max Depth", help="Maximum drawdown from breach start price during this event", format="%.1f%%"),
                        "Depth Rank": st.column_config.NumberColumn("Depth Rank", help="Depth rank among all breaches (1 = deepest)", format="%d"),
                        "Max-Depth Entry Return (%)": st.column_config.NumberColumn("Return", help="Portfolio return: entry at lowest point during breach, exit at recovery", format="%.1f%%"),
                        "SPYSIM Max-Depth Return (%)": st.column_config.NumberColumn("SPY Return", help="SPYSIM return for same max-depth to recovery window", format="%.1f%%"),
                        "Max-Depth Entry Alpha (%)": st.column_config.NumberColumn("Alpha", help="Outperformance vs SPY (positive = beat SPY)", format="%+.1f%%"),
                    }

                    st.dataframe(
                        styled_df,
                        column_config=comp_column_config,
                        use_container_width=True,
                        hide_index=True,
                        key=f"comparison_table_{key_suffix}"
                    )
            else:
                st.info("No breach events to display.")


# -----------------------------------------------------------------------------
# Munger 200 Week Moving Average Analysis
# -----------------------------------------------------------------------------
def render_munger_wma_tab(port_series, portfolio_name, unique_id, window=200):
    """
    Renders the Munger 200 Week Moving Average Analysis tab.
    Charlie Munger advocated for long-term thinking - the 200WMA (~4 years)
    filters out short-term noise and shows secular trends.
    """
    st.subheader(f"{portfolio_name} Munger {window}-Week Moving Average Analysis")

    st.info("""
    💡 **Munger's Wisdom:** *"The big money is not in the buying and selling, but in the waiting."*

    The 200-Week Moving Average (~4 years) filters out short-term noise and reveals secular trends.
    This indicator helps identify generational buying opportunities during major market dislocations.
    """)

    # Controls
    c_ctrl1, c_ctrl2 = st.columns(2)
    key_suffix = f"{unique_id}_wma_{window}" if unique_id else f"wma_{window}"

    merge_tol = c_ctrl1.slider(
        "Merge Events Tolerance (Weeks)",
        min_value=0, max_value=12, value=2, step=1,
        key=f"wma_merge_{key_suffix}",
        help=f"**Merge Tolerance**: Ignores short recoveries. If the price recovers above {window}WMA for fewer than X weeks before dropping again, it is considered a single continuous 'Under' event."
    )
    min_weeks = c_ctrl2.slider(
        "Signal Filter (Min Weeks)",
        min_value=0, max_value=52, value=4, step=1,
        key=f"wma_min_{key_suffix}",
        help=f"**Signal Filter**: Excludes short-lived drops below the {window}WMA (noise). Events shorter than X weeks will be hidden from the analysis."
    )

    # Calculate Weekly MA
    weekly_series, wma_series, events_df = calculations.analyze_wma(port_series, window=window, tolerance_weeks=merge_tol)

    if weekly_series is None or wma_series is None or wma_series.dropna().empty:
        st.warning(f"Insufficient data to calculate {window}WMA. Need at least {window} weeks (~{window/52:.1f} years) of data.")
        return

    filtered_events = pd.DataFrame()
    if events_df.empty:
        st.success(f"🎉 Price has never been below the {window}WMA in this period - Strong secular uptrend!")
    else:
        # Apply Min Weeks Filter
        filtered_events = events_df[events_df["Duration (Weeks)"] >= min_weeks]

    # Chart - Weekly data with WMA overlay
    fig = go.Figure()

    # Weekly Price (Blue)
    fig.add_trace(go.Scatter(
        x=weekly_series.index, y=weekly_series,
        name="Weekly Close",
        line=dict(color='#2E86C1', width=2),
        hovertemplate="Price: $%{y:,.0f}<extra></extra>"
    ))

    # Price Below WMA (Gold Overlay)
    price_below = weekly_series.copy()
    price_below[weekly_series >= wma_series] = None

    fig.add_trace(go.Scatter(
        x=price_below.index, y=price_below,
        name=f"Below {window}WMA",
        line=dict(color='#FFD700', width=2),
        hovertemplate="Price: $%{y:,.0f}<extra></extra>",
        showlegend=False
    ))

    # WMA Line (Red)
    fig.add_trace(go.Scatter(
        x=wma_series.index, y=wma_series,
        name=f"{window}WMA",
        line=dict(color='#E74C3C', width=1.5),
        hovertemplate=f"{window}WMA: $%{{y:,.0f}}<extra></extra>"
    ))

    # Add Peak markers
    if not events_df.empty and "Peak Date" in events_df.columns:
        filtered_peak_dates = set(filtered_events["Peak Date"].dropna()) if not filtered_events.empty else set()
        peak_data = events_df[["Peak Date", "Bottom to Peak (%)"]].dropna(subset=["Peak Date"])
        if not peak_data.empty:
            for _, row in peak_data.iterrows():
                d = row["Peak Date"]
                rally = row["Bottom to Peak (%)"]
                if d in weekly_series.index:
                    is_filtered = d in filtered_peak_dates
                    rally_val = rally if pd.notna(rally) else 0
                    fig.add_trace(go.Scatter(
                        x=[d], y=[weekly_series.loc[d]],
                        mode='markers',
                        name="Peak" if is_filtered else "Peak (Filtered)",
                        legendgroup="peak" if is_filtered else "peak_filtered",
                        showlegend=False,
                        visible=True if is_filtered else 'legendonly',
                        marker=dict(
                            symbol='diamond', size=10 if is_filtered else 7,
                            color='#00CC96' if is_filtered else 'rgba(100, 100, 100, 0.5)',
                            line=dict(width=1, color='white' if is_filtered else 'grey')
                        ),
                        hovertemplate=f"Peak: $%{{y:,.0f}} (+{rally_val:.1f}%)<br>%{{x|%b %d, %Y}}<extra></extra>"
                    ))

    # Add Bottom markers
    if not events_df.empty and "Bottom Date" in events_df.columns:
        filtered_bottom_dates = set(filtered_events["Bottom Date"].dropna()) if not filtered_events.empty else set()
        bottom_data = events_df[["Bottom Date", "Max Depth (%)"]].dropna(subset=["Bottom Date"])
        if not bottom_data.empty:
            for _, row in bottom_data.iterrows():
                d = row["Bottom Date"]
                depth = row["Max Depth (%)"]
                if d in weekly_series.index:
                    is_filtered = d in filtered_bottom_dates
                    depth_val = depth if pd.notna(depth) else 0
                    fig.add_trace(go.Scatter(
                        x=[d], y=[weekly_series.loc[d]],
                        mode='markers',
                        name="Bottom" if is_filtered else "Bottom (Filtered)",
                        legendgroup="bottom" if is_filtered else "bottom_filtered",
                        showlegend=False,
                        visible=True if is_filtered else 'legendonly',
                        marker=dict(
                            symbol='triangle-down', size=10 if is_filtered else 7,
                            color='#EF553B' if is_filtered else 'rgba(100, 100, 100, 0.5)',
                            line=dict(width=1, color='white' if is_filtered else 'grey')
                        ),
                        hovertemplate=f"Bottom: $%{{y:,.0f}} ({depth_val:.1f}%)<br>%{{x|%b %d, %Y}}<extra></extra>"
                    ))

    fig.update_layout(
        title=f"Weekly Price vs {window}WMA (Munger Indicator)",
        template="plotly_dark",
        height=500,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        xaxis_title="Date",
        xaxis=dict(range=[weekly_series.index[0], weekly_series.index[-1]]),
        yaxis_title="Price ($)",
        yaxis_type="log",
        hovermode="x unified"
    )
    st.plotly_chart(fig, use_container_width=True, key=f"wma_chart_{key_suffix}")

    if not events_df.empty:
        # Summary Metrics
        total_weeks = len(weekly_series)
        if not filtered_events.empty:
            weeks_under = filtered_events["Duration (Weeks)"].sum()
            pct_under = (weeks_under / total_weeks) * 100 if total_weeks > 0 else 0

            longest_event_idx = filtered_events["Duration (Weeks)"].idxmax()
            longest_event = filtered_events.loc[longest_event_idx]
            l_dur = longest_event['Duration (Weeks)']
            l_depth = longest_event['Max Depth (%)']

            # Median stats and range
            median_depth = filtered_events["Max Depth (%)"].median()
            min_depth = filtered_events["Max Depth (%)"].min()
            max_depth = filtered_events["Max Depth (%)"].max()
            total_breaches = len(filtered_events)

            if "Recovery Weeks" in filtered_events.columns:
                recovered_events = filtered_events[filtered_events["Recovery Weeks"].notna()]
                median_recovery = recovered_events["Recovery Weeks"].median() if not recovered_events.empty else None
            else:
                median_recovery = None

            if "Weeks Bottom to Peak" in filtered_events.columns:
                rally_events = filtered_events[filtered_events["Weeks Bottom to Peak"].notna()]
                median_rally_weeks = rally_events["Weeks Bottom to Peak"].median() if not rally_events.empty else None
            else:
                median_rally_weeks = None

            if "Bottom to Peak (%)" in filtered_events.columns:
                rally_pct_events = filtered_events[filtered_events["Bottom to Peak (%)"].notna()]
                median_rally_pct = rally_pct_events["Bottom to Peak (%)"].median() if not rally_pct_events.empty else None
            else:
                median_rally_pct = None

            if "Weeks to ATH" in filtered_events.columns:
                ath_events = filtered_events[filtered_events["Weeks to ATH"].notna()]
                median_weeks_to_ath = ath_events["Weeks to ATH"].median() if not ath_events.empty else None
            else:
                median_weeks_to_ath = None
        else:
            weeks_under = 0
            pct_under = 0
            l_dur = 0
            l_depth = 0
            median_recovery = None
            median_rally_weeks = None
            median_rally_pct = None
            median_depth = None
            min_depth = None
            max_depth = None
            total_breaches = 0
            median_weeks_to_ath = None

        # Check current status and calculate depth metrics
        last_price = weekly_series.iloc[-1]
        last_wma = wma_series.iloc[-1]
        current_depth = None
        current_depth_rank = None
        is_below = False
        status_text = ""
        status_delta = ""

        if events_df.empty:
            if last_price >= last_wma:
                status_text = "🟢 Above"
                status_delta = f"Never below {window}WMA"
            else:
                status_text = "🔴 Below"
                status_delta = f"First breach of {window}WMA"
                is_below = True
        elif events_df.iloc[-1]["Status"] == "Ongoing":
            last_event = events_df.iloc[-1]
            status_text = "🔴 Below"
            status_delta = f"{last_event['Duration (Weeks)']}w under {window}WMA"
            is_below = True
            # Calculate current depth
            start_date = last_event["Start Date"]
            event_prices = weekly_series[start_date:]
            if not event_prices.empty:
                start_price_val = event_prices.iloc[0]
                min_price = event_prices.min()
                current_depth = ((min_price - start_price_val) / start_price_val) * 100
                # Calculate depth rank
                if total_breaches > 0 and min_depth is not None:
                    all_depths = filtered_events["Max Depth (%)"].dropna().tolist()
                    sorted_depths = sorted(all_depths)
                    current_depth_rank = 1
                    for d in sorted_depths:
                        if current_depth <= d:
                            break
                        current_depth_rank += 1
        elif last_price < last_wma:
            status_text = "🔴 Below"
            status_delta = f"Just crossed {window}WMA"
            is_below = True
        else:
            last_event = events_df.iloc[-1]
            last_end = last_event["End Date"]
            if pd.notna(last_end):
                weeks_above = len(weekly_series[last_end:]) - 1
                status_text = "🟢 Above"
                status_delta = f"{weeks_above}w over {window}WMA"
            else:
                status_text = "🟢 Above"
                status_delta = f"{window}WMA"

        # Calculate recovery stats from similar depths (for below WMA state)
        recovery_rate_similar = None
        num_similar = 0
        num_recovered = 0
        med_recovery_similar = None
        max_recovery_similar = None

        if is_below and current_depth is not None and total_breaches >= 1:
            similar_or_deeper = filtered_events[filtered_events["Max Depth (%)"] <= current_depth]
            num_similar = len(similar_or_deeper)
            if num_similar > 0:
                recovered = similar_or_deeper[similar_or_deeper["Status"] == "Recovered"]
                num_recovered = len(recovered)
                recovery_rate_similar = (num_recovered / num_similar) * 100
                if num_recovered > 0 and "Recovery Weeks" in recovered.columns:
                    recovery_weeks_data = recovered["Recovery Weeks"].dropna()
                    if not recovery_weeks_data.empty:
                        med_recovery_similar = recovery_weeks_data.median()
                        max_recovery_similar = recovery_weeks_data.max()

        # Row 1: Current State (4 cols)
        r1c1, r1c2, r1c3, r1c4 = st.columns(4)
        r1c1.metric("Status", status_text, status_delta)

        if current_depth is not None:
            rank_text = f"Rank #{current_depth_rank}/{total_breaches}" if current_depth_rank else ""
            r1c2.metric("Current Depth", f"{current_depth:.1f}%", rank_text)
        elif median_depth is not None:
            depth_range = f"Range: {min_depth:.0f}% to {max_depth:.0f}%" if min_depth is not None else ""
            r1c2.metric("Med. Depth", f"{median_depth:.1f}%", depth_range)

        r1c3.metric("Time Under", f"{pct_under:.1f}%", f"{weeks_under}w total")
        r1c4.metric("Longest", f"{l_dur:.0f}w", f"Depth: {l_depth:.1f}%")

        # Row 2: Recovery Outlook (4 cols)
        r2c1, r2c2, r2c3, r2c4 = st.columns(4)

        if is_below and recovery_rate_similar is not None:
            # Show recovery outlook for current situation
            r2c1.metric("Recovery Rate", f"{recovery_rate_similar:.0f}%", f"{num_recovered}/{num_similar} similar")
            if med_recovery_similar is not None:
                r2c2.metric("Similar Recovery", f"{med_recovery_similar:.0f}w", f"Max: {max_recovery_similar:.0f}w")
            elif median_recovery is not None:
                r2c2.metric("Med. Recovery", f"{median_recovery:.0f}w", "Start→Even")
            if median_rally_pct is not None and median_rally_weeks is not None:
                r2c3.metric("Med. Rally", f"{median_rally_pct:.1f}%", f"{median_rally_weeks:.0f}w Bottom→Peak")
            elif median_rally_pct is not None:
                r2c3.metric("Rally Gain", f"{median_rally_pct:.1f}%", "Bottom→Peak")
            if median_weeks_to_ath is not None:
                r2c4.metric("To ATH", f"{median_weeks_to_ath:.0f}w", "Cross→New High")
        else:
            # Show general historical stats
            if median_recovery is not None:
                r2c1.metric("Med. Recovery", f"{median_recovery:.0f}w", "Start→Even")
            if median_rally_pct is not None and median_rally_weeks is not None:
                r2c2.metric("Med. Rally", f"{median_rally_pct:.1f}%", f"{median_rally_weeks:.0f}w Bottom→Peak")
            elif median_rally_weeks is not None:
                r2c2.metric("Med. Rally", f"{median_rally_weeks:.0f}w", "Bottom→Peak")
            if median_weeks_to_ath is not None:
                r2c3.metric("To ATH", f"{median_weeks_to_ath:.0f}w", "Cross→New High")
            r2c4.metric("# Breaches", f"{total_breaches}", "Historical events")

        # Events Table
        st.subheader(f"Periods Under {window}WMA")

        with st.expander("ℹ️ Understanding Munger 200WMA Metrics"):
            st.markdown(f"""
**Why 200 Weeks (~4 Years)?**

Charlie Munger and Warren Buffett emphasize patience and long-term thinking. The 200-Week Moving Average:
- Filters out noise from business cycles and market corrections
- Reveals true secular trends
- Historically, drops below the 200WMA represent major bear markets or generational buying opportunities

**Historical Context:**
- S&P 500 has dropped below its 200WMA only a handful of times in the last 50 years
- Major instances: 1974, 2002-2003, 2008-2009, 2020 (briefly), 2022
- These often marked exceptional long-term entry points

**Row 1 - Current State:**
| Metric | Meaning |
|--------|---------|
| **Status** | Current position: 🟢 Above or 🔴 Below the {window}WMA, with duration |
| **Current Depth** | *(When below)* Current drawdown with rank (1 = deepest in history) |
| **Med. Depth** | *(When above)* Typical drawdown with historical range |
| **Time Under** | % of period spent below WMA, with total weeks |
| **Longest** | Longest breach duration with its max depth |

**Row 2 - Recovery Outlook** *(when below WMA)*:
| Metric | Meaning |
|--------|---------|
| **Recovery Rate** | % of similar-depth breaches that recovered |
| **Similar Recovery** | Median recovery time from similar depths, with max |
| **Med. Rally** | Typical rally gain (bottom to peak) |
| **To ATH** | Typical weeks to new all-time high |

**Table Columns:**
| Column | Meaning |
|--------|---------|
| **Start / End** | When price dropped below / recovered above the {window}WMA |
| **Weeks Under WMA** | Duration in weeks |
| **Max Depth** | Maximum drawdown from event start |
| **Breakeven Gain** | Rally needed from bottom to recover start price |
| **Recovery Weeks** | Weeks from start to breakeven |
| **Full Rally %** | Gain from bottom to subsequent peak |
| **Rally Weeks** | Weeks from bottom to peak |
| **Weeks to ATH** | Weeks from WMA crossover to new all-time high |
            """)

        display_df = filtered_events.copy()
        if not display_df.empty:
            display_df["Start"] = display_df["Start Date"].dt.date
            display_df["End"] = display_df["End Date"].apply(lambda x: x.date() if pd.notna(x) else "Ongoing")
            if "Peak Date" in display_df.columns:
                display_df["Peak Date"] = display_df["Peak Date"].apply(lambda x: x.date() if pd.notna(x) else "-")

            # Recovery Pattern Classification
            def classify_recovery(row):
                status = row.get("Status", "")
                if "Recovered" not in str(status):
                    return status

                weeks = row.get("Weeks Bottom to Peak")
                pct = row.get("Bottom to Peak (%)")

                if pd.isna(weeks) or pd.isna(pct):
                    return status

                median_weeks = filtered_events["Weeks Bottom to Peak"].median()
                median_pct = filtered_events["Bottom to Peak (%)"].median()

                short_weeks = weeks <= median_weeks
                high_pct = pct >= median_pct

                if short_weeks and high_pct:
                    return "⚡ V-Shape"
                elif not short_weeks and high_pct:
                    return "📈 Grind"
                elif not short_weeks and not high_pct:
                    return "🐌 Choppy"
                else:
                    return "📉 Weak"

            display_df["Pattern"] = display_df.apply(classify_recovery, axis=1)

            cols_map = {
                "Start": "Start",
                "End": "End",
                "Duration (Weeks)": "Weeks Under WMA",
                "Duration (Years)": "Years",
                "Max Depth (%)": "Max Depth",
                "Bottom to Recovery (%)": "Breakeven Gain",
                "Subsequent Peak (%)": "Post-WMA Rally",
                "Post-WMA Rally Weeks": "Post-WMA Rally Wks",
                "Bottom to Peak (%)": "Full Rally %",
                "Weeks Bottom to Peak": "Rally Weeks",
                "Recovery Weeks": "Recovery Weeks",
                "Weeks to ATH": "Weeks to ATH",
                "Status": "Status",
                "Pattern": "Pattern",
                "Entry WMA": "Entry WMA",
                "Exit WMA": "Exit WMA",
                "WMA Change (%)": "WMA Δ%",
                "True Recovery Weeks": "True Recovery"
            }

            final_cols = [c for c in cols_map.keys() if c in display_df.columns or c in ["Start", "End"]]
            display_df = display_df[final_cols].rename(columns=cols_map)
            display_df = display_df.sort_values("Start", ascending=False)

            display_df["Status"] = display_df["Status"].apply(
                lambda x: f"🟡 {x}" if "Ongoing" in str(x) or "Current" in str(x) else x
            )

            # Toggle WMA context
            show_wma_context = st.checkbox(
                "Show WMA Context",
                value=False,
                help="Show Entry WMA, Exit WMA, WMA Δ%, and True Recovery columns.",
                key=f"wma_context_{unique_id}_{window}"
            )

            if show_wma_context:
                final_display_cols = [
                    "Start", "End", "Weeks Under WMA", "Years", "Max Depth", "Breakeven Gain",
                    "Recovery Weeks", "True Recovery",
                    "Entry WMA", "Exit WMA", "WMA Δ%",
                    "Full Rally %", "Rally Weeks", "Post-WMA Rally", "Post-WMA Rally Wks",
                    "Weeks to ATH", "Status", "Pattern"
                ]
            else:
                final_display_cols = [
                    "Start", "End", "Weeks Under WMA", "Years", "Max Depth", "Breakeven Gain",
                    "Recovery Weeks",
                    "Full Rally %", "Rally Weeks", "Post-WMA Rally", "Post-WMA Rally Wks",
                    "Weeks to ATH", "Status", "Pattern"
                ]
            final_display_cols = [c for c in final_display_cols if c in display_df.columns]
            display_df = display_df[final_display_cols]

            column_config = {
                "Start": st.column_config.DateColumn("Start", help="Date when price dropped below the WMA", format="YYYY-MM-DD"),
                "End": st.column_config.DateColumn("End", help="Date when price recovered above the WMA", format="YYYY-MM-DD"),
                "Weeks Under WMA": st.column_config.NumberColumn("Weeks Under WMA", help="Total weeks spent below the WMA", format="%.0f"),
                "Years": st.column_config.NumberColumn("Years", help="Duration in years", format="%.1f"),
                "Max Depth": st.column_config.NumberColumn("Max Depth", help="Price drawdown from event start to lowest point", format="%.1f%%"),
                "Breakeven Gain": st.column_config.NumberColumn("Breakeven Gain", help="% gain needed from bottom to recover start price", format="%.1f%%"),
                "Recovery Weeks": st.column_config.NumberColumn("Recovery Weeks", help="Weeks from event start to breakeven", format="%.0f"),
                "True Recovery": st.column_config.NumberColumn("True Recovery", help="Weeks until price ≥ start AND > WMA", format="%.0f"),
                "Entry WMA": st.column_config.NumberColumn("Entry WMA", help="WMA value when price dropped below", format="$%.2f"),
                "Exit WMA": st.column_config.NumberColumn("Exit WMA", help="WMA value when price recovered", format="$%.2f"),
                "WMA Δ%": st.column_config.NumberColumn("WMA Δ%", help="WMA change during event", format="%+.1f%%"),
                "Full Rally %": st.column_config.NumberColumn("Full Rally %", help="Gain from bottom to peak", format="%.1f%%"),
                "Rally Weeks": st.column_config.NumberColumn("Rally Weeks", help="Weeks from bottom to peak", format="%.0f"),
                "Post-WMA Rally": st.column_config.NumberColumn("Post-WMA Rally", help="Gain from WMA crossover to peak", format="%.1f%%"),
                "Post-WMA Rally Wks": st.column_config.NumberColumn("Post-WMA Rally Wks", help="Weeks from WMA crossover to peak", format="%.0f"),
                "Weeks to ATH": st.column_config.NumberColumn("Weeks to ATH", help="Weeks from WMA crossover to new ATH", format="%.0f"),
                "Status": st.column_config.TextColumn("Status", help="Recovery status"),
                "Pattern": st.column_config.TextColumn("Pattern", help="Recovery shape classification"),
            }

            st.dataframe(
                display_df,
                column_config=column_config,
                use_container_width=True,
                hide_index=True
            )

            # Entry Strategy Comparison Table
            st.subheader(f"{portfolio_name} Entry Strategy Comparison (vs SPYSIM)")

            with st.expander("ℹ️ Understanding the Metrics"):
                st.markdown("""
**Strategy:** Buy at the **maximum depth** (lowest point) during each WMA breach, sell when price recovers above the WMA. Compare returns to buying SPY at the same time.

**Summary Metrics:**
| Metric | Meaning |
|--------|---------|
| **Total Events** | Number of completed (recovered) breach events analyzed |
| **Win Rate** | % of events where buying at max-depth beat buying SPY |
| **Avg Alpha** | Average outperformance vs SPY across all events |
| **Median Alpha** | Typical (median) outperformance vs SPY |

**Table Columns:**
| Column | Meaning |
|--------|---------|
| **Start / End** | When price dropped below / recovered above the WMA |
| **Weeks** | Duration of the breach event in weeks |
| **Max Depth** | Maximum drawdown from breach start price during this event |
| **Depth Rank** | Depth rank among all breaches (1 = deepest in history) |
| **Return** | Portfolio return from max-depth entry to recovery |
| **SPY Return** | SPYSIM return for the same window (max-depth to recovery) |
| **Alpha** | Outperformance vs SPY (Return - SPY Return). Green = beat SPY, Red = underperformed |
                """)

            comparison_df = calculations.compare_wma_breach_events(
                port_series,
                window=window,
                tolerance_weeks=merge_tol
            )

            # Also get ongoing event if exists (from the full events_df)
            _, _, all_events_df = calculations.analyze_wma(port_series, window=window, tolerance_weeks=merge_tol)
            ongoing_event = None
            if not all_events_df.empty:
                last_event = all_events_df.iloc[-1]
                if last_event["Status"] == "Ongoing":
                    ongoing_event = last_event

            # Summary Statistics Row (only from recovered events)
            if not comparison_df.empty or ongoing_event is not None:
                total_recovered = len(comparison_df) if not comparison_df.empty else 0

                # Extract alpha column with NA handling
                maxdepth_alpha = comparison_df["Max-Depth Entry Alpha (%)"].dropna() if not comparison_df.empty else pd.Series(dtype=float)

                # Calculate statistics
                maxdepth_wins = (maxdepth_alpha > 0).sum() if len(maxdepth_alpha) > 0 else 0
                maxdepth_win_rate = (maxdepth_wins / len(maxdepth_alpha) * 100) if len(maxdepth_alpha) > 0 else 0
                maxdepth_avg = maxdepth_alpha.mean() if len(maxdepth_alpha) > 0 else 0
                maxdepth_median = maxdepth_alpha.median() if len(maxdepth_alpha) > 0 else 0

                # Display metrics row
                m1, m2, m3, m4, m5 = st.columns(5)
                m1.metric("Total Events", total_recovered + (1 if ongoing_event is not None else 0))
                m2.metric("Win Rate", f"{maxdepth_win_rate:.1f}%", help="% of recovered events where max-depth entry beat SPY")
                m3.metric("Avg Alpha", f"{maxdepth_avg:+.1f}%", help="Average outperformance vs SPY (recovered events)")
                m4.metric("Median Alpha", f"{maxdepth_median:+.1f}%", help="Median outperformance vs SPY (recovered events)")
                m5.metric("Events Analyzed", f"{len(maxdepth_alpha)}/{total_recovered}" + (" +1 ongoing" if ongoing_event is not None else ""))

                st.markdown("---")

                # Prepare display DataFrame
                comp_display = comparison_df.copy() if not comparison_df.empty else pd.DataFrame()

                # Add ongoing event to display if exists
                if ongoing_event is not None:
                    ongoing_row = {
                        "Start Date": ongoing_event["Start Date"],
                        "End Date": pd.NaT,
                        "Duration (Weeks)": ongoing_event["Duration (Weeks)"],
                        "Max Depth (%)": ongoing_event["Max Depth (%)"],
                        "Max-Depth Entry Return (%)": None,
                        "SPYSIM Max-Depth Return (%)": None,
                        "Max-Depth Entry Alpha (%)": None,
                        "Status": "🟠 Ongoing"
                    }
                    ongoing_df = pd.DataFrame([ongoing_row])
                    comp_display = pd.concat([ongoing_df, comp_display], ignore_index=True)

                if not comp_display.empty:
                    # Convert date columns to .date for cleaner display
                    comp_display["Start"] = comp_display["Start Date"].dt.date
                    comp_display["End"] = comp_display["End Date"].apply(lambda x: x.date() if pd.notna(x) else "Ongoing")
                    comp_display["Duration"] = comp_display["Duration (Weeks)"]

                    # Calculate Depth Rank across ALL events including ongoing (1 = deepest)
                    if "Max Depth (%)" in comp_display.columns:
                        comp_display["Depth Rank"] = comp_display["Max Depth (%)"].rank(method='min').astype(int)

                    display_cols = [
                        "Start", "End", "Duration", "Max Depth (%)", "Depth Rank",
                        "Max-Depth Entry Return (%)", "SPYSIM Max-Depth Return (%)",
                        "Max-Depth Entry Alpha (%)"
                    ]
                    comp_display = comp_display[[c for c in display_cols if c in comp_display.columns]]
                    comp_display = comp_display.sort_values("Start", ascending=False)

                    def color_alpha(val):
                        if pd.isna(val):
                            return ''
                        color = '#00CC96' if val >= 0 else '#EF553B'
                        return f'color: {color}'

                    alpha_cols_present = [c for c in ["Max-Depth Entry Alpha (%)"] if c in comp_display.columns]
                    styled_df = comp_display.style.map(color_alpha, subset=alpha_cols_present)

                    comp_column_config = {
                        "Start": st.column_config.DateColumn("Start", help="Date when price dropped below the WMA", format="YYYY-MM-DD"),
                        "End": st.column_config.TextColumn("End", help="Date when price recovered above the WMA (or 'Ongoing')"),
                        "Duration": st.column_config.NumberColumn("Weeks", help="Total weeks of the breach event", format="%.0f"),
                        "Max Depth (%)": st.column_config.NumberColumn("Max Depth", help="Maximum drawdown from breach start price during this event", format="%.1f%%"),
                        "Depth Rank": st.column_config.NumberColumn("Depth Rank", help="Depth rank among all breaches (1 = deepest)", format="%d"),
                        "Max-Depth Entry Return (%)": st.column_config.NumberColumn("Return", help="Portfolio return: entry at lowest point during breach, exit at recovery", format="%.1f%%"),
                        "SPYSIM Max-Depth Return (%)": st.column_config.NumberColumn("SPY Return", help="SPYSIM return for same max-depth to recovery window", format="%.1f%%"),
                        "Max-Depth Entry Alpha (%)": st.column_config.NumberColumn("Alpha", help="Outperformance vs SPY (positive = beat SPY)", format="%+.1f%%"),
                    }

                    st.dataframe(
                        styled_df,
                        column_config=comp_column_config,
                        use_container_width=True,
                        hide_index=True,
                        key=f"wma_comparison_table_{key_suffix}"
                    )
            else:
                st.info("No breach events to display.")


# -----------------------------------------------------------------------------
# Cheat Sheet Analysis
# -----------------------------------------------------------------------------
def render_cheat_sheet(port_series, portfolio_name, unique_id, component_data=None):
    # Determine target series and name
    target_series = port_series
    target_name = portfolio_name
    
    if component_data is not None and not component_data.empty:
        if isinstance(component_data, pd.DataFrame):
            cols = list(component_data.columns)
            if len(cols) == 1:
                    target_name = cols[0]
                    target_series = component_data[target_name]
            elif len(cols) > 1:
                    target_name = st.selectbox("Select Asset to Analyze", cols, key=f"cs_sel_{unique_id}")
                    target_series = component_data[target_name]
                    
    # Fetch OHLC for Pivot Points
    ohlc_data = None
    try:
            from app.core.shadow_backtest import parse_ticker
            import yfinance as yf
            
            mapped_ticker, _ = parse_ticker(target_name)
            if not target_series.empty:
                last_dt = target_series.index[-1]
                start_f = (last_dt - pd.Timedelta(days=7)).strftime('%Y-%m-%d')
                end_f = (last_dt + pd.Timedelta(days=3)).strftime('%Y-%m-%d')
                
                base_yf = yf.Ticker(mapped_ticker)
                hist_ohlc = base_yf.history(start=start_f, end=end_f, auto_adjust=True)
                
                if hist_ohlc.index.tz is not None:
                    hist_ohlc.index = hist_ohlc.index.tz_localize(None)

                if not hist_ohlc.empty:
                    if last_dt in hist_ohlc.index:
                        ref_bar = hist_ohlc.loc[last_dt]
                    else:
                        ref_bar = hist_ohlc.iloc[-1]
                    
                    ohlc_data = {
                        'High': ref_bar['High'], 
                        'Low': ref_bar['Low'], 
                        'Close': ref_bar['Close']
                    }
    except Exception:
            pass
            
    st.subheader(f"{target_name} Trader's Cheat Sheet")
    cs_df = calculations.calculate_cheat_sheet(target_series, ohlc_data=ohlc_data)
    
    if cs_df is None or cs_df.empty:
            st.info("Insufficient data for Technical Analysis.")
    else:
            # Transform to 3-Column Layout (Barchart Style)
            # Left Column: High/Low, StdDev, Pivot S/R, Session Levels
            # Right Column: Moving Averages, Fibonacci, Pivot Point (P)
            
            left_types = ["High/Low", "StdDev", "Pivot Support", "Pivot Resistance", "Session Level"]
            
            display_rows = []
            for _, row in cs_df.iterrows():
                lbl = row['Label']
                typ = row['Type']
                px = row['Price']
                
                # Format Price as String immediately to allow centering (Numeric defaults to Right)
                px_str = "{:,.2f}".format(px)
                
                if typ == "Current":
                    display_rows.append({"Support/Resistance Levels": "Latest", "Price": px_str, "Key Turning Points": "Latest", "Type": "Current"})
                elif typ in left_types:
                    display_rows.append({"Support/Resistance Levels": lbl, "Price": px_str, "Key Turning Points": "", "Type": "Left"})
                else:
                    display_rows.append({"Support/Resistance Levels": "", "Price": px_str, "Key Turning Points": lbl, "Type": "Right"})
        
            disp_df = pd.DataFrame(display_rows)
            
            # Identify Current Price Row Index for coloring
            try:
                curr_idx = disp_df[disp_df["Type"] == "Current"].index[0]
            except IndexError:
                curr_idx = -1

            # Dark Mode Toggle
            is_dark = st.toggle("Dark Mode Colors", value=True, key=f"cs_dark_{unique_id}")
            
            if is_dark:
                # Dark Mode Palette
                c_res_bg = "#4a1c1c" # Dark Red BG
                c_res_txt = "#ffcdd2" # Light Red Text
                c_sup_bg = "#1b3e20" # Dark Green BG
                c_sup_txt = "#c8e6c9" # Light Green Text
                c_neutral_bg = "#262730" # Streamlit Secondary Dark BG
                c_neutral_txt = "#fafafa" # White Text
                c_current_bg = "#FFD700"
                c_current_txt = "black"
            else:
                # Light Mode Palette (Barchart)
                c_res_bg = "#FFEBEE"
                c_res_txt = "#B71C1C"
                c_sup_bg = "#E8F5E9"
                c_sup_txt = "#1B5E20"
                c_neutral_bg = "white"
                c_neutral_txt = "#333333"
                c_current_bg = "#FFD700"
                c_current_txt = "black"

            def style_barchart(row):
                idx = row.name
                styles = []
                
                # Determine Active Colors based on Row Position
                if idx < curr_idx: # Resistance
                    bg_active = c_res_bg
                    txt_active = c_res_txt
                    weight = "600"
                elif idx > curr_idx: # Support
                    bg_active = c_sup_bg
                    txt_active = c_sup_txt
                    weight = "600"
                else: # Current
                    # Current Row is special: Full Gold
                    return [f'background-color: {c_current_bg}; color: {c_current_txt}; font-weight: bold; text-align: center !important; vertical-align: middle;'] * len(row)

                # Default (Empty/Price) Style
                bg_neutral = c_neutral_bg
                txt_neutral = c_neutral_txt
                
                # Helper to check if string is non-empty
                def is_populated(val):
                    return bool(val and str(val).strip())

                # Col 0: Support/Resistance Levels (Left) => Right Align
                # Using iloc to access by position is safer if col names change slightly
                val_left = row.iloc[0] 
                if is_populated(val_left):
                    s_left = f'background-color: {bg_active}; color: {txt_active}; font-weight: {weight};'
                else:
                    s_left = f'background-color: {bg_neutral}; color: {txt_neutral};'
                styles.append(f'{s_left} text-align: right !important; padding-right: 15px; vertical-align: middle;')
                
                # Col 1: Price (Center) => Center Align
                styles.append(f'background-color: {bg_neutral}; color: {txt_neutral}; font-weight: normal; text-align: center !important; vertical-align: middle;')
                
                # Col 2: Key Turning Points (Right) => Left Align
                val_right = row.iloc[2]
                if is_populated(val_right):
                    s_right = f'background-color: {bg_active}; color: {txt_active}; font-weight: {weight};'
                else:
                    s_right = f'background-color: {bg_neutral}; color: {txt_neutral};'
                styles.append(f'{s_right} text-align: left !important; padding-left: 15px; vertical-align: middle;')
                
                return styles

            # Display
            final_view = disp_df.drop(columns=["Type"])
            
            st.dataframe(
                final_view.style.apply(style_barchart, axis=1), 
                use_container_width=True, 
                height=(len(final_view) + 1) * 35,
                hide_index=True
            )
            
            # Legend and Explanation
            st.markdown("""
            <div style="margin-top: 20px; font-size: 0.9em; color: #888;">
            <p><strong>Standard deviation</strong> is calculated using the closing price over the past 20-periods. To calculate standard deviation:</p>
            <ul style="list-style-type: disc; margin-left: 20px;">
                <li><strong>Step 1:</strong> Average = Calculate the average closing price over the past 20-days.</li>
                <li><strong>Step 2:</strong> Difference = Calculate the variance from the Average for each Price.</li>
                <li><strong>Step 3:</strong> Square the variance of each data point.</li>
                <li><strong>Step 4:</strong> Sum of the squared variance value.</li>
                <li><strong>Step 5:</strong> For Standard Deviation 2 multiple the result by 2. For Standard Deviation 3 multiple the result by 3.</li>
                <li><strong>Step 6:</strong> Divide the result by the number of data points in the series less 1.</li>
                <li><strong>Step 7:</strong> The final result is the Square root of the result of Step 6.</li>
            </ul>
            
            <p><strong>Legend:</strong></p>
            <ul style="list-style-type: none; padding-left: 10px;">
                <li><span style="color: #E8F5E9; background-color: #E8F5E9; border: 1px solid #ccc;">&nbsp;&nbsp;&nbsp;&nbsp;</span> <strong>Green areas below the Last Price</strong> will tend to provide support to limit the downward move.</li>
                <li><span style="color: #FFEBEE; background-color: #FFEBEE; border: 1px solid #ccc;">&nbsp;&nbsp;&nbsp;&nbsp;</span> <strong>Red areas above the Last Price</strong> will tend to provide resistance to limit the upward move.</li>
            </ul>
            </div>
            """, unsafe_allow_html=True)

def render_returns_analysis(port_series, bench_series=None, comparison_series=None, unique_id="", portfolio_name="Strategy", component_data=None, raw_port_series=None):
    daily_ret = port_series.pct_change().dropna()
    monthly_ret = port_series.resample("ME").last().pct_change().dropna()
    quarterly_ret = port_series.resample("QE").last().pct_change().dropna()
    annual_ret = port_series.resample("YE").last().pct_change().dropna()

    # --- HEATMAP HELPERS ---
    def render_seasonal_summary(series, suffix=""):
        full_suffix = f"{unique_id}_{suffix}" if unique_id else suffix
        if series.empty: return
        
        # 1. Prepare Data (Same as Monthly View)
        m_ret = series.resample("ME").last().pct_change().dropna()
        df_monthly = m_ret.to_frame(name="Return")
        df_monthly["Year"] = df_monthly.index.year
        df_monthly["Month"] = df_monthly.index.month
        
        pivot = df_monthly.pivot(index="Year", columns="Month", values="Return")
        for i in range(1, 13):
            if i not in pivot.columns: pivot[i] = float("nan")
        pivot = pivot.sort_index(ascending=True)
        month_names = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
        
        # 2. Add Yearly Return Column
        yearly_col = []
        years = pivot.index
        for y in years:
            row = pivot.loc[y]
            ret = (1 + row.fillna(0)).prod() - 1
            yearly_col.append(ret)
        
        # Add Yearly column to pivot for display/calculation
        display_pivot = pivot.copy()
        display_pivot.columns = month_names
        # display_pivot["Yearly Return"] = yearly_col # Removed per user request
        
        # 3. Calculate Summary Statistics Table
        st.subheader("Summary")
        
        # Create a DF for stats calc: Join pivot (months) and yearly_col
        stats_source = display_pivot.copy() # Columns: Jan..Dec, Yearly Return
        
        # Define rows
        stats_rows = ["Average", "% Positive", "% Negative", "Median", "Best", "Worst", "Abs Average", "Abs Best", "Abs Worst"]
        stats_df = pd.DataFrame(index=stats_rows, columns=stats_source.columns)
        
        for col in stats_source.columns:
            s_data = stats_source[col].dropna()
            if s_data.empty:
                continue
                
            stats_df.loc["Average", col] = s_data.mean()
            stats_df.loc["% Positive", col] = (s_data > 0).mean()
            stats_df.loc["% Negative", col] = (s_data < 0).mean()
            stats_df.loc["Median", col] = s_data.median()
            stats_df.loc["Best", col] = s_data.max()
            stats_df.loc["Worst", col] = s_data.min()
            stats_df.loc["Abs Average", col] = s_data.abs().mean()
            stats_df.loc["Abs Best", col] = s_data.abs().max()
            stats_df.loc["Abs Worst", col] = s_data.abs().min()
            
        stats_df = stats_df.astype(float)
        
        return_rows = ["Average", "Median", "Best", "Worst"]
        pct_rows = ["% Positive", "% Negative"]
        abs_rows = ["Abs Average", "Abs Best", "Abs Worst"]
        
        def style_summary(df):
            return df.style.format("{:.2%}") \
                .map(color_return, subset=pd.IndexSlice[return_rows, :]) \
                .background_gradient(cmap="Greens", subset=pd.IndexSlice["% Positive", :], vmin=0, vmax=1) \
                .background_gradient(cmap="Reds", subset=pd.IndexSlice["% Negative", :], vmin=0, vmax=1) \
                .background_gradient(cmap="Oranges", subset=pd.IndexSlice[abs_rows, :])

        st.dataframe(style_summary(stats_df), use_container_width=True)

    # --- HEATMAP HELPERS ---
    def render_quarterly_returns_view(series, suffix=""):
        # Combine unique_id with suffix for truly unique keys
        full_suffix = f"{unique_id}_{suffix}" if unique_id else suffix
        if series.empty: return
        quarterly_ret = series.resample("QE").last().pct_change().dropna()

        
        q_ret = quarterly_ret.to_frame(name="Return")
        q_ret["Year"] = q_ret.index.year
        q_ret["Quarter"] = q_ret.index.quarter
        q_ret["Quarter Name"] = "Q" + q_ret["Quarter"].astype(str)
        
        pivot = q_ret.pivot(index="Year", columns="Quarter", values="Return")
        for i in range(1, 5):
            if i not in pivot.columns: pivot[i] = float("nan")
        pivot = pivot.sort_index(ascending=True)
        


        quarter_names = ["Q1", "Q2", "Q3", "Q4"]
        quarterly_avgs = pivot.mean()
        z_data = pivot.values
        z_avgs = quarterly_avgs.values.reshape(1, -1)
        z_combined_main = np.concatenate([z_data, z_avgs], axis=0)
        
        years = pivot.index
        yearly_col = []
        for y in years:
            # Calculate geometric sum of the quarters for consistency
            row = pivot.loc[y]
            ret = (1 + row.fillna(0)).prod() - 1
            yearly_col.append(ret)
        yearly_avg = np.nanmean(yearly_col) if len(yearly_col) > 0 else float("nan")
        z_combined_yearly = np.array(yearly_col + [yearly_avg]).reshape(-1, 1)

        y_labels = [str(y) for y in pivot.index] + ["Average"]
        
        # Scaling
        std_dev_main = np.nanstd(z_combined_main * 100)
        scale_main = 2 * std_dev_main if not np.isnan(std_dev_main) else 10
        std_dev_yearly = np.nanstd(z_combined_yearly * 100)
        scale_yearly = 2 * std_dev_yearly if not np.isnan(std_dev_yearly) else 10
        colorscale_heatmap = [[0, '#E53935'], [0.5, '#FFFFFF'], [1, '#43A047']]

        # Hover Text
        date_map = {}
        try:
            periods = series.index.to_period("Q")
            for p, dates in series.index.groupby(periods).items():
                if not dates.empty:
                    date_map[(p.year, p.quarter)] = f"{dates.min().strftime('%b %d')} - {dates.max().strftime('%b %d')}"
        except: pass

        hover_main = []
        z_rounded_main = (z_combined_main * 100).round(2)
        for i, row_label in enumerate(y_labels):
            row_txt = []
            for j, col_label in enumerate(quarter_names):
                val = z_rounded_main[i][j]
                if np.isnan(val): row_txt.append("")
                elif row_label == "Average": row_txt.append(f"Average<br>{col_label}: {val:+.2f}%")
                else:
                    dr = date_map.get((int(row_label), j+1), "") if row_label.isdigit() else ""
                    dr_str = f"<br>{dr}" if dr else ""
                    row_txt.append(f"Year: {row_label}<br>{col_label}: {val:+.2f}%{dr_str}")
            hover_main.append(row_txt)

        hover_yearly = []
        z_rounded_yearly = (z_combined_yearly * 100).round(2)
        for i, row_label in enumerate(y_labels):
            val = z_rounded_yearly[i][0]
            val_str = "" if np.isnan(val) else f"{val:+.2f}%"
            if row_label == "Average": hover_yearly.append([f"Average Annual<br>{val_str}"])
            else: hover_yearly.append([f"Year: {row_label}<br>Annual: {val_str}"])

        # Plot
        n_years = len(y_labels) - 1
        fig = make_subplots(rows=2, cols=2, shared_xaxes=True, shared_yaxes=True,
            horizontal_spacing=0.03, vertical_spacing=0.02, column_widths=[0.85, 0.15],
            row_heights=[n_years/(n_years+1), 1/(n_years+1)]
        )

        fig.add_trace(go.Heatmap(z=(z_combined_main[:-1] * 100).round(2), x=quarter_names, y=y_labels[:-1], colorscale=colorscale_heatmap, zmid=0, zmin=-scale_main, zmax=scale_main, texttemplate="%{z:+.2f}%", hoverinfo="text", hovertext=hover_main[:-1], showscale=False), row=1, col=1)
        fig.add_trace(go.Heatmap(z=(z_combined_yearly[:-1] * 100).round(2), x=["Yearly"], y=y_labels[:-1], colorscale=colorscale_heatmap, zmid=0, zmin=-scale_yearly, zmax=scale_yearly, texttemplate="%{z:+.2f}%", hoverinfo="text", hovertext=hover_yearly[:-1], showscale=False), row=1, col=2)
        fig.add_trace(go.Heatmap(z=(z_combined_main[-1:] * 100).round(2), x=quarter_names, y=y_labels[-1:], colorscale=colorscale_heatmap, zmid=0, zmin=-scale_main, zmax=scale_main, texttemplate="%{z:+.2f}%", hoverinfo="text", hovertext=hover_main[-1:], showscale=False), row=2, col=1)
        fig.add_trace(go.Heatmap(z=(z_combined_yearly[-1:] * 100).round(2), x=["Yearly"], y=y_labels[-1:], colorscale=colorscale_heatmap, zmid=0, zmin=-scale_yearly, zmax=scale_yearly, texttemplate="%{z:+.2f}%", hoverinfo="text", hovertext=hover_yearly[-1:], showscale=False), row=2, col=2)

        fig.update_layout(title="Quarterly Returns Heatmap (%)", template="plotly_white", height=max(400, (len(y_labels)+1)*30), yaxis=dict(autorange="reversed", type="category"), yaxis3=dict(autorange="reversed", type="category"))
        fig.update_yaxes(showticklabels=False, col=2)
        st.plotly_chart(fig, use_container_width=True, key=f"q_hm_{full_suffix}")
        
        st.subheader("Quarterly Returns List")
        df_quarterly_list = q_ret.copy()
        df_quarterly_list["Period"] = df_quarterly_list.index.to_period("Q").astype(str)
        df_quarterly_list = df_quarterly_list[["Period", "Return"]].sort_index(ascending=False)
        
        st.dataframe(
            df_quarterly_list.style.format({"Return": "{:+.1%}"}).map(color_return, subset=["Return"]),
            use_container_width=True,
            hide_index=True
        )

    def render_monthly_returns_view(series, suffix=""):
        # Combine unique_id with suffix for truly unique keys
        full_suffix = f"{unique_id}_{suffix}" if unique_id else suffix
        if series.empty: return
        m_ret = series.resample("ME").last().pct_change().dropna()

        
        df_monthly = m_ret.to_frame(name="Return")
        df_monthly["Year"] = df_monthly.index.year
        df_monthly["Month"] = df_monthly.index.month
        
        pivot = df_monthly.pivot(index="Year", columns="Month", values="Return")
        for i in range(1, 13):
            if i not in pivot.columns: pivot[i] = float("nan")
        pivot = pivot.sort_index(ascending=True)
        month_names = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
        monthly_avgs = pivot.mean()

        z_data = pivot.values
        z_combined_main = np.concatenate([z_data, monthly_avgs.values.reshape(1, -1)], axis=0)
        
        years = pivot.index
        yearly_col = []
        for y in years:
            # Calculate geometric sum of the months for consistency
            # This ensures the Yearly column matches the compounded value of the displayed months
            row = pivot.loc[y]
            # compound: product(1+r) - 1. Treat NaNs as 0 (no return for that period)
            ret = (1 + row.fillna(0)).prod() - 1
            yearly_col.append(ret)
        yearly_avg = np.nanmean(yearly_col) if len(yearly_col) > 0 else float("nan")
        z_combined_yearly = np.array(yearly_col + [yearly_avg]).reshape(-1, 1)

        y_labels = [str(y) for y in pivot.index] + ["Average"]
        
        std_dev_main = np.nanstd(z_combined_main * 100)
        scale_main = 2 * std_dev_main if not np.isnan(std_dev_main) else 10
        std_dev_yearly = np.nanstd(z_combined_yearly * 100)
        scale_yearly = 1.0 * std_dev_yearly if not np.isnan(std_dev_yearly) else 10
        colorscale_heatmap = [[0, '#E53935'], [0.5, '#FFFFFF'], [1, '#43A047']]
        
        hover_main = []
        z_rounded_main = (z_combined_main * 100).round(2)
        for i, row_label in enumerate(y_labels):
            row_txt = []
            for j, col_label in enumerate(month_names):
                val = z_rounded_main[i][j]
                val_str = "" if np.isnan(val) else f"{val:+.2f}%"
                row_txt.append(f"{row_label} {col_label}<br>{val_str}")
            hover_main.append(row_txt)

        hover_yearly = []
        z_rounded_yearly = (z_combined_yearly * 100).round(2)
        for i, row_label in enumerate(y_labels):
            val = z_rounded_yearly[i][0]
            val_str = "" if np.isnan(val) else f"{val:+.2f}%"
            hover_yearly.append([f"{row_label} Total<br>{val_str}"])

        n_years = len(y_labels) - 1
        fig = make_subplots(rows=2, cols=2, shared_xaxes=True, shared_yaxes=True,
            horizontal_spacing=0.03, vertical_spacing=0.02, column_widths=[0.85, 0.15],
            row_heights=[n_years/(n_years+1), 1/(n_years+1)]
        )

        fig.add_trace(go.Heatmap(z=(z_combined_main[:-1] * 100).round(2), x=month_names, y=y_labels[:-1], colorscale=colorscale_heatmap, zmid=0, zmin=-scale_main, zmax=scale_main, texttemplate="%{z:+.2f}%", hoverinfo="text", hovertext=hover_main[:-1], showscale=False), row=1, col=1)
        fig.add_trace(go.Heatmap(z=(z_combined_yearly[:-1] * 100).round(2), x=["Yearly"], y=y_labels[:-1], colorscale=colorscale_heatmap, zmid=0, zmin=-scale_yearly, zmax=scale_yearly, texttemplate="%{z:+.2f}%", hoverinfo="text", hovertext=hover_yearly[:-1], showscale=False), row=1, col=2)
        fig.add_trace(go.Heatmap(z=(z_combined_main[-1:] * 100).round(2), x=month_names, y=y_labels[-1:], colorscale=colorscale_heatmap, zmid=0, zmin=-scale_main, zmax=scale_main, texttemplate="%{z:+.2f}%", hoverinfo="text", hovertext=hover_main[-1:], showscale=False), row=2, col=1)
        fig.add_trace(go.Heatmap(z=(z_combined_yearly[-1:] * 100).round(2), x=["Yearly"], y=y_labels[-1:], colorscale=colorscale_heatmap, zmid=0, zmin=-scale_yearly, zmax=scale_yearly, texttemplate="%{z:+.2f}%", hoverinfo="text", hovertext=hover_yearly[-1:], showscale=False), row=2, col=2)

        fig.update_layout(title="Monthly Returns Heatmap (%)", template="plotly_white", height=max(400, (len(y_labels)+1)*30), yaxis=dict(autorange="reversed", type="category"), yaxis3=dict(autorange="reversed", type="category"))
        fig.update_yaxes(showticklabels=False, col=2)
        st.plotly_chart(fig, use_container_width=True, key=f"m_hm_{full_suffix}")

    
    tab_summary, tab_annual, tab_quarterly, tab_monthly, tab_daily = st.tabs(["📋 Summary", "📅 Annual", "📆 Quarterly", "🗓️ Monthly", "📊 Daily"])

    with tab_summary:
        st.subheader(f"{portfolio_name} Seasonal Summary")
        render_seasonal_summary(port_series)
    
    with tab_annual:
        st.subheader(f"{portfolio_name} Annual Returns")
        
        colors = ["#00CC96" if x >= 0 else "#EF553B" for x in annual_ret]
        fig = go.Figure(go.Bar(
            x=annual_ret.index.year,
            y=annual_ret * 100,
            marker_color=colors,
            text=(annual_ret * 100).apply(lambda x: f"{x:+.1f}%"),
            textposition="auto"
        ))
        fig.update_layout(
            title="Annual Returns (%)",
            yaxis_title="Return (%)",
            xaxis_title="Year",
            template="plotly_dark",
            showlegend=False,
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)
        
        df_annual = pd.DataFrame({
            "Year": annual_ret.index.year,
            "Return": annual_ret.values
        }).sort_values("Year", ascending=False)
        
        st.dataframe(
            df_annual.style.format({"Return": "{:+.1%}"}).map(color_return, subset=["Return"]),
            use_container_width=True,
            hide_index=True
        )

    with tab_quarterly:
        qt_tabs = [portfolio_name]
        if comparison_series is not None and not comparison_series.empty: qt_tabs.append("Benchmark (Comparison)")
        if bench_series is not None and not bench_series.empty: qt_tabs.append("Benchmark (Primary)")
            
        q_view_tabs = st.tabs(qt_tabs)
        with q_view_tabs[0]:
            st.subheader(f"{portfolio_name} Quarterly Returns")
            render_quarterly_returns_view(port_series)
        
        if len(qt_tabs) > 1 and "Benchmark (Comparison)" in qt_tabs:
            with q_view_tabs[qt_tabs.index("Benchmark (Comparison)")]:
                st.subheader("Standard Rebalance (Comparison) Quarterly Returns")
                render_quarterly_returns_view(comparison_series, suffix="_comp")
        
        if len(qt_tabs) > 1 and "Benchmark (Primary)" in qt_tabs:
             with q_view_tabs[qt_tabs.index("Benchmark (Primary)")]:
                st.subheader("Primary Benchmark Quarterly Returns")
                render_quarterly_returns_view(bench_series, suffix="_bench")

    with tab_monthly:
        hm_tabs = [portfolio_name]
        if comparison_series is not None and not comparison_series.empty: hm_tabs.append("Benchmark (Comparison)")
        if bench_series is not None and not bench_series.empty: hm_tabs.append("Benchmark (Primary)")
            
        m_view_tabs = st.tabs(hm_tabs)
        with m_view_tabs[0]:
            st.subheader(f"{portfolio_name} Monthly Returns")
            render_monthly_returns_view(port_series)
            
            st.subheader("Monthly Returns List")
            df_monthly_list = monthly_ret.to_frame(name="Return")
            df_monthly_list["Date"] = df_monthly_list.index.strftime("%Y-%m")
            df_monthly_list = df_monthly_list[["Date", "Return"]].sort_index(ascending=False)
            st.dataframe(df_monthly_list.style.format({"Return": "{:+.1%}"}).map(color_return, subset=["Return"]), use_container_width=True, hide_index=True)

        if len(hm_tabs) > 1 and "Benchmark (Comparison)" in hm_tabs:
            with m_view_tabs[hm_tabs.index("Benchmark (Comparison)")]:
                st.subheader("Standard Rebalance (Comparison) Monthly Returns")
                render_monthly_returns_view(comparison_series, suffix="_comp")
                
        if len(hm_tabs) > 1 and "Benchmark (Primary)" in hm_tabs:
             with m_view_tabs[hm_tabs.index("Benchmark (Primary)")]:
                st.subheader("Primary Benchmark Monthly Returns")
                render_monthly_returns_view(bench_series, suffix="_bench")

    with tab_daily:
        st.subheader(f"{portfolio_name} Daily Returns")
        
        c1, c2, c3 = st.columns(3)
        c1.metric("Best Day", f"{daily_ret.max()*100:+.2f}%")
        c2.metric("Worst Day", f"{daily_ret.min()*100:+.2f}%")
        c3.metric("Positive Days", f"{(daily_ret > 0).mean()*100:.1f}%")
        
        fig = go.Figure(go.Histogram(
            x=daily_ret * 100,
            nbinsx=100,
            marker_color="#636EFA"
        ))
        
        fig.update_layout(
            title="Distribution of Daily Returns",
            xaxis_title="Return (%)",
            yaxis_title="Frequency",
            template="plotly_dark",
            height=400,
            bargap=0.1
        )
        st.plotly_chart(fig, use_container_width=True)
        
        st.subheader("Daily Returns List")
        df_daily_list = daily_ret.to_frame(name="Return")
        df_daily_list["Date"] = df_daily_list.index.date
        df_daily_list = df_daily_list[["Date", "Return"]].sort_index(ascending=False)
        
        st.dataframe(
            df_daily_list.style.format({"Return": "{:+.1%}"}).map(color_return, subset=["Return"]),
            use_container_width=True,
            hide_index=True
        )


def render_rebalance_sankey(trades_df, view_freq="Yearly", unique_id=None):
    if trades_df.empty:
        return

    st.subheader("Flow Visualization")
    
    df = trades_df.copy()
    
    # Filter out current incomplete year ONLY if viewing Yearly
    current_year = dt.date.today().year
    if view_freq == "Yearly":
        df = df[df["Date"].dt.year < current_year]
    
    if df.empty:
        st.info(f"No rebalancing data available ({view_freq}).")
        return

    # Create Period column based on view_freq
    if view_freq == "Yearly":
        df["Period"] = df["Date"].dt.year.astype(str)
    elif view_freq == "Quarterly":
        df["Period"] = df["Date"].dt.to_period("Q").astype(str)
    elif view_freq == "Monthly":
        df["Period"] = df["Date"].dt.to_period("M").astype(str)
    elif view_freq == "Per Event":
        df["Period"] = df["Date"].dt.strftime('%Y-%m-%d')
        
    # Period Selection
    periods = sorted(df["Period"].unique(), reverse=True)
    key_suffix = f"_{unique_id}" if unique_id else ""
    selected_period = st.selectbox("Select Period for Flow", periods, index=0, key=f"rebal_period_selector{key_suffix}")
    
    # Filter data for selected period
    df_period = df[df["Period"] == selected_period]
    
    # Calculate Net Flow per ticker for this period
    net_flows = df_period.groupby("Ticker")["Trade Amount"].sum().sort_values()
    
    sources = net_flows[net_flows < 0].abs() # Sold
    targets = net_flows[net_flows > 0]       # Bought
    
    if sources.empty and targets.empty:
        st.info("No rebalancing flow for this year.")
        return

    # Create Nodes
    # Sources -> Rebalancing -> Targets
    
    label_list = []
    color_list = []
    
    # Source Nodes
    source_indices = {}
    for i, (ticker, val) in enumerate(sources.items()):
        label_list.append(f"{ticker} (Sold)")
        color_list.append("#EF553B") # Red
        source_indices[ticker] = i
        
    # Center Node
    center_idx = len(label_list)
    label_list.append("Rebalancing")
    color_list.append("#888888") # Grey
    
    # Target Nodes
    target_indices = {}
    for i, (ticker, val) in enumerate(targets.items()):
        label_list.append(f"{ticker} (Bought)")
        color_list.append("#00CC96") # Green
        target_indices[ticker] = center_idx + 1 + i
        
    # Create Links
    source_links = [] # Indices
    target_links = [] # Indices
    values = []
    
    # Links: Source -> Center
    for ticker, val in sources.items():
        source_links.append(source_indices[ticker])
        target_links.append(center_idx)
        values.append(val)
        
    # Links: Center -> Target
    for ticker, val in targets.items():
        source_links.append(center_idx)
        target_links.append(target_indices[ticker])
        values.append(val)
        
    fig = go.Figure(data=[go.Sankey(
        node = dict(
          pad = 15,
          thickness = 20,
          line = dict(color = "black", width = 0.5),
          label = label_list,
          color = color_list
        ),
        link = dict(
          source = source_links,
          target = target_links,
          value = values,
          color = ["rgba(239, 85, 59, 0.4)"] * len(sources) + ["rgba(0, 204, 150, 0.4)"] * len(targets)
        ))])

    fig.update_layout(title_text=f"Rebalancing Flow {selected_period}", font_size=12, height=500)
    st.plotly_chart(fig, use_container_width=True)

@st.cache_data(show_spinner=False)
def render_portfolio_composition(composition_df, view_freq="Yearly"):
    if composition_df.empty:
        return

    st.subheader("Portfolio Composition")
    
    # Ensure sorted by date
    df = composition_df.sort_values(["Date", "Ticker"])
    
    # Filtering Logic based on View Frequency to avoid summing multiple snapshots in one bar
    if view_freq == "Yearly":
        # Keep only the last snapshot available for each Year
        last_dates = df.groupby(df['Date'].dt.year)['Date'].max()
        df = df[df['Date'].isin(last_dates)]
    elif view_freq == "Quarterly":
        # Keep last snapshot for each Quarter
        last_dates = df.groupby(df['Date'].dt.to_period('Q'))['Date'].max()
        df = df[df['Date'].isin(last_dates)]
    elif view_freq == "Monthly":
        # Keep last snapshot for each Month
        last_dates = df.groupby(df['Date'].dt.to_period('M'))['Date'].max()
        df = df[df['Date'].isin(last_dates)]
    elif view_freq == "Per Event":
        # Keep every unique rebalance snapshot date
        pass

    # Format Date for the axis labeling (categorical)
    df['Date Label'] = df['Date'].dt.strftime('%Y-%m-%d')
    
    # Calculate Total Value per date for hover display
    totals = df.groupby('Date Label')['Value'].sum().rename('Total Value')
    df = df.merge(totals, on='Date Label')

    df = df.sort_values(["Date Label", "Value"], ascending=[True, True])
    
    fig = px.bar(
        df, 
        y="Date Label", 
        x="Value", 
        color="Ticker", 
        title=f"Portfolio Value by Asset (Pre-Rebalance, {view_freq})",
        text_auto="$.2s",
        orientation='h',
        template="plotly_dark",
        custom_data=["Total Value"]
    )
    
    fig.update_traces(
        hovertemplate="<b>%{fullData.name}</b><br>Asset Value: %{x:$,.0f}<br>Total Portfolio: %{customdata[0]:$,.0f}<extra></extra>"
    )
    
    fig.update_layout(
        xaxis_title="Value ($)",
        yaxis_title="Rebalance Date",
        legend_title="Asset",
        height=min(1200, max(400, len(df['Date Label'].unique()) * 30)), # Dynamic height
        yaxis=dict(type='category', categoryorder='category ascending') # Recent at top (Y-axis ascending puts largest/latest at top)
    )
    
    st.plotly_chart(fig, use_container_width=True)

    # --- Data Table View ---
    with st.expander(f"📋 Portfolio Composition Details ({view_freq})", expanded=False):
        # Pivot the data for a clean table view: Dates as rows, Tickers as columns
        table_df = df.pivot(index='Date Label', columns='Ticker', values='Value').fillna(0)
        
        # Add Total Column
        table_df['Total'] = table_df.sum(axis=1)
        
        # Sort by date descending (latest at top)
        table_df = table_df.sort_index(ascending=False)
        
        # Format for display
        st.dataframe(
            table_df.style.format("${:,.0f}"),
            use_container_width=True
        )


def render_rebalancing_analysis(trades_df, pl_by_year, composition_df, tax_method, other_income, filing_status, state_tax_rate, rebalance_freq="Yearly", use_standard_deduction=True, unrealized_pl_df=None, custom_freq="Yearly", unique_id=None):
    if trades_df.empty:
        st.info("No rebalancing events found.")
        return
        
    # Determine default index based on rebalance_freq
    freq_options = ["Yearly", "Quarterly", "Monthly", "Per Event"]
    try:
        default_idx = freq_options.index(rebalance_freq)
    except ValueError:
        if rebalance_freq == "Custom":
            # Try to match the custom frequency (Yearly/Quarterly/Monthly)
            if custom_freq in freq_options:
                default_idx = freq_options.index(custom_freq)
            else:
                default_idx = 3 # Default to "Per Event"
        else:
            default_idx = 0
        
    key_suffix = f"_{unique_id}" if unique_id else ""
    
    # View Frequency Selector
    view_freq = st.selectbox(
        "View Frequency", 
        freq_options, 
        index=default_idx,
        key=f"rebal_view_freq{key_suffix}"
    )

    # Optional "Mag 7 Fund" Grouping
    group_mag7 = st.toggle("Enable Mag 7 Grouping", value=False, key=f"rebal_mag7{key_suffix}", help="Groups AAPL, MSFT, GOOG, AMZN, NVDA, META, TSLA, and AVGO into a single 'Mag 7' fund.")
    
    # Process Composition Data for Mag 7 Grouping
    comp_df_to_plot = composition_df.copy()
    if group_mag7 and not comp_df_to_plot.empty:
        # Define sets
        mag7_standard = ["MSFT", "TSLA", "GOOG", "AAPL", "NVDA", "META", "AMZN"]
        mag7_plus_avgo = mag7_standard + ["AVGO"]
        
        # Helper to categorize tickers
        def get_group(ticker):
            # Check for leverage
            is_lev = "?L=2" in ticker
            
            # Clean base ticker for checking (handle ?L=2 and other suffixes if any)
            base = ticker.split("?")[0]
            
            # Check if it starts with any known root (e.g. GOOGL matches GOOG)
            # We must be careful not to match random things, but for these specific tickers likely safe
            
            # 1. QQQU Check: Mag 7 + AVGO (Leveraged)
            # User said: "When you see it [Mag7+AVGO] with L2 consider the tag 'QQQU'"
            if is_lev:
                # Check root match against Mag 7 + AVGO
                match = False
                if base in mag7_plus_avgo: match = True
                else:
                    for r in mag7_plus_avgo:
                        if base.startswith(r): 
                            match = True
                            break
                if match: return "QQQU"

            # 2. Mag 7 Check: Standard Mag 7 (Unleveraged, NO AVGO)
            # User said: "Mag 7 without AVGO ... without ?L=2 consider it labeled 'Mag 7'"
            else:
                # Check root match against Standard Mag 7 ONLY
                match = False
                if base in mag7_standard: match = True
                else: 
                     for r in mag7_standard:
                        if base.startswith(r):
                            match = True
                            break
                if match: return "Mag 7"
            
            # Default: No Group
            return None

        # Apply grouping
        comp_df_to_plot["Group"] = comp_df_to_plot["Ticker"].apply(get_group)
        
        # Split into grouped and ungrouped
        grouped_rows = comp_df_to_plot[comp_df_to_plot["Group"].notna()]
        ungrouped_rows = comp_df_to_plot[comp_df_to_plot["Group"].isna()].drop(columns=["Group"])
        
        if not grouped_rows.empty:
            # Aggregate by Date AND Group
            grouped_agg = grouped_rows.groupby(["Date", "Group"])["Value"].sum().reset_index()
            grouped_agg = grouped_agg.rename(columns={"Group": "Ticker"})
            
            # Combine back
            comp_df_to_plot = pd.concat([ungrouped_rows, grouped_agg], ignore_index=True).sort_values("Date")
        else:
            # Cleanup if nothing matched
             comp_df_to_plot = comp_df_to_plot.drop(columns=["Group"])
    
    # Aggregate Data based on Frequency
    df_chart = trades_df.copy()
    
    if view_freq == "Yearly":
        # Already have pl_by_year, but let's re-aggregate from trades_df to be consistent
        # Group by Year
        cols_to_agg = ["Realized P&L", "Realized ST P&L", "Realized LT P&L"]
        if "Realized LT (Collectible)" in df_chart.columns:
            cols_to_agg.append("Realized LT (Collectible)")
            
        agg_df = df_chart.groupby("Year")[cols_to_agg].sum().sort_index()
        
        # Merge Unrealized P&L if available
        if unrealized_pl_df is not None and not unrealized_pl_df.empty:
            # Resample to Year End (taking the last value of the year)
            unrealized_yearly = unrealized_pl_df.resample("YE").last()
            unrealized_yearly.index = unrealized_yearly.index.year
            agg_df = agg_df.join(unrealized_yearly[["Unrealized P&L"]], how="outer").fillna(0.0)
            
        x_axis = agg_df.index
        
    elif view_freq == "Quarterly":
        # Group by Year-Quarter
        df_chart["Quarter"] = df_chart["Date"].dt.to_period("Q")
        
        cols_to_agg = ["Realized P&L", "Realized ST P&L", "Realized LT P&L"]
        if "Realized LT (Collectible)" in df_chart.columns:
            cols_to_agg.append("Realized LT (Collectible)")
            
        agg_df = df_chart.groupby("Quarter")[cols_to_agg].sum().sort_index()
        
        # Merge Unrealized P&L
        if unrealized_pl_df is not None and not unrealized_pl_df.empty:
            # Resample to Quarter End
            unrealized_q = unrealized_pl_df.resample("Q").last()
            unrealized_q.index = unrealized_q.index.to_period("Q")
            agg_df = agg_df.join(unrealized_q[["Unrealized P&L"]], how="outer").fillna(0.0)
            
        x_axis = agg_df.index.to_timestamp()
        
    elif view_freq == "Monthly":
        # Group by Year-Month
        df_chart["Month"] = df_chart["Date"].dt.to_period("M")
        
        cols_to_agg = ["Realized P&L", "Realized ST P&L", "Realized LT P&L"]
        if "Realized LT (Collectible)" in df_chart.columns:
            cols_to_agg.append("Realized LT (Collectible)")
            
        agg_df = df_chart.groupby("Month")[cols_to_agg].sum().sort_index()
        
        # Merge Unrealized P&L
        if unrealized_pl_df is not None and not unrealized_pl_df.empty:
            # Resample to Month End (should match index mostly)
            unrealized_m = unrealized_pl_df.resample("M").last()
            unrealized_m.index = unrealized_m.index.to_period("M")
            agg_df = agg_df.join(unrealized_m[["Unrealized P&L"]], how="outer").fillna(0.0)
            
        x_axis = agg_df.index.to_timestamp()
        
    elif view_freq == "Per Event":
        # Use exact dates of each trade event
        cols_to_agg = ["Realized P&L", "Realized ST P&L", "Realized LT P&L"]
        if "Realized LT (Collectible)" in df_chart.columns:
            cols_to_agg.append("Realized LT (Collectible)")
            
        agg_df = df_chart.groupby("Date")[cols_to_agg].sum().sort_index()
        
        # Merge Unrealized P&L (Match to exact event dates)
        if unrealized_pl_df is not None and not unrealized_pl_df.empty:
            # We align unrealized P&L to the exact rebalance dates
            unrealized_aligned = unrealized_pl_df.reindex(agg_df.index, method='ffill').fillna(0.0)
            agg_df = agg_df.join(unrealized_aligned[["Unrealized P&L"]], how="outer").fillna(0.0)
            
        x_axis = agg_df.index
        
    c1, c2 = st.columns([2, 1])
    
    with c1:
        st.subheader(f"Realized P&L ({view_freq})")
        
        # Use simple color coding for total P&L
        colors = ["#00CC96" if x >= 0 else "#EF553B" for x in agg_df["Realized P&L"]]
        
        # Create stacked bar chart for ST/LT split?
        # Or just total P&L?
        # Let's show Total P&L for simplicity, but maybe add hover details
        
        fig = go.Figure()
        
        # Stacked Bar for ST and LT?
        # If we have ST and LT columns (which we should from shadow_backtest)
        if "Realized ST P&L" in agg_df.columns:
            fig.add_trace(go.Bar(
                x=x_axis, 
                y=agg_df["Realized ST P&L"], 
                name="Realized ST (Ordinary)",
                marker_color="#EF553B", # Red/Orange
                hovertemplate="%{y:$,.0f}"
            ))
            fig.add_trace(go.Bar(
                x=x_axis, 
                y=agg_df["Realized LT P&L"], 
                name="Realized LT (Preferential)",
                marker_color="#00CC96", # Green/Teal
                hovertemplate="%{y:$,.0f}"
            ))
            
            if "Realized LT (Collectible)" in agg_df.columns and agg_df["Realized LT (Collectible)"].abs().sum() > 0:
                fig.add_trace(go.Bar(
                    x=x_axis, 
                    y=agg_df["Realized LT (Collectible)"], 
                    name="Realized LT (Collectible)",
                    marker_color="#FFA15A", # Gold/Orange
                    hovertemplate="%{y:$,.0f}"
                ))
            
            # Add Unrealized P&L Trace
            if "Unrealized P&L" in agg_df.columns and agg_df["Unrealized P&L"].abs().sum() > 0:
                 fig.add_trace(go.Bar(
                    x=x_axis, 
                    y=agg_df["Unrealized P&L"], 
                    name="Unrealized P&L (Deferred)",
                    marker_color="#636EFA", # Blue/Purple
                    opacity=0.6, # Slightly transparent to distinguish from realized
                    hovertemplate="%{y:$,.0f}"
                ))
                
            
            fig.update_layout(
                barmode='relative', # Stacked (Relative handles mixed signs better than 'stack')
                title="P&L Composition",
                xaxis_title="Period",
                yaxis_title="Amount ($)",
                legend_title="Type",
                hovermode="x unified"
            )
             
        else:
            # Fallback to simple bar
            fig.add_trace(go.Bar(
                x=x_axis,
                y=agg_df["Realized P&L"],
                marker_color=colors,
                text=agg_df["Realized P&L"].apply(lambda x: f"${x:,.0f}"),
                textposition="auto"
            ))
            
        fig.update_layout(
            yaxis_title="Realized P&L ($)",
            xaxis_title=view_freq[:-2], # Year/Quarter/Month
            template="plotly_dark",
            showlegend=True,
            height=400,
            hovermode="x unified"
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Estimated Tax Owed Chart
        if not pl_by_year.empty:
            st.subheader(f"Estimated Total Tax Owed ({view_freq} - {tax_method})")
            
            # 1. Calculate Annual Federal Tax (Base)
            federal_tax_annual = tax_library.calculate_tax_series_with_carryforward(
                pl_by_year, 
                other_income,
                filing_status,
                method=tax_method,
                use_standard_deduction=use_standard_deduction
            )
            
            # 2. Calculate Annual State Tax (Base)
            state_tax_annual = pd.Series(0.0, index=federal_tax_annual.index)
            loss_cf = 0.0
            total_pl_series = pl_by_year["Realized P&L"] if isinstance(pl_by_year, pd.DataFrame) else pl_by_year
            
            for y, pl in total_pl_series.sort_index().items():
                net = pl - loss_cf
                if net > 0:
                    state_tax_annual[y] = net * state_tax_rate
                    loss_cf = 0.0
                else:
                    loss_cf = abs(net)
            
            total_tax_annual = federal_tax_annual + state_tax_annual
            
            # 3. Allocate to Periods if needed
            if view_freq == "Yearly":
                tax_to_plot = total_tax_annual
                x_axis_tax = tax_to_plot.index
            else:
                # Allocate annual tax to periods based on realized gains
                # We need the period data (agg_df) which we calculated above
                
                # Create a Series to hold allocated tax
                tax_to_plot = pd.Series(0.0, index=agg_df.index)
                
                # Iterate through each year
                for year in total_tax_annual.index:
                    annual_tax = total_tax_annual.get(year, 0.0)
                    if annual_tax <= 0:
                        continue
                        
                    # Get periods for this year
                    if view_freq == "Quarterly":
                        # Filter agg_df for this year (Quarter index)
                        # Period index is like "2021Q1"
                        periods_in_year = [p for p in agg_df.index if p.year == year]
                    elif view_freq == "Monthly":
                        periods_in_year = [p for p in agg_df.index if p.year == year]
                    elif view_freq == "Per Event":
                        periods_in_year = [p for p in agg_df.index if p.year == year]
                    
                    # Calculate total POSITIVE realized P&L for this year from the periods
                    # We only allocate tax to periods that had gains
                    year_gains = 0.0
                    period_gains = {}
                    
                    for p in periods_in_year:
                        gain = agg_df.loc[p, "Realized P&L"]
                        if gain > 0:
                            year_gains += gain
                            period_gains[p] = gain
                        else:
                            period_gains[p] = 0.0
                            
                    # Allocate
                    if year_gains > 0:
                        for p in periods_in_year:
                            if period_gains[p] > 0:
                                allocation_ratio = period_gains[p] / year_gains
                                tax_to_plot[p] = annual_tax * allocation_ratio
                
                x_axis_tax = tax_to_plot.index.astype(str)

            if tax_to_plot.sum() > 0:
                fig_tax = go.Figure(go.Bar(
                    x=x_axis_tax,
                    y=tax_to_plot,
                    marker_color="#EF553B", # Red for taxes
                    texttemplate="%{y:$.2s}",
                    textposition="auto",
                    hovertemplate="%{y:$,.0f}<extra></extra>",
                    name="Estimated Tax"
                ))
                fig_tax.update_layout(
                    yaxis_title="Tax Owed ($)",
                    xaxis_title=view_freq[:-2], # Year/Quarter/Month
                    template="plotly_dark",
                    showlegend=False,
                    height=400,
                    hovermode="x unified"
                )
                st.plotly_chart(fig_tax, use_container_width=True)
                
                total_tax = tax_to_plot.sum()
                st.metric("Total Estimated Tax Owed", f"${total_tax:,.2f}")
            else:
                st.info("No taxable gains realized.")
        
    with c2:
        st.subheader("Total Turnover")
        
        # Total Bought/Sold per ticker
        buys = trades_df[trades_df["Trade Amount"] > 0].groupby("Ticker")["Trade Amount"].sum()
        sells = trades_df[trades_df["Trade Amount"] < 0].groupby("Ticker")["Trade Amount"].sum().abs()
        
        summary = pd.DataFrame({"Bought": buys, "Sold": sells}).fillna(0)
        summary["Net Flow"] = summary["Bought"] - summary["Sold"]
        summary = summary.sort_values("Bought", ascending=False)
        
        st.dataframe(
            summary.style.format("${:,.0f}"),
            use_container_width=True
        )


    # Portfolio Composition
    render_portfolio_composition(comp_df_to_plot, view_freq=view_freq)
        
    # Sankey Diagram
    render_rebalance_sankey(trades_df, view_freq=view_freq, unique_id=unique_id)
    
    with st.expander(f"Rebalancing Details ({view_freq} - Net Flow)", expanded=True):
        current_year = dt.date.today().year
        if view_freq == "Yearly":
            st.caption(f"Positive values indicate Net Buy, Negative values indicate Net Sell. (Excluding {current_year})")
        else:
            st.caption("Positive values indicate Net Buy, Negative values indicate Net Sell.")
        
        df_details = trades_df.copy()
        
        # Filter out current incomplete year ONLY if viewing Yearly
        if view_freq == "Yearly":
            df_details = df_details[df_details["Date"].dt.year < current_year]
        
        if df_details.empty:
            st.info(f"No data available for details ({view_freq}).")
        else:
            # Create Period column
            if view_freq == "Yearly":
                df_details["Period"] = df_details["Date"].dt.year.astype(str)
            elif view_freq == "Quarterly":
                df_details["Period"] = df_details["Date"].dt.to_period("Q").astype(str)
            elif view_freq == "Monthly":
                df_details["Period"] = df_details["Date"].dt.to_period("M").astype(str)
            elif view_freq == "Per Event":
                df_details["Period"] = df_details["Date"].dt.strftime('%Y-%m-%d')

            # Create Pivot Table: Period vs Ticker (Net Flow)
            pivot_df = df_details.pivot_table(
                index="Period", 
                columns="Ticker", 
                values="Trade Amount", 
                aggfunc="sum"
            ).fillna(0)
            
            # Sort index descending (newest first)
            pivot_df = pivot_df.sort_index(ascending=False)
            
            # Add Total column
            pivot_df["Total Net Flow"] = pivot_df.sum(axis=1)
            
            st.dataframe(
                pivot_df.style.format("${:,.0f}").map(color_return),
                use_container_width=True
            )
        
    with st.expander("Detailed Trade Log"):
        display_trades = trades_df.copy()
        display_trades["Date"] = display_trades["Date"].dt.date
        display_trades = display_trades[["Date", "Ticker", "Trade Amount", "Realized P&L"]]
        st.dataframe(
            display_trades.style.format({
                "Trade Amount": "${:,.2f}",
                "Realized P&L": "${:,.2f}"
            }).map(color_return, subset=["Trade Amount", "Realized P&L"]),
            use_container_width=True
        )
@st.cache_data(show_spinner=False)
def render_tax_analysis(pl_by_year, other_income, filing_status, state_tax_rate, tax_method="2024_fixed", use_standard_deduction=True, unrealized_pl_df=None, trades_df=None, pay_tax_cash=False, pay_tax_margin=False):
    """
    Renders the Tax Analysis tab.
    """
    st.markdown("### 🏛️ Tax Analysis")
    
    # 1. Configuration Summary
    with st.expander("Tax Configuration", expanded=False):
        c1, c2, c3 = st.columns(3)
        c1.metric("Filing Status", filing_status)
        c2.metric("Other Income", f"${other_income:,.0f}")
        c3.metric("State Tax Rate", f"{state_tax_rate*100:.1f}%")
        
        c4, c5, c6 = st.columns(3)
        c4.metric("Tax Method", tax_method)
        c5.metric("Std Deduction", "Yes" if use_standard_deduction else "No")
        
        payment_source = "External Cash" if pay_tax_cash else ("Margin Loan" if pay_tax_margin else "None (Gross)")
        c6.metric("Payment Source", payment_source)

    if pl_by_year.empty:
        st.info("No realized P&L to analyze.")
        return

    # 2. Calculate Detailed Tax Series
    # We recalculate here to show the breakdown (Fed vs State)
    fed_tax_series = tax_library.calculate_tax_series_with_carryforward(
        pl_by_year, 
        other_income,
        filing_status,
        method=tax_method,
        use_standard_deduction=use_standard_deduction
    )
    
    # State Tax
    state_tax_series = pd.Series(0.0, index=fed_tax_series.index)
    loss_cf_state = 0.0
    total_pl_series = pl_by_year["Realized P&L"] if isinstance(pl_by_year, pd.DataFrame) else pl_by_year
    
    for y, pl in total_pl_series.sort_index().items():
        net = pl - loss_cf_state
        if net > 0:
            state_tax_series[y] = net * state_tax_rate
            loss_cf_state = 0.0
        else:
            loss_cf_state = abs(net)
            
    total_tax_series = fed_tax_series + state_tax_series
    
    # 3. Summary Metrics
    total_tax_paid = total_tax_series.sum()
    total_realized_pl = total_pl_series.sum()
    effective_tax_rate = total_tax_paid / total_realized_pl if total_realized_pl > 0 else 0.0
    
    m1, m2, m3 = st.columns(3)
    m1.metric("Total Tax Liability", f"${total_tax_paid:,.0f}")
    m2.metric("Total Realized Gains", f"${total_realized_pl:,.0f}")
    m3.metric("Effective Tax Rate", f"{effective_tax_rate*100:.1f}%", help="Total Tax / Total Realized Gains")
    
    # 4. Tax Breakdown Chart
    st.subheader("Tax Liability by Year")
    
    tax_df = pd.DataFrame({
        "Federal Tax": fed_tax_series,
        "State Tax": state_tax_series
    })
    
    fig = px.bar(
        tax_df, 
        x=tax_df.index, 
        y=["Federal Tax", "State Tax"],
        title="Annual Tax Liability",
        labels={"value": "Tax Amount ($)", "index": "Year", "variable": "Type"},
        color_discrete_map={"Federal Tax": "#EF553B", "State Tax": "#636EFA"}
    )
    fig.update_layout(barmode='stack', hovermode="x unified")
    st.plotly_chart(fig, use_container_width=True)
    
    # 5. Detailed Table
    st.subheader("Detailed Tax Log")
    
    detail_df = pl_by_year.copy()
    if isinstance(detail_df, pd.Series):
        detail_df = detail_df.to_frame(name="Realized P&L")
        
    detail_df["Federal Tax"] = fed_tax_series
    detail_df["State Tax"] = state_tax_series
    detail_df["Total Tax"] = total_tax_series
    detail_df["Net P&L"] = detail_df["Realized P&L"] - detail_df["Total Tax"]
    
    st.dataframe(
        detail_df.style.format("${:,.2f}"),
        use_container_width=True
    )


# -----------------------------------------------------------------------------
# Monte Carlo Visualization
# -----------------------------------------------------------------------------
def render_monte_carlo_view(mc_results, unique_id=None):
    """
    Renders the Monte Carlo 'Cone of Uncertainty' chart.
    """
    if not mc_results or mc_results.get("percentiles").empty:
        st.info("Insufficient data for Monte Carlo simulation.")
        return

    df = mc_results["percentiles"]
    m = mc_results["metrics"]
    initial_val = m["initial_val"]
    
    # 1. Metrics Header
    # Row 1: Performance Range
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Median Final Value", f"${m['median_final']:,.0f}", help="50th Percentile Outcome")
    c2.metric("Median CAGR", f"{m['cagr_median']:.1%}", help="Annualized return of the median path")
    
    multiplier = m['median_final'] / m.get('total_invested', initial_val)
    c3.metric("Growth Multiple", f"{multiplier:.1f}x", help="Final Value / Total Invested Capital (Initial + Monthly Adds)")
    c4.metric("CAGR Range (P10-P90)", f"{m['cagr_p10']:.1%} - {m['cagr_p90']:.1%}", help="Pessimistic to Optimistic Annual Return")

    # Row 2: Risk Analysis
    st.markdown("##### ⚠️ Risk Analysis")
    r1, r2, r3, r4 = st.columns(4)
    
    r1.metric("Median Max Drawdown", f"{m['max_dd_median']:.1%}", help="Expected peak-to-trough decline")
    r2.metric("Worst Case DD (P90)", f"{m['max_dd_p90']:.1%}", help="90th Percentile Max Drawdown (Severe Crash)")
    r3.metric("Chance of Loss", f"{m['prob_loss']:.1%}", help="Probability of ending lower than starting value")
    r4.metric("Best Case DD (P10)", f"{m['max_dd_p10']:.1%}", help="10th Percentile Max Drawdown (Mild Correction)")
    
    # 2. Options
    c1, c2 = st.columns([1, 4])
    key_suffix = f"_{unique_id}" if unique_id else ""
    use_log = c1.toggle("Log Scale", help="Use logarithmic scale to see percentage changes better", key=f"mc_log_scale{key_suffix}")
    show_paths = c1.toggle("Show Paths", help="Display 100 random individual simulation paths (Spaghetti Chart)", key=f"mc_show_paths{key_suffix}")
    
    # 3. Plotly Fan Chart
    fig = go.Figure()
    
    x = df.index
    
    # Optional: Spaghetti Paths (Behind the cone)
    if show_paths and "paths" in mc_results:
        paths_df = mc_results["paths"]
        # Sample up to 100 paths (Visual sampling cap for performance)
        n_sample = min(100, paths_df.shape[1])
        if n_sample > 0:
            sample_cols = np.random.choice(paths_df.columns, n_sample, replace=False)
            sample_paths = paths_df[sample_cols]
            
            for col in sample_cols:
                fig.add_trace(go.Scatter(
                    x=x, y=sample_paths[col],
                    mode='lines',
                    line=dict(width=1, color='rgba(100, 255, 255, 0.15)'), # Faint cyan (increased opacity)
                    showlegend=False,
                    hoverinfo='skip' # Don't hover on spaghetti
                ))
    
    # --- Background Fill (Invisible Helper Traces) ---
    # Draw P10 then P90 to create the shaded cone "behind" everything
    fig.add_trace(go.Scatter(
        x=x, y=df["P10"],
        mode='lines',
        line=dict(width=0),
        showlegend=False,
        hoverinfo='skip'
    ))
    fig.add_trace(go.Scatter(
        x=x, y=df["P90"],
        mode='lines',
        line=dict(width=0),
        fill='tonexty',
        fillcolor='rgba(255, 215, 0, 0.15)', # Faint Gold
        showlegend=False,
        hoverinfo='skip'
    ))

    # --- Helper & Pre-calc ---
    def fmt_currency_custom(val):
        if abs(val) >= 1e9: return f"${val/1e9:.2f}B" # 2 decimals for Billions
        if abs(val) >= 1e6: return f"${val/1e6:.1f}M" # 1 decimal for Millions
        if abs(val) >= 1e3: return f"${val/1e3:.0f}k"
        return f"${val:.0f}"
        
    def calculate_nice_ticks(min_v, max_v, is_log=False):
        """Generate human-friendly tick values for axes."""
        if is_log:
            # Powers of 10
            start_exp = np.floor(np.log10(max(1, min_v)))
            end_exp = np.ceil(np.log10(max(1, max_v)))
            ticks = 10 ** np.arange(start_exp, end_exp + 1)
            # Filter to relevant range
            ticks = ticks[(ticks >= min_v * 0.5) & (ticks <= max_v * 1.5)]
            return ticks
        else:
            # Simple linear grid (5-6 ticks)
            return np.linspace(0, max_v, num=6)

    # Pre-calculate custom labels for tooltips
    chart_cols = ["Median", "P10", "P90", "P25", "P75"]
    for col in chart_cols:
        df[f"{col}_chk"] = df[col].apply(fmt_currency_custom)
        
    # Calculate Axis Ticks (Force 'B' suffix)
    y_max_main = df["P90"].max()
    y_min_main = df["P10"].min()
    main_ticks = calculate_nice_ticks(y_min_main, y_max_main, use_log)
    main_tickdata = [fmt_currency_custom(x) for x in main_ticks]

    # --- Visible Lines (Ordered Bottom-to-Top for Z-Order) ---
    
    # 1. P10 (Red) - Bottom
    fig.add_trace(go.Scatter(
        x=x, y=df["P10"],
        mode='lines',
        line=dict(width=1, color='rgba(255, 100, 100, 0.5)', dash='dash'),
        name='P10',
        customdata=df["P10_chk"],
        hovertemplate="<b>P10 (Pessimistic)</b>: %{customdata}<extra></extra>",
        showlegend=True
    ))

    # 2. P25 (Orange)
    fig.add_trace(go.Scatter(
        x=x, y=df["P25"],
        mode='lines',
        line=dict(width=1, color='rgba(255, 165, 0, 0.5)', dash='dot'),
        name='P25',
        customdata=df["P25_chk"],
        hovertemplate="<b>P25 (Mod. Downside)</b>: %{customdata}<extra></extra>"
    ))

    # 3. Median (Cyan)
    fig.add_trace(go.Scatter(
        x=x, y=df["Median"],
        mode='lines',
        line=dict(color='cyan', width=2),
        name='Median',
        customdata=df["Median_chk"],
        hovertemplate="<b>Median</b>: %{customdata}<extra></extra>"
    ))

    # 4. P75 (Silver)
    fig.add_trace(go.Scatter(
        x=x, y=df["P75"],
        mode='lines',
        line=dict(width=1, color='silver', dash='dot'),
        name='P75',
        customdata=df["P75_chk"],
        hovertemplate="<b>P75 (Mod. Upside)</b>: %{customdata}<extra></extra>"
    ))

    # 5. P90 (Gold) - Top
    fig.add_trace(go.Scatter(
        x=x, y=df["P90"],
        mode='lines',
        line=dict(width=1, color='#FFD700', dash='dash'), 
        name='P90',
        customdata=df["P90_chk"],
        hovertemplate="<b>P90 (Optimistic)</b>: %{customdata}<extra></extra>"
    ))
    
    fig.update_layout(
        title=f"Monte Carlo Simulation (10 Years)",
        yaxis_title="Portfolio Value ($)",
        xaxis_title="Date",
        hovermode="x unified", # Force sort by Value (descending)
        height=600,
        yaxis=dict(
            type='log' if use_log else 'linear', 
            tickformat='$.2s', # Compact currency format
            gridcolor='rgba(128,128,128,0.2)'
        ),
        xaxis=dict(gridcolor='rgba(128,128,128,0.2)'),
        template="plotly_dark",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    st.caption("""
    **Methodology:** Historical Bootstrap using your strategy's daily volatility.
    *   **Median Path:** The central outcome (50% probability).
    *   **Outer Cone (Shaded):** The broad range of probable outcomes (**10th** to **90th** percentile).
    *   **Inner Lines (Dashed):** The "likely" zone (**25th** to **75th** percentile), capturing the middle 50% of scenarios.
    *   **Visualization:** For browser performance, we display a random sample of **up to 100 paths** (spaghetti), but metrics use all simulated scenarios.
    *   **Assumptions:** Projection based on the 'Start Value' and 'Monthly Add' configured above. Reinvests all dividends/returns.
    """)
    
    # 4. Distribution Chart (Histogram) (Clipped to P95 for better focus)
    st.markdown("##### 📊 Distribution of Outcomes (Year 10)")
    path_finals = mc_results["paths"].iloc[-1]
    
    # Clip outliers for visualization (Show up to P95)
    p95_val = np.percentile(path_finals, 95)
    
    # --- Manual Binning for Robust Scale Handling ---
    if use_log:
        # Log Mode: Bin in log-space, then map back to dollars
        safe_data = path_finals[path_finals > 0] 
        log_data = np.log10(safe_data)
        counts, bin_edges_log = np.histogram(log_data, bins=100)
        
        # Calculate widths in linear domain (critical for go.Bar on log axis)
        bin_edges = 10**bin_edges_log
        widths = np.diff(bin_edges)
        ctrs = (bin_edges[:-1] + bin_edges[1:]) / 2
        
        x_plot = ctrs
        y_plot = counts
        bar_widths = widths
        
        xaxis_config = dict(
            tickformat='$.2s', 
            gridcolor='rgba(128,128,128,0.2)',
            type='log'
        )
    else:
        # Linear Mode: Bin the clipped data
        data_clipped = path_finals[path_finals <= p95_val * 1.05]
        data_clipped = data_clipped[data_clipped >= 0] 
        
        counts, bin_edges = np.histogram(data_clipped, bins=100)
        widths = np.diff(bin_edges)
        x_plot = (bin_edges[:-1] + bin_edges[1:]) / 2
        y_plot = counts
        bar_widths = widths
        
        xaxis_config = dict(
            tickformat='$.2s', 
            gridcolor='rgba(128,128,128,0.2)',
            type='linear',
            range=[0, p95_val * 1.05] 
        )
    
    # Calculate Manual Ticks for Histogram (to fix 'G')
    hist_min = np.min(x_plot) if use_log else 0
    hist_max = np.max(x_plot) if use_log else p95_val * 1.05
    hist_ticks = calculate_nice_ticks(hist_min, hist_max, use_log)
    hist_tickdata = [fmt_currency_custom(x) for x in hist_ticks]
    
    # Enable manual ticks
    xaxis_config["tickmode"] = "array"
    xaxis_config["tickvals"] = hist_ticks
    xaxis_config["ticktext"] = hist_tickdata

    fig_hist = go.Figure()
    
    # Render as Explicit Bar Chart (for Separator Lines)
    hist_custom_labels = [fmt_currency_custom(v) for v in x_plot]
    
    fig_hist.add_trace(go.Bar(
        x=x_plot, y=y_plot,
        width=bar_widths, # Explicit widths for log scale correctness
        marker_color='rgba(0, 150, 150, 0.6)', # Darker Cyan Fill
        marker_line_color='rgba(200, 255, 255, 0.8)', # Bright separation lines
        marker_line_width=1,
        name='Frequency',
        customdata=hist_custom_labels,
        hovertemplate="<b>Value</b>: %{customdata}<br><b>Count</b>: %{y}<extra></extra>"
    ))
    
    # Add Percentile Lines
    df_pct = mc_results["percentiles"]
    
    # Stagger labels to avoid overlap (3 tiers)
    annotations = [
        (m['median_final'], 'cyan', 'Median', 1.20),      # Top Tier
        (df_pct['P25'].iloc[-1], 'orange', 'P25', 1.12),  # Mid Tier
        (df_pct['P75'].iloc[-1], 'silver', 'P75', 1.12),  # Mid Tier
        (df_pct['P10'].iloc[-1], 'red', 'P10', 1.05),     # Low Tier
        (df_pct['P90'].iloc[-1], '#FFD700', 'P90', 1.05)  # Low Tier
    ]
    
    # Calculate max frequency for line height scaling
    max_freq = np.max(y_plot) if len(y_plot) > 0 else 1
    
    for val, color, label, y_pos in annotations:
        # Use dot for P25/P75 to match main chart, dash for others
        dash_style = "dot" if "P25" in label or "P75" in label else "dash"
        
        # 1. Interactive Line (Trace) for Tooltip
        fmt_val = fmt_currency_custom(val)
        fig_hist.add_trace(go.Scatter(
            x=[val, val], 
            y=[0, max_freq], # Draw line up to the peak frequency
            mode='lines',
            line=dict(color=color, width=2, dash=dash_style),
            name=label,
            hovertemplate=f"<b>{label}</b>: {fmt_val}<extra></extra>", # Use pre-formatted string
            showlegend=False
        ))
        
        # 2. Text Label (Annotation)
        fig_hist.add_annotation(
            x=val, y=y_pos, yref="paper", # Staggered positions
            text=f"<b>{label}</b>", 
            showarrow=False,
            font=dict(color=color, size=11),
            yshift=0
        )

    fig_hist.update_layout(
        height=350,
        xaxis_title="Final Portfolio Value ($)",
        yaxis_title="Frequency (Scenarios)",
        template="plotly_dark",
        bargap=0.1,
        xaxis=xaxis_config,
        yaxis=dict(gridcolor='rgba(128,128,128,0.2)'),
        margin=dict(t=60), # More space for 3-tier annotations
        hovermode="x" # Snap to x-axis (easier to hit lines)
    )
    
    st.plotly_chart(fig_hist, use_container_width=True)



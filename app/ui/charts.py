import plotly.graph_objects as go
import plotly.express as px
import streamlit as st
import pandas as pd
import numpy as np
import datetime as dt
from app.core import tax_library
import os
from app.common.utils import color_return

def render_classic_chart(port, equity, loan, equity_pct, usage_pct, series_opts, log_scale, bench_series=None, comparison_series=None):
    fig = go.Figure()
    TRACES = {
        "Portfolio":       (port.index, port, {"width":2}, "$%{y:,.0f}"),
        "Equity":          (equity.index, equity, {"dash":"dot"}, "$%{y:,.0f}"),
        "Loan":            (loan.index, loan, {"dash":"dot","width":1,"color":"lime"}, "$%{y:,.0f}"),
        "Margin usage %":  (usage_pct.index, usage_pct*100, {"width":2,"color":"yellow"}, "%{y:.2f}%"),
        "Equity %":        (equity_pct.index, equity_pct*100, {"dash":"dash"}, "%{y:.2f}%"),
    }
    
    for key in series_opts:
        if key in TRACES:
            x, y, line, fmt = TRACES[key]
            fig.add_scatter(
                x=x, y=y, name=key,
                line=line,
                hovertemplate=fmt+"<extra></extra>",
                yaxis="y2" if "%" in key else "y"
            )
            
    # Add Benchmark Trace if available (Moved to end for Z-order visibility)
    if bench_series is not None:
         bench_name = bench_series.name if hasattr(bench_series, 'name') and bench_series.name else "Benchmark (Gross)"
         fig.add_scatter(
             x=bench_series.index, y=bench_series, 
             name=bench_name,
             line=dict(color="#FFD700", width=2, dash="dash"),
             hovertemplate="$%{y:,.0f}<extra></extra>"
         )
         
    # Add Comparison Trace if available
    if comparison_series is not None:
         comp_name = comparison_series.name if hasattr(comparison_series, 'name') and comparison_series.name else "Standard Rebalance"
         fig.add_scatter(
             x=comparison_series.index, y=comparison_series, 
             name=comp_name,
             line=dict(color="#00FFFF", width=2, dash="dot"), # Cyan Dot
             hovertemplate="$%{y:,.0f}<extra></extra>"
         )
         
    fig.add_hline(y=100, yref="y2", line={"dash":"dot"},
                  annotation_text="Margin call", annotation_position="top right")
    fig.update_layout(
        template="plotly_white",
        hovermode="x unified",
        xaxis=dict(
            showgrid=True, gridcolor="rgba(0,0,0,0.05)",
            showspikes=True, spikemode="across", spikesnap="cursor",
            spikecolor="rgba(0,0,0,0.3)", spikethickness=1
        ),
        yaxis=dict(
            title="Portfolio value", showgrid=True,
            gridcolor="rgba(0,0,0,0.05)",
            type="log" if log_scale else "linear", rangemode="tozero"
        ),
        yaxis2=dict(
            overlaying="y", side="right",
            title="% of portfolio / allowance", rangemode="tozero"
        ),
        legend=dict(orientation="h", yanchor="top", y=-0.25,
                    xanchor="center", x=0.5),
        margin=dict(t=48, b=100, l=60, r=60)
    )
    st.plotly_chart(fig, use_container_width=True)

def render_dashboard_view(port, equity, loan, equity_pct, usage_pct, maint_pct, stats, log_opts, bench_series=None, comparison_series=None):
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
    leveraged_mult = 1 / (st.session_state.equity_init / 100) if st.session_state.equity_init < 100 else 1
    leveraged_port = port * leveraged_mult
    
    fig1.add_trace(go.Scatter(
        x=port.index, y=leveraged_port,
        name=f"Margin Portfolio ({leveraged_mult:.1f}x Leveraged)",
        line=dict(color="#4A90E2", width=2)
    ))
    fig1.add_trace(go.Scatter(
        x=port.index, y=port,
        name="Margin Portfolio (Unleveraged)",
        line=dict(color="#1DB954", width=2)
    ))

    # Add Benchmark Trace if available (Moved to end)
    if bench_series is not None:
         bench_name = bench_series.name if hasattr(bench_series, 'name') and bench_series.name else "Benchmark (Gross)"
         fig1.add_trace(go.Scatter(
             x=bench_series.index, y=bench_series,
             name=bench_name,
             line=dict(color="#FFD700", width=2, dash="dash")
         ))
         
    # Add Comparison Trace if available
    if comparison_series is not None:
         comp_name = comparison_series.name if hasattr(comparison_series, 'name') and comparison_series.name else "Standard Rebalance"
         fig1.add_trace(go.Scatter(
             x=comparison_series.index, y=comparison_series,
             name=comp_name,
             line=dict(color="#00FFFF", width=2, dash="dot")
         ))
    
    fig1.update_layout(
        **dark_theme,
        height=400,
        xaxis_title="Month",
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
        target_leverage = 1 / (st.session_state.equity_init / 100) if st.session_state.equity_init < 100 else 1
        max_allowed = 1 / (1 - maint_pct)
        
        fig2.add_trace(go.Scatter(
            x=port.index, y=current_leverage,
            name="Current Leverage",
            line=dict(color="#1DB954", width=2)
        ))
        fig2.add_trace(go.Scatter(
            x=port.index, y=[target_leverage]*len(port),
            name="Target Leverage",
            line=dict(color="#FFD700", width=2, dash="dot")
        ))
        fig2.add_trace(go.Scatter(
            x=port.index, y=[max_allowed]*len(port),
            name="Max Allowed",
            line=dict(color="#FF6B6B", width=2, dash="dash")
        ))
        
        fig2.update_layout(
            **dark_theme,
            height=350,
            xaxis_title="Date",
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
            fillcolor="rgba(255,107,107,0.3)"
        ))
        
        # Portfolio value line
        fig3.add_trace(go.Scatter(
            x=port.index, y=port,
            name="Portfolio Value",
            line=dict(color="#4A90E2", width=2),
            yaxis="y"
        ))
        
        # Net liquidating value
        fig3.add_trace(go.Scatter(
            x=equity.index, y=equity,
            name="Net Liquidating Value",
            line=dict(color="#1DB954", width=2),
            yaxis="y"
        ))
        
        # Monthly interest on secondary axis
        monthly_interest = loan * (st.session_state.rate_annual / 100 / 12)
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
    
    # Create subplots - main chart and optionally volume
    from plotly.subplots import make_subplots
    
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

def render_returns_analysis(port_series, bench_series=None, comparison_series=None):
    daily_ret = port_series.pct_change().dropna()
    monthly_ret = port_series.resample("ME").last().pct_change().dropna()
    quarterly_ret = port_series.resample("QE").last().pct_change().dropna()
    annual_ret = port_series.resample("YE").last().pct_change().dropna()

    # --- HEATMAP HELPERS ---
    def render_quarterly_returns_view(series, suffix=""):
        if series.empty: return
        quarterly_ret = series.resample("QE").last().pct_change().dropna()
        annual_ret = series.resample("YE").last().pct_change().dropna()
        
        q_ret = quarterly_ret.to_frame(name="Return")
        q_ret["Year"] = q_ret.index.year
        q_ret["Quarter"] = q_ret.index.quarter
        q_ret["Quarter Name"] = "Q" + q_ret["Quarter"].astype(str)
        
        pivot = q_ret.pivot(index="Year", columns="Quarter", values="Return")
        for i in range(1, 5):
            if i not in pivot.columns: pivot[i] = float("nan")
        pivot = pivot.sort_index(ascending=True)
        
        quarter_names = ["Q1", "Q2", "Q3", "Q4"]
        from plotly.subplots import make_subplots
        
        # Averages & Yearly
        quarterly_avgs = pivot.mean()
        z_data = pivot.values
        z_avgs = quarterly_avgs.values.reshape(1, -1)
        z_combined_main = np.concatenate([z_data, z_avgs], axis=0)
        
        years = pivot.index
        yearly_col = []
        for y in years:
            val = annual_ret[annual_ret.index.year == y]
            yearly_col.append(val.values[0] if not val.empty else float("nan"))
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
        z_rounded_main = (z_combined_main * 100).round(1)
        for i, row_label in enumerate(y_labels):
            row_txt = []
            for j, col_label in enumerate(quarter_names):
                val = z_rounded_main[i][j]
                if np.isnan(val): row_txt.append("")
                elif row_label == "Average": row_txt.append(f"Average<br>{col_label}: {val:+.1f}%")
                else:
                    dr = date_map.get((int(row_label), j+1), "") if row_label.isdigit() else ""
                    dr_str = f"<br>{dr}" if dr else ""
                    row_txt.append(f"Year: {row_label}<br>{col_label}: {val:+.1f}%{dr_str}")
            hover_main.append(row_txt)

        hover_yearly = []
        z_rounded_yearly = (z_combined_yearly * 100).round(1)
        for i, row_label in enumerate(y_labels):
            val = z_rounded_yearly[i][0]
            val_str = "" if np.isnan(val) else f"{val:+.1f}%"
            if row_label == "Average": hover_yearly.append([f"Average Annual<br>{val_str}"])
            else: hover_yearly.append([f"Year: {row_label}<br>Annual: {val_str}"])

        # Plot
        n_years = len(y_labels) - 1
        fig = make_subplots(rows=2, cols=2, shared_xaxes=True, shared_yaxes=True,
            horizontal_spacing=0.03, vertical_spacing=0.02, column_widths=[0.85, 0.15],
            row_heights=[n_years/(n_years+1), 1/(n_years+1)]
        )
        
        # Traces
        # Traces
        # Traces
        fig.add_trace(go.Heatmap(z=(z_combined_main[:-1] * 100).round(1), x=quarter_names, y=y_labels[:-1], colorscale=colorscale_heatmap, zmid=0, zmin=-scale_main, zmax=scale_main, texttemplate="%{z:+.1f}%", hoverinfo="text", hovertext=hover_main[:-1], showscale=False), row=1, col=1)
        fig.add_trace(go.Heatmap(z=(z_combined_yearly[:-1] * 100).round(1), x=["Yearly"], y=y_labels[:-1], colorscale=colorscale_heatmap, zmid=0, zmin=-scale_yearly, zmax=scale_yearly, texttemplate="%{z:+.1f}%", hoverinfo="text", hovertext=hover_yearly[:-1], showscale=False), row=1, col=2)
        fig.add_trace(go.Heatmap(z=(z_combined_main[-1:] * 100).round(1), x=quarter_names, y=y_labels[-1:], colorscale=colorscale_heatmap, zmid=0, zmin=-scale_main, zmax=scale_main, texttemplate="%{z:+.1f}%", hoverinfo="text", hovertext=hover_main[-1:], showscale=False), row=2, col=1)
        fig.add_trace(go.Heatmap(z=(z_combined_yearly[-1:] * 100).round(1), x=["Yearly"], y=y_labels[-1:], colorscale=colorscale_heatmap, zmid=0, zmin=-scale_yearly, zmax=scale_yearly, texttemplate="%{z:+.1f}%", hoverinfo="text", hovertext=hover_yearly[-1:], showscale=False), row=2, col=2)

        fig.update_layout(title="Quarterly Returns Heatmap (%)", template="plotly_white", height=max(400, (len(y_labels)+1)*30), yaxis=dict(autorange="reversed", type="category"), yaxis3=dict(autorange="reversed", type="category"))
        fig.update_yaxes(showticklabels=False, col=2)
        st.plotly_chart(fig, use_container_width=True, key=f"q_hm_{suffix}")
        
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
        if series.empty: return
        m_ret = series.resample("ME").last().pct_change().dropna()
        a_ret = series.resample("YE").last().pct_change().dropna()
        
        df_monthly = m_ret.to_frame(name="Return")
        df_monthly["Year"] = df_monthly.index.year
        df_monthly["Month"] = df_monthly.index.month
        
        pivot = df_monthly.pivot(index="Year", columns="Month", values="Return")
        for i in range(1, 13):
            if i not in pivot.columns: pivot[i] = float("nan")
        pivot = pivot.sort_index(ascending=True)
        month_names = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
        
        from plotly.subplots import make_subplots
        monthly_avgs = pivot.mean()
        z_data = pivot.values
        z_combined_main = np.concatenate([z_data, monthly_avgs.values.reshape(1, -1)], axis=0)
        
        years = pivot.index
        yearly_col = []
        for y in years:
            val = a_ret[a_ret.index.year == y]
            yearly_col.append(val.values[0] if not val.empty else float("nan"))
        yearly_avg = np.nanmean(yearly_col) if len(yearly_col) > 0 else float("nan")
        z_combined_yearly = np.array(yearly_col + [yearly_avg]).reshape(-1, 1)

        y_labels = [str(y) for y in pivot.index] + ["Average"]
        
        std_dev_main = np.nanstd(z_combined_main * 100)
        scale_main = 2 * std_dev_main if not np.isnan(std_dev_main) else 10
        std_dev_yearly = np.nanstd(z_combined_yearly * 100)
        scale_yearly = 1.0 * std_dev_yearly if not np.isnan(std_dev_yearly) else 10
        colorscale_heatmap = [[0, '#E53935'], [0.5, '#FFFFFF'], [1, '#43A047']]
        
        hover_main = []
        z_rounded_main = (z_combined_main * 100).round(1)
        for i, row_label in enumerate(y_labels):
            row_txt = []
            for j, col_label in enumerate(month_names):
                val = z_rounded_main[i][j]
                val_str = "" if np.isnan(val) else f"{val:+.1f}%"
                row_txt.append(f"{row_label} {col_label}<br>{val_str}")
            hover_main.append(row_txt)
            
        hover_yearly = []
        z_rounded_yearly = (z_combined_yearly * 100).round(1)
        for i, row_label in enumerate(y_labels):
            val = z_rounded_yearly[i][0]
            val_str = "" if np.isnan(val) else f"{val:+.1f}%"
            hover_yearly.append([f"{row_label} Total<br>{val_str}"])

        n_years = len(y_labels) - 1
        fig = make_subplots(rows=2, cols=2, shared_xaxes=True, shared_yaxes=True,
            horizontal_spacing=0.03, vertical_spacing=0.02, column_widths=[0.85, 0.15],
            row_heights=[n_years/(n_years+1), 1/(n_years+1)]
        )
        
        fig.add_trace(go.Heatmap(z=(z_combined_main[:-1] * 100).round(1), x=month_names, y=y_labels[:-1], colorscale=colorscale_heatmap, zmid=0, zmin=-scale_main, zmax=scale_main, texttemplate="%{z:+.1f}%", hoverinfo="text", hovertext=hover_main[:-1], showscale=False), row=1, col=1)
        fig.add_trace(go.Heatmap(z=(z_combined_yearly[:-1] * 100).round(1), x=["Yearly"], y=y_labels[:-1], colorscale=colorscale_heatmap, zmid=0, zmin=-scale_yearly, zmax=scale_yearly, texttemplate="%{z:+.1f}%", hoverinfo="text", hovertext=hover_yearly[:-1], showscale=False), row=1, col=2)
        fig.add_trace(go.Heatmap(z=(z_combined_main[-1:] * 100).round(1), x=month_names, y=y_labels[-1:], colorscale=colorscale_heatmap, zmid=0, zmin=-scale_main, zmax=scale_main, texttemplate="%{z:+.1f}%", hoverinfo="text", hovertext=hover_main[-1:], showscale=False), row=2, col=1)
        fig.add_trace(go.Heatmap(z=(z_combined_yearly[-1:] * 100).round(1), x=["Yearly"], y=y_labels[-1:], colorscale=colorscale_heatmap, zmid=0, zmin=-scale_yearly, zmax=scale_yearly, texttemplate="%{z:+.1f}%", hoverinfo="text", hovertext=hover_yearly[-1:], showscale=False), row=2, col=2)

        fig.update_layout(title="Monthly Returns Heatmap (%)", template="plotly_white", height=max(400, (len(y_labels)+1)*30), yaxis=dict(autorange="reversed", type="category"), yaxis3=dict(autorange="reversed", type="category"))
        fig.update_yaxes(showticklabels=False, col=2)
        st.plotly_chart(fig, use_container_width=True, key=f"m_hm_{suffix}")

    
    tab_annual, tab_quarterly, tab_monthly, tab_daily = st.tabs(["📅 Annual", "📆 Quarterly", "🗓️ Monthly", "📊 Daily"])
    
    with tab_annual:
        st.subheader("Annual Returns")
        
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
        qt_tabs = ["Strategy"]
        if comparison_series is not None and not comparison_series.empty: qt_tabs.append("Benchmark (Comparison)")
        if bench_series is not None and not bench_series.empty: qt_tabs.append("Benchmark (Primary)")
            
        q_view_tabs = st.tabs(qt_tabs)
        with q_view_tabs[0]:
            st.subheader("Strategy Quarterly Returns")
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
        hm_tabs = ["Strategy"]
        if comparison_series is not None and not comparison_series.empty: hm_tabs.append("Benchmark (Comparison)")
        if bench_series is not None and not bench_series.empty: hm_tabs.append("Benchmark (Primary)")
            
        m_view_tabs = st.tabs(hm_tabs)
        with m_view_tabs[0]:
            st.subheader("Strategy Monthly Returns")
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
        st.subheader("Daily Returns")
        
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
def render_rebalance_sankey(trades_df, view_freq="Yearly"):
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
    selected_period = st.selectbox("Select Period for Flow", periods, index=0, key="rebal_period_selector")
    
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
    
    import plotly.express as px
    
    # Sort by Value (Smallest to Largest) for the stack order
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


def render_rebalancing_analysis(trades_df, pl_by_year, composition_df, tax_method, other_income, filing_status, state_tax_rate, rebalance_freq="Yearly", use_standard_deduction=True, unrealized_pl_df=None, custom_freq="Yearly"):
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
        
    # View Frequency Selector
    view_freq = st.selectbox(
        "View Frequency", 
        freq_options, 
        index=default_idx,
        key="rebal_view_freq"
    )

    # Optional "Mag 7 Fund" Grouping
    group_mag7 = st.toggle("Enable Mag 7 Grouping", value=False, help="Groups AAPL, MSFT, GOOG, AMZN, NVDA, META, TSLA, and AVGO into a single 'Mag 7' fund.")
    
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
    render_rebalance_sankey(trades_df, view_freq=view_freq)
    
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



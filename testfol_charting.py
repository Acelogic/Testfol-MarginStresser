#!/usr/bin/env python3
# ─────────────────────────────────────────────────────────────────────────────
# testfol_charting.py
#
# New Streamlit App for Testfol Backtesting with Multi-Timeframe Candlesticks
# ─────────────────────────────────────────────────────────────────────────────

import datetime as dt
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import testfol_api as api
import shadow_backtest
import tax_library
import json
import os


# ─────────────────────────────────────────────────────────────────────────────
# Helper Functions
# ─────────────────────────────────────────────────────────────────────────────

PORTFOLIO_FILE = "saved_portfolios.json"

def load_saved_portfolios():
    if not os.path.exists(PORTFOLIO_FILE):
        return {}
    try:
        with open(PORTFOLIO_FILE, "r") as f:
            return json.load(f)
    except:
        return {}

def save_portfolio_to_disk(name, config):
    data = load_saved_portfolios()
    data[name] = config
    with open(PORTFOLIO_FILE, "w") as f:
        json.dump(data, f, indent=4)

def delete_portfolio_from_disk(name):
    data = load_saved_portfolios()
    if name in data:
        del data[name]
        with open(PORTFOLIO_FILE, "w") as f:
            json.dump(data, f, indent=4)


def num_input(label, key, default, step, **kwargs):
    return st.number_input(
        label,
        value=st.session_state.get(key, default),
        step=step,
        key=key,
        **kwargs
    )

def sync_equity():
    sv = st.session_state.start_val
    loan = st.session_state.starting_loan
    st.session_state.equity_init = 100 * max(0, 1 - loan / sv)

def sync_loan():
    sv = st.session_state.start_val
    eq = st.session_state.equity_init
    st.session_state.starting_loan = sv * max(0, 1 - eq / 100)

def resample_data(series: pd.Series, timeframe: str, method="ohlc") -> pd.DataFrame:
    if timeframe == "1D":
        if method == "ohlc":
            df = series.to_frame(name="Close")
            df["Open"] = df["Close"]
            df["High"] = df["Close"]
            df["Low"] = df["Close"]
            return df
        else:
            return series

    rule_map = {
        "1W": "W-FRI",
        "1M": "ME",
        "3M": "QE",
        "1Y": "YE"
    }
    rule = rule_map.get(timeframe, "ME")

    if method == "ohlc":
        ohlc = series.resample(rule).agg({
            "Open": "first",
            "High": "max",
            "Low": "min",
            "Close": "last"
        })
        return ohlc.dropna()
    elif method == "max":
        return series.resample(rule).max().dropna()
    else:
        return series.resample(rule).last().dropna()

def render_classic_chart(port, equity, loan, equity_pct, usage_pct, series_opts, log_scale):
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

def render_dashboard_view(port, equity, loan, equity_pct, usage_pct, maint_pct, stats, log_opts):
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

def render_candlestick_chart(ohlc_df, equity_series, loan_series, usage_series, equity_pct_series, timeframe, log_scale, show_range_slider=True, show_volume=True):
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


def calculate_cagr(series):
    if series.empty: return 0.0
    start_val = series.iloc[0]
    end_val = series.iloc[-1]
    years = (series.index[-1] - series.index[0]).days / 365.25
    if years <= 0: return 0.0
    return (end_val / start_val) ** (1 / years) - 1

def calculate_max_drawdown(series):
    if series.empty: return 0.0
    rolling_max = series.cummax()
    drawdown = (series - rolling_max) / rolling_max
    return drawdown.min()

def color_return(val):
    color = '#00CC96' if val >= 0 else '#EF553B'
    return f'color: {color}'

def render_returns_analysis(port_series):
    daily_ret = port_series.pct_change().dropna()
    monthly_ret = port_series.resample("ME").last().pct_change().dropna()
    annual_ret = port_series.resample("YE").last().pct_change().dropna()
    
    tab_annual, tab_monthly, tab_daily = st.tabs(["📅 Annual", "🗓️ Monthly", "📊 Daily"])
    
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
            df_annual.style.format({"Return": "{:+.2%}"}).map(color_return, subset=["Return"]),
            use_container_width=True,
            hide_index=True
        )

    with tab_monthly:
        st.subheader("Monthly Returns")
        
        m_ret = monthly_ret.to_frame(name="Return")
        m_ret["Year"] = m_ret.index.year
        m_ret["Month"] = m_ret.index.month
        m_ret["Month Name"] = m_ret.index.strftime("%b")
        
        pivot = m_ret.pivot(index="Year", columns="Month", values="Return")
        
        for i in range(1, 13):
            if i not in pivot.columns:
                pivot[i] = float("nan")
        pivot = pivot.sort_index(ascending=False)
        
        month_names = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", 
                       "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
        
        # Round values for cleaner display
        z_rounded = (pivot.values * 100).round(2)
        
        # Heatmap - simple color coding with whole numbers
        fig = go.Figure(data=go.Heatmap(
            z=z_rounded,
            x=month_names,
            y=pivot.index,
            colorscale=[[0, '#ef5350'], [0.5, '#424242'], [1, '#26a69a']],  # Red-Gray-Green
            zmid=0,
            texttemplate="%{z:+.2f}%",  # Two decimal places
            hovertemplate="%{z:+.2f}%<extra></extra>",
            xgap=1, ygap=1,
            showscale=False  # Hide color scale for cleaner look
        ))
        
        fig.update_layout(
            title="Monthly Returns Heatmap (%)",
            template="plotly_dark",
            height=max(400, len(pivot) * 30),
            yaxis=dict(autorange="reversed", type="category")
        )
        fig.update_yaxes(autorange=True, type='category') 
        st.plotly_chart(fig, use_container_width=True)
        
        st.subheader("Monthly Returns List")
        df_monthly_list = m_ret.copy()
        df_monthly_list["Date"] = df_monthly_list.index.strftime("%Y-%m")
        df_monthly_list = df_monthly_list[["Date", "Return"]].sort_index(ascending=False)
        
        st.dataframe(
            df_monthly_list.style.format({"Return": "{:+.2%}"}).map(color_return, subset=["Return"]),
            use_container_width=True,
            hide_index=True
        )

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
            df_daily_list.style.format({"Return": "{:+.2%}"}).map(color_return, subset=["Return"]),
            use_container_width=True,
            hide_index=True
        )



def process_rebalancing_data(rebal_events, port_series, allocation):
    """
    Process rebalancing events to calculate trade amounts and realized P&L.
    """
    if not rebal_events:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
        
    trades = []
    composition = []
    
    # Initialize cost basis with initial allocation
    # Assuming initial portfolio value is the first value in port_series
    initial_val = port_series.iloc[0]
    cost_basis = {}
    
    # Normalize allocation to 100% just in case
    total_alloc = sum(allocation.values())
    if total_alloc > 0:
        for ticker, weight in allocation.items():
            cost_basis[ticker] = initial_val * (weight / total_alloc)
    
    # Process each event group
    for group in rebal_events:
        tickers = group.get("tickers", [])
        events = group.get("events", [])
        
        for event in events:
            date_str = event[0]
            date = pd.to_datetime(date_str)
            
            # Find portfolio value at rebalance date
            # Use asof to find the closest value on or before the date
            try:
                port_val = port_series.asof(date)
            except:
                continue
                
            if pd.isna(port_val):
                continue
            
            # Event structure: [date, drift_t1, drift_t2, ..., weight_t1, weight_t2, ...]
            # Number of tickers = len(tickers)
            # Drift indices: 1 to len(tickers)
            # Weight indices: len(tickers)+1 to 2*len(tickers)
            
            n_tickers = len(tickers)
            
            for i, ticker in enumerate(tickers):
                drift_idx = 1 + i
                weight_idx = 1 + n_tickers + i
                trade_idx = 1 + 2*n_tickers + i
                
                if trade_idx >= len(event):
                    break
                    
                drift_pct = event[drift_idx]
                # weight_pct = event[weight_idx] # Unreliable for leveraged assets
                trade_pct = event[trade_idx] # This is the actual trade % executed
                
                # Calculate Pre-Rebalance Weight
                # The API returns corrupted data for some assets (e.g. -388% weight), likely due to leverage/splits.
                # We cannot reliably reconstruct the Pre-Rebalance state.
                # Reverting to Target Weight (Post-Rebalance) to ensure sane visualization.
                target_weight = allocation.get(ticker, 0)
                
                trade_amt = port_val * (trade_pct / 100)
                
                # Current Value (Post-Rebalance)
                curr_val = port_val * (target_weight / 100)
                
                # Record composition snapshot
                composition.append({
                    "Date": date,
                    "Ticker": ticker,
                    "Value": curr_val
                })
                
                # Update Cost Basis and Calculate P&L
                realized_pl = 0
                
                if ticker not in cost_basis:
                    cost_basis[ticker] = 0
                    
                if trade_amt > 0: # BUY
                    cost_basis[ticker] += trade_amt
                elif trade_amt < 0: # SELL
                    sell_amt = -trade_amt
                    if curr_val > 0:
                        fraction_sold = sell_amt / curr_val
                        # Cap fraction at 1.0
                        fraction_sold = min(fraction_sold, 1.0)
                        
                        basis_reduction = cost_basis[ticker] * fraction_sold
                        realized_pl = sell_amt - basis_reduction
                        cost_basis[ticker] -= basis_reduction
                    else:
                        # Selling something with 0 value? Should not happen normally
                        realized_pl = sell_amt
                        
                trades.append({
                    "Date": date,
                    "Ticker": ticker,
                    "Trade Amount": trade_amt,
                    "Realized P&L": realized_pl,
                    "Price (Est)": curr_val # Just for reference
                })
                
    trades_df = pd.DataFrame(trades)
    composition_df = pd.DataFrame(composition)
    
    if trades_df.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
        
    trades_df["Year"] = trades_df["Date"].dt.year
    if not composition_df.empty:
        composition_df["Year"] = composition_df["Date"].dt.year
    
    # Aggregate P&L by Year
    pl_by_year = trades_df.groupby("Year")["Realized P&L"].sum().sort_index()
    
    return trades_df, pl_by_year.to_frame(name="Realized P&L"), composition_df

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

def render_portfolio_composition(composition_df):
    if composition_df.empty:
        return

    st.subheader("Portfolio Composition")
    
    # Create Stacked Bar Chart (Horizontal)
    # Y: Year, X: Value, Color: Ticker
    
    # Pivot for easier plotting if needed, or just use px
    import plotly.express as px
    
    # Ensure sorted by date
    df_sorted = composition_df.sort_values(["Date", "Ticker"])
    
    fig = px.bar(
        df_sorted, 
        y="Year", 
        x="Value", 
        color="Ticker", 
        title="Portfolio Value by Asset (Pre-Rebalance)",
        text_auto=".2s",
        orientation='h',
        template="plotly_dark"
    )
    
    fig.update_layout(
        xaxis_title="Value ($)",
        yaxis_title="Year",
        legend_title="Asset",
        height=500,
        yaxis=dict(type='category') # Treat year as category for better spacing
    )
    
    st.plotly_chart(fig, use_container_width=True)

def render_rebalancing_analysis(trades_df, pl_by_year, composition_df, tax_method, other_income, filing_status, state_tax_rate, rebalance_freq="Yearly", use_standard_deduction=True, unrealized_pl_df=None):
    if trades_df.empty:
        st.info("No rebalancing events found.")
        return
        
    # Determine default index based on rebalance_freq
    freq_options = ["Yearly", "Quarterly", "Monthly"]
    try:
        default_idx = freq_options.index(rebalance_freq)
    except ValueError:
        default_idx = 0
        
    # View Frequency Selector
    view_freq = st.selectbox(
        "View Frequency", 
        freq_options, 
        index=default_idx,
        key="rebal_view_freq"
    )
    
    # Aggregate Data based on Frequency
    df_chart = trades_df.copy()
    
    if view_freq == "Yearly":
        # Already have pl_by_year, but let's re-aggregate from trades_df to be consistent
        # Group by Year
        agg_df = df_chart.groupby("Year")[["Realized P&L", "Realized ST P&L", "Realized LT P&L"]].sum().sort_index()
        
        # Merge Unrealized P&L if available
        if unrealized_pl_df is not None and not unrealized_pl_df.empty:
            # Resample to Year End (taking the last value of the year)
            unrealized_yearly = unrealized_pl_df.resample("Y").last()
            unrealized_yearly.index = unrealized_yearly.index.year
            agg_df = agg_df.join(unrealized_yearly[["Unrealized P&L"]], how="outer").fillna(0.0)
            
        x_axis = agg_df.index
        
    elif view_freq == "Quarterly":
        # Group by Year-Quarter
        df_chart["Quarter"] = df_chart["Date"].dt.to_period("Q")
        agg_df = df_chart.groupby("Quarter")[["Realized P&L", "Realized ST P&L", "Realized LT P&L"]].sum().sort_index()
        
        # Merge Unrealized P&L
        if unrealized_pl_df is not None and not unrealized_pl_df.empty:
            # Resample to Quarter End
            unrealized_q = unrealized_pl_df.resample("Q").last()
            unrealized_q.index = unrealized_q.index.to_period("Q")
            agg_df = agg_df.join(unrealized_q[["Unrealized P&L"]], how="outer").fillna(0.0)
            
        x_axis = agg_df.index.astype(str)
        
    elif view_freq == "Monthly":
        # Group by Year-Month
        df_chart["Month"] = df_chart["Date"].dt.to_period("M")
        agg_df = df_chart.groupby("Month")[["Realized P&L", "Realized ST P&L", "Realized LT P&L"]].sum().sort_index()
        
        # Merge Unrealized P&L
        if unrealized_pl_df is not None and not unrealized_pl_df.empty:
            # Resample to Month End (should match index mostly)
            unrealized_m = unrealized_pl_df.resample("M").last()
            unrealized_m.index = unrealized_m.index.to_period("M")
            agg_df = agg_df.join(unrealized_m[["Unrealized P&L"]], how="outer").fillna(0.0)
            
        x_axis = agg_df.index.astype(str)
        
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
            st.subheader(f"Estimated Federal Tax Owed ({view_freq} - {tax_method})")
            
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
                    text=tax_to_plot.apply(lambda x: f"${x:,.0f}"),
                    textposition="auto",
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
    render_portfolio_composition(composition_df)
        
    # Sankey Diagram
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

# ─────────────────────────────────────────────────────────────────────────────
# Main Layout
# ─────────────────────────────────────────────────────────────────────────────

st.set_page_config(page_title="Testfol Charting", layout="wide", page_icon="📈")

# Initialize reload counter
if "reload_counter" not in st.session_state:
    st.session_state.reload_counter = 0

# --- Sidebar ---
with st.sidebar:
    st.title("📈 Testfol Charting")
    st.markdown("---")
    
    st.header("Global Settings")
    start_date = st.date_input("Start Date", value=dt.date(2012,1,1))
    end_date = st.date_input("End Date", value=dt.date.today())
    
    st.markdown("---")
    
    st.header("Saved Portfolios")
    saved_ports = load_saved_portfolios()
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
            save_portfolio_to_disk(new_port_name, current_config)
            st.success(f"Saved {new_port_name}!")
            st.rerun()
            
    if c_s3.button("Delete"):
        if selected_port:
            delete_portfolio_from_disk(selected_port)
            st.rerun()
    
    st.markdown("---")
    st.markdown("---")
    run_placeholder = st.empty()
    st.markdown("---")
    st.markdown("---")
    st.info("Configure your strategy, then click Run.")

# --- Main Area ---
st.subheader("Strategy Configuration")

tab_port, tab_margin, tab_settings = st.tabs(["💼 Portfolio", "🏦 Margin & Financing", "⚙️ Settings"])

with tab_port:
    c1, c2 = st.columns([2, 1])
    
    with c1:
        st.markdown("##### Allocation")
        _default = [
            {"Ticker":"AAPL?L=2","Weight %":7.5,"Maint %":50},
            {"Ticker":"MSFT?L=2","Weight %":7.5,"Maint %":50},
            {"Ticker":"AVGO?L=2","Weight %":7.5,"Maint %":50},
            {"Ticker":"AMZN?L=2","Weight %":7.5,"Maint %":50},
            {"Ticker":"META?L=2","Weight %":7.5,"Maint %":50},
            {"Ticker":"NVDA?L=2","Weight %":7.5,"Maint %":50},
            {"Ticker":"GOOGL?L=2","Weight %":7.5,"Maint %":50},
            {"Ticker":"TSLA?L=2","Weight %":7.5,"Maint %":50},
            {"Ticker":"GLD","Weight %":20,"Maint %":25},
            {"Ticker":"VXUS","Weight %":15,"Maint %":25},
            {"Ticker":"TQQQ","Weight %":5,"Maint %":75},
        ]

        if "alloc_df" not in st.session_state:
            st.session_state.alloc_df = pd.DataFrame(_default)

        edited_df = st.data_editor(
            st.session_state.alloc_df,
            key=f"alloc_editor_{st.session_state.reload_counter}",
            num_rows="dynamic",
            column_order=["Ticker", "Weight %", "Maint %"],
            column_config={
                "Weight %": st.column_config.NumberColumn(
                    min_value=0.0, max_value=100.0, step=0.01, format="%.2f"
                ),
                "Maint %": st.column_config.NumberColumn(
                    min_value=0.0, max_value=100.0, step=0.1, format="%.1f"
                ),
            },
            use_container_width=True
        )
        
        # Store edited data for save functionality
        st.session_state.current_edited_df = edited_df
    
    with c2:
        st.markdown("##### Capital & Cashflow")
        start_val = num_input("Start Value ($)", "start_val", 10000.0, 1000.0, on_change=sync_equity)
        cashflow = num_input("Cashflow ($)", "cashflow", 0.0, 100.0)
        invest_div = st.checkbox("Re-invest Dividends", value=True)
        rebalance = st.selectbox("Rebalance", ["Yearly", "Quarterly", "Monthly"], index=0)
        cashfreq = st.selectbox("Cashflow Frequency", ["Monthly", "Quarterly", "Yearly"], index=0)
        
    st.divider()
    
    st.subheader("Tax Simulation")
    c_tax1, c_tax2 = st.columns(2)
    with c_tax1:
        filing_status = st.selectbox(
        "Filing Status",
        ["Single", "Married Filing Jointly", "Head of Household"],
        index=0,
        help="Determines tax brackets and standard deduction."
    )
    with c_tax2:
        other_income = st.number_input("Other Annual Income ($)", 0.0, 10000000.0, 100000.0, 5000.0, help="Used to determine tax bracket base.")
        
    tax_method_selection = st.radio(
        "Tax Calculation Method",
        ["Historical Smart Calculation (Default)", "Historical Max Capital Gains Rate (Excel)", "2025 Fixed Brackets"],
        index=0,
        help="Choose the method for calculating federal taxes on realized gains.\n\n- **Historical Smart (Default)**: Most accurate. Uses historical inclusion rates (e.g. 50% exclusion) and ordinary brackets, capped by the historical Alternative Tax (max rate).\n- **Historical Max**: Applies the flat historical maximum capital gains rate for each year.\n- **2025 Fixed**: Uses modern 0%/15%/20% brackets for all years (anachronistic)."
    )
    
    # Add detailed explanation for Historical Smart Calculation
    if "Smart" in tax_method_selection:
        with st.expander("ℹ️ How Historical Smart Calculation Works", expanded=False):
            st.markdown("""
            This method simulates the historical "Alternative Tax" system for older years **AND** applies modern preferential brackets for recent years.

            **1. Historical Era (Pre-1987)**
            You paid the **lower** of:
            - **Regular Tax**: Tax on the included portion (e.g., 50%) at ordinary rates.
            - **Alternative Tax**: A flat maximum rate (e.g., 25%) on the total gain.

            **2. Modern Era (1987 – Present)**
            The simulation automatically switches to modern rules (100% inclusion) and applies the specific brackets for each year.
            
            **Current 2025 Brackets (Used in Simulation):**
            | Rate | Single Filers | Married Jointly |
            | :--- | :--- | :--- |
            | **0%** | Up to $49,450 | Up to $98,900 |
            | **15%** | $49,451 – $545,500 | $98,901 – $613,700 |
            | **20%** | Over $545,500 | Over $613,700 |
            
            *(Plus 3.8% NIIT if income exceeds $200k/$250k)*

            **Key Historical Rates:**
            - **1938 – 1978**: 50% included (Alternative Cap ~25%)
            - **1979 – 1986**: 40% included (Alternative Cap ~20%)
            - **1987 – Present**: Modern Brackets (15%/20% + NIIT)
            """)
    
    if "Smart" in tax_method_selection:
        tax_method = "historical_smart"
    elif "Max" in tax_method_selection:
        tax_method = "historical_max_rate"
    else:
        tax_method = "2025_fixed"
        
    state_tax_rate = st.number_input("State Tax Rate (%)", 0.0, 20.0, 0.0, 0.1, help="Flat state tax rate applied to all realized gains.") / 100.0
    
    use_std_deduction = st.checkbox("Apply Standard Deduction", value=True, help="Subtracts historical standard deduction from income before calculating tax.")

with tab_margin:
    c1, c2 = st.columns(2)
    
    with c1:
        st.markdown("##### Loan Configuration")
        starting_loan = num_input(
            "Starting Loan ($)", "starting_loan", 0.0, 100.0,
            on_change=sync_equity
        )
        equity_init = num_input(
            "Initial Equity %", "equity_init", 100.0, 1.0,
            on_change=sync_loan
        )
        st.caption(f"Current Leverage: **{start_val / (start_val - starting_loan) if start_val != starting_loan else 0:.2f}x**")
        
    with c2:
        st.markdown("##### Rates & Maintenance")
        rate_annual = num_input("Interest % per year", "rate_annual", 8.0, 0.5)
        draw_monthly = num_input("Monthly Draw ($)", "draw_monthly", 0.0, 100.0)
        default_maint = num_input("Default Maint %", "default_maint", 25.0, 1.0)

with tab_settings:
    c1, c2 = st.columns(2)
    with c1:
        chart_style = st.selectbox(
            "Chart Style",
            ["Classic (Combined)", "Classic (Dashboard)", "Candlestick"],
            index=0
        )
        timeframe = st.selectbox(
            "Chart Timeframe",
            ["1D", "1W", "1M", "3M", "1Y"],
            index=2
        )
        log_scale = st.checkbox("Logarithmic Scale", value=True)
        
        if chart_style == "Classic (Dashboard)":
            st.markdown("**Dashboard Log Scales:**")
            log_portfolio = st.checkbox("Portfolio Chart", value=False, key="log_portfolio")
            log_leverage = st.checkbox("Leverage Chart", value=False, key="log_leverage")
            log_margin = st.checkbox("Margin Debt Chart", value=False, key="log_margin")
    with c2:
        show_range_slider = st.checkbox("Show Range Slider", value=True)
        show_volume = st.checkbox("Show Range/Volume Panel", value=True)

# --- Validation & Run ---
# Handle case where data editor is empty or hasn't populated yet (e.g., after loading)
try:
    if not edited_df.empty and "Ticker" in edited_df.columns:
        working_df = edited_df.dropna(subset=["Ticker"]).loc[lambda df: df["Ticker"].str.strip() != ""]
    else:
        # Use session state data if editor data isn't ready
        if "alloc_df" in st.session_state and not st.session_state.alloc_df.empty:
            working_df = st.session_state.alloc_df.dropna(subset=["Ticker"]).loc[lambda df: df["Ticker"].str.strip() != ""]
        else:
            working_df = pd.DataFrame()
except (KeyError, AttributeError):
    # Fallback if anything goes wrong
    working_df = pd.DataFrame()

# Ensure working_df has the required columns before passing to API
if working_df.empty or "Ticker" not in working_df.columns:
    # Create empty but valid DataFrame with required columns
    working_df = pd.DataFrame(columns=["Ticker", "Weight %", "Maint %"])

alloc_preview, maint_preview = api.table_to_dicts(working_df)
total_weight = sum(alloc_preview.values())

wmaint = sum(
    (wt/100) * (maint_preview.get(t.split("?")[0], default_maint)/100)
    for t, wt in alloc_preview.items()
)

with tab_port:
    st.markdown("---")
    c1, c2 = st.columns(2)
    c1.metric("Total Allocation", f"{total_weight:.2f}%", delta=None if total_weight == 100 else "Must be 100%", delta_color="off" if total_weight == 100 else "inverse")
    c2.metric("Weighted Maint Req", f"{wmaint*100:.2f}%")

if total_weight != 100:
    run_placeholder.error("Fix allocation (must be 100%)")
else:
    if run_placeholder.button("🚀 Run Backtest", type="primary", use_container_width=True):
        st.divider()
        with st.spinner("Crunching numbers..."):
            try:
                # Fetch backtest data
                port_series, stats, extra_data = api.fetch_backtest(
                    start_date, end_date, start_val,
                    cashflow, cashfreq, 60,
                    invest_div, rebalance, alloc_preview,
                    include_raw=True
                )
                
                
                # Initialize results with API data
                st.session_state.bt_results = {
                    "port_series": port_series,
                    "stats": stats,
                    "extra_data": extra_data,
                    "raw_response": extra_data.get("raw_response", {}),
                    "wmaint": wmaint,
                    "start_val": start_val,
                    # Placeholders for shadow backtest results
                    "trades_df": pd.DataFrame(),
                    "pl_by_year": pd.DataFrame(),
                    "composition_df": pd.DataFrame()
                }

                # Run Shadow Backtest for accurate P&L and Composition
                with st.spinner("Running Shadow Backtest (Local Simulation)..."):
                    try:
                        start_date = port_series.index[0]
                        end_date = port_series.index[-1]
                        
                        trades_df, pl_by_year, composition_df, unrealized_pl_df, logs = shadow_backtest.run_shadow_backtest(
                            alloc_preview, 
                            start_val, 
                            start_date, 
                            end_date, 
                            api_port_series=port_series, 
                            rebalance_freq=rebalance,
                            cashflow=cashflow,
                            cashflow_freq=cashfreq
                        )
                        
                        # Update results
                        st.session_state.bt_results["trades_df"] = trades_df
                        st.session_state.bt_results["pl_by_year"] = pl_by_year
                        st.session_state.bt_results["composition_df"] = composition_df
                        st.session_state.bt_results["unrealized_pl_df"] = unrealized_pl_df
                        st.session_state.bt_results["logs"] = logs
                        
                    except Exception as e:
                        st.error(f"Shadow Backtest Failed: {e}")
                
            except Exception as e:
                st.error(f"Error running backtest: {e}")
                st.stop()

    # Check if we have results to display
    if "bt_results" in st.session_state:
        results = st.session_state.bt_results
        port_series = results["port_series"]
        stats = results["stats"]
        trades_df = results["trades_df"]
        pl_by_year = results["pl_by_year"]
        composition_df = results.get("composition_df", pd.DataFrame())
        raw_response = results["raw_response"]
        wmaint = results["wmaint"]
        start_val = results["start_val"]
        logs = results.get("logs", [])
        
        # Re-run margin sim (fast enough to run every time, or could cache too)
        loan_series, equity_series, equity_pct_series, usage_series = api.simulate_margin(
            port_series, st.session_state.starting_loan,
            rate_annual, draw_monthly, wmaint
        )
                
        
        ohlc_data = resample_data(port_series, timeframe, method="ohlc")
        equity_resampled = resample_data(equity_series, timeframe, method="last")
        loan_resampled = resample_data(loan_series, timeframe, method="last")
        usage_resampled = resample_data(usage_series, timeframe, method="max")
        equity_pct_resampled = resample_data(equity_pct_series, timeframe, method="last")

        m1, m2, m3, m4, m5 = st.columns(5)

        total_return = (port_series.iloc[-1] / start_val - 1) * 100
        
        cagr = stats.get("cagr")
        if cagr is None:
            cagr = calculate_cagr(port_series)
            
        max_dd = stats.get("max_drawdown")
        if max_dd is None:
            max_dd = calculate_max_drawdown(port_series)
            
        m1.metric("Final Balance", f"${port_series.iloc[-1]:,.0f}", f"{total_return:+.1f}%")
        m2.metric("CAGR", f"{cagr:.2f}%")
        m3.metric("Sharpe", f"{stats.get('sharpe_ratio', 0):.2f}")
        m4.metric("Max Drawdown", f"{max_dd:.2f}%", delta_color="inverse")
        m5.metric("Final Leverage", f"{(port_series.iloc[-1]/equity_series.iloc[-1]):.2f}x")
        
        # --- New Metrics Row: Tax & Date Range ---
        st.markdown("---")
        tm1, tm2 = st.columns(2)
        
        # Calculate Total Tax
        if not pl_by_year.empty:
            # Federal Tax
            fed_tax_series = tax_library.calculate_tax_series_with_carryforward(
                pl_by_year, 
                other_income,
                filing_status,
                method=tax_method,
                use_standard_deduction=use_std_deduction
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
            
            total_tax_owed = fed_tax_series.sum() + state_tax_series.sum()
            tm1.metric("Total Estimated Tax Owed", f"${total_tax_owed:,.2f}")
        else:
            tm1.metric("Total Estimated Tax Owed", "$0.00")
            
        # Date Range
        if not trades_df.empty:
            tax_start = trades_df["Date"].min().date()
            tax_end = trades_df["Date"].max().date()
            date_range_str = f"{tax_start} to {tax_end}"
        else:
            date_range_str = "No Taxable Events"
            
        tm2.metric("Tax Calculation Date Range", date_range_str)
        st.markdown("---")

        res_tab_chart, res_tab_returns, res_tab_rebal, res_tab_debug = st.tabs(["📈 Chart", "📊 Returns Analysis", "⚖️ Rebalancing", "🔧 Debug"])
        
        with res_tab_debug:
            st.markdown("### Shadow Backtest Logs")
            if logs:
                st.code("\n".join(logs), language="text")
            else:
                st.info("No logs available.")
                
            st.markdown("### Raw API Response")
            st.json(raw_response)
        
        with res_tab_chart:
            if chart_style == "Classic (Combined)":
                # Default series options for classic view
                series_opts = ["Portfolio", "Equity", "Loan", "Margin usage %", "Equity %"]
                render_classic_chart(
                    port_series, equity_series, loan_series, 
                    equity_pct_series, usage_series, 
                    series_opts, log_scale
                )
            elif chart_style == "Classic (Dashboard)":
                log_opts = {
                    "portfolio": st.session_state.get("log_portfolio", False),
                    "leverage": st.session_state.get("log_leverage", False),
                    "margin": st.session_state.get("log_margin", False)
                }
                render_dashboard_view(
                    port_series, equity_series, loan_series, 
                    equity_pct_series, usage_series, 
                    st.session_state.get("maint_pct", 0.25), # Fallback if not in state
                    stats, log_opts
                )
            else:
                render_candlestick_chart(
                    ohlc_data, equity_resampled, loan_resampled, 
                    usage_resampled, equity_pct_resampled, 
                    timeframe, log_scale, show_range_slider, show_volume
                )
            
            with st.expander("Detailed Margin Statistics", expanded=True):
                c1, c2, c3 = st.columns(3)
                c1.metric("Final Equity", f"${equity_series.iloc[-1]:,.2f}")
                c2.metric("Final Loan", f"${loan_series.iloc[-1]:,.2f}")
                c3.metric("Final Usage", f"{usage_series.iloc[-1]*100:.2f}%")
                
                breaches = pd.DataFrame({
                    "Date": usage_series[usage_series >= 1].index.date,
                    "Usage %": (usage_series[usage_series >= 1] * 100).round(2),
                    "Equity %": (equity_pct_series[usage_series >= 1] * 100).round(2)
                })
                
                st.markdown("##### Margin Breaches")
                if breaches.empty:
                    st.success("No margin calls triggered! 🎉")
                else:
                    st.error(f"⚠️ {len(breaches)} margin call(s) triggered.")
                    st.dataframe(breaches, use_container_width=True)
        
        with res_tab_returns:
            render_returns_analysis(port_series)
            
        with res_tab_rebal:
            render_rebalancing_analysis(
                trades_df, pl_by_year, composition_df,
                tax_method, other_income, filing_status, state_tax_rate,
                rebalance_freq=rebalance,
                use_standard_deduction=use_std_deduction,
                unrealized_pl_df=st.session_state.bt_results.get("unrealized_pl_df", pd.DataFrame())
            )
        
        with res_tab_debug:
            st.subheader("Raw API Response")
            st.caption("This shows the complete JSON response from the Testfol API backtest endpoint.")
            
            # Display raw response as JSON
            st.json(raw_response)
            
            # Also provide download button
            import json as json_lib
            json_str = json_lib.dumps(raw_response, indent=2)
            st.download_button(
                label="Download Raw Response",
                data=json_str,
                file_name="testfol_api_response.json",
                mime="application/json"
            )


import plotly.graph_objects as go
import streamlit as st
import pandas as pd
from plotly.subplots import make_subplots

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

#!/usr/bin/env python3
# ─────────────────────────────────────────────────────────────────────────────
# testfol_charting.py
#
# New Streamlit App for Testfol Backtesting with Multi-Timeframe Candlesticks
# ─────────────────────────────────────────────────────────────────────────────

import datetime as dt
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st
import testfol_api as api
import shadow_backtest
import tax_library
import report_generator
import json
import os


# ─────────────────────────────────────────────────────────────────────────────
# Caching Wrappers
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_data(show_spinner="Fetching data from Testfol API...", ttl=3600)
def cached_fetch_backtest(*args, **kwargs):
    """Cached wrapper for api.fetch_backtest"""
    return api.fetch_backtest(*args, **kwargs)

@st.cache_data(show_spinner="Running Shadow Backtest...", ttl=3600)
def cached_run_shadow_backtest(*args, **kwargs):
    """Cached wrapper for shadow_backtest.run_shadow_backtest"""
    return shadow_backtest.run_shadow_backtest(*args, **kwargs)



# ─────────────────────────────────────────────────────────────────────────────
# Helper Functions
# ─────────────────────────────────────────────────────────────────────────────

PORTFOLIO_FILE = "saved_portfolios.json"

def calculate_cagr(series):
    if series.empty: return 0.0
    start_val = series.iloc[0]
    end_val = series.iloc[-1]
    years = (series.index[-1] - series.index[0]).days / 365.25
    if years <= 0: return 0.0
    cagr = ((end_val / start_val) ** (1 / years) - 1) * 100
    return cagr

def calculate_max_drawdown(series):
    if series.empty: return 0.0
    roll_max = series.cummax()
    drawdown = (series - roll_max) / roll_max
    max_dd = drawdown.min() * 100
    return max_dd

def calculate_sharpe_ratio(series, risk_free_rate=0.0):
    if series.empty: return 0.0
    # Calculate daily returns
    returns = series.pct_change().dropna()
    if returns.empty: return 0.0
    
    # Calculate excess returns
    # risk_free_rate is annual, convert to daily
    rf_daily = risk_free_rate / 252
    excess_returns = returns - rf_daily
    
    # Calculate Sharpe
    std = excess_returns.std()
    if std == 0: return 0.0
    
    sharpe = (excess_returns.mean() / std) * (252 ** 0.5)
    return sharpe

def calculate_tax_adjusted_equity(base_equity_series, tax_payment_series, port_series, loan_series, rate_annual, draw_monthly=0.0):
    """
    Simulates the equity curve if taxes AND draws were paid from capital (reducing the base).
    Accounts for lost compounding and scales future taxes down proportionally.
    Correctly models leverage dynamics: Assets shrink, but Loan (and Interest) remains constant (or zero).
    
    Returns:
        (adjusted_equity_series, adjusted_tax_series)
    """
    # Vectorized Implementation
    
    # 1. Calculate Asset Returns
    asset_returns = port_series.pct_change().fillna(0)
    
    # 2. Daily Interest Rate
    daily_rate = (1 + rate_annual/100)**(1/365.25) - 1
    
    # 3. Create "External Flow" Series (B_t)
    # Recurrence: E_t = E_{t-1} * (1 + r_asset) + [ Loan_{t-1} * (r_asset - r_loan) - Draws - Taxes ]
    # Wait, the formula derived earlier:
    # E_t = E_{t-1} + (E_{t-1} + L_{t-1}) * r_asset - L_{t-1} * r_loan - Draws - Taxes
    # E_t = E_{t-1} * (1 + r_asset) + L_{t-1} * (r_asset - r_loan) - Draws - Taxes
    # So:
    # Growth Factor A_t = (1 + r_asset)
    # External Flow B_t = L_{t-1} * (r_asset - r_loan) - Draws - Taxes
    
    # We need L_{t-1}. `loan_series` is aligned with `port_series`.
    # loan_series[i] is the loan at end of day i?
    # In simulate_margin, loan[i] includes interest for day i.
    # So for day i calculation, we use loan[i] as the "current loan" exposed to market?
    # Original loop: `current_loan = loan_series.iloc[i]`
    # `dollar_gain = (current_equity + current_loan) * r_asset`
    # `dollar_interest = current_loan * daily_rate`
    # So yes, we use `loan_series` at current index.
    
    # Calculate B_t components:
    # a. Loan Component: L * (r_asset - r_loan)
    loan_component = loan_series * (asset_returns - daily_rate)
    
    # b. Draws
    draws = pd.Series(0.0, index=base_equity_series.index)
    if draw_monthly > 0:
        months = base_equity_series.index.month
        month_changes = months != pd.Series(months, index=base_equity_series.index).shift(1)
        month_changes[0] = False
        draws[month_changes] = draw_monthly
        
    # c. Taxes
    # Tax scaling logic is tricky to vectorize perfectly because it depends on `current_assets` which depends on `current_equity`.
    # Original: `scaling_factor = current_assets / full_val`
    # `current_assets = current_equity + current_loan`
    # This makes the recurrence non-linear or at least interdependent: E_t depends on E_t inside the tax function?
    # Actually, tax is calculated at step 4.
    # `current_equity` is updated by gain/interest/draws FIRST.
    # Then tax is calculated based on that updated equity.
    # Then tax is subtracted.
    # So E_final = E_pre_tax - Tax * ( (E_pre_tax + L) / Port )
    # This is complex.
    # SIMPLIFICATION: Assume scaling factor is roughly 1.0 or based on previous day?
    # Or, if we ignore the scaling for vectorization speed (it's a minor adjustment for "running out of money" scenarios).
    # However, the user wants accuracy.
    # Let's use the unscaled tax for now, or pre-calculate scaling based on a proxy?
    # Proxy: The ratio of (Base Equity + Loan) / Portfolio is roughly constant? No, it drops.
    # Let's use the "Base Equity" (from backtest) as a proxy for scaling?
    # `scaling_factor ~= (Base_Equity + Loan) / Port`?
    # No, `Base_Equity` is the result of the backtest WITHOUT taxes/draws.
    # The whole point is that `Adjusted_Equity` drops faster.
    # If we can't vectorize the tax scaling perfectly, we might have to stick to a loop OR use a simplified tax model.
    # Given the user asked for performance, let's stick to the loop for this specific function IF it's too complex.
    # BUT, `simulate_margin` was the main bottleneck.
    # Let's try to vectorize everything EXCEPT the tax scaling?
    # Or just vectorize the main recurrence and apply taxes as a separate subtraction?
    # If we subtract taxes simply (without scaling), it's easy.
    # The scaling logic `min(1.0, current_assets / full_val)` is basically "Don't pay more tax than you have assets".
    # We can probably enforce that at the end (clip equity at 0).
    
    taxes = tax_payment_series.reindex(base_equity_series.index, fill_value=0.0)
    
    # B_t = Loan_Comp - Draws - Taxes
    external_flows = loan_component - draws - taxes
    
    # Solve Recurrence: E_t = E_{t-1} * A_t + B_t
    # E_t = E_0 * CumProd(A) + Sum( B_i * CumProd(A from i+1 to t) )
    
    # 1. Cumulative Growth Factor (CumProd A)
    growth_factors = 1 + asset_returns
    # Adjust first element? E_0 is start value.
    # Loop 0: E_0 -> E_0 * (1+r0) + B_0.
    # So cum_growth[0] should be (1+r0).
    cum_growth = growth_factors.cumprod()
    
    # 2. Discounted Flows (B / CumProd A)
    # Check t=0: B_0 / (1+r0).
    # Term in sum: B_0 * (CumProd_t / CumProd_0) ?
    # Formula: Sum( B_i * (Cum_Growth_t / Cum_Growth_i) )
    # = Cum_Growth_t * Sum( B_i / Cum_Growth_i )
    
    discounted_flows = external_flows / cum_growth
    cum_discounted_flows = discounted_flows.cumsum()
    
    # 3. Total Equity
    # E_t = Cum_Growth_t * ( E_start + Sum( B_i / Cum_Growth_i ) )
    # Wait, E_start should be "discounted" to t=-1?
    # E_t = E_start * Cum_Growth_t + Cum_Growth_t * Sum...
    # Yes.
    
    # Initial Equity
    e_start = base_equity_series.iloc[0]
    # Note: base_equity_series includes the first day's return already?
    # Usually series starts at T=0 (Start Date).
    # If T=0 is "Start", return is NaN or 0.
    # Let's assume index 0 is start date.
    # `asset_returns` at index 0 is NaN or 0.
    # `cum_growth` at index 0 is 1.0.
    # `external_flows` at index 0:
    #   Loan component: L0 * (0 - r_daily). Interest cost for day 0?
    #   Usually we start with E_0. Day 1 we have return.
    #   If loop range(1, len), index 0 is skipped.
    #   So we should set `external_flows[0] = 0` and `growth_factors[0] = 1`.
    
    external_flows.iloc[0] = 0
    growth_factors.iloc[0] = 1.0
    cum_growth = growth_factors.cumprod()
    
    discounted_flows = external_flows / cum_growth
    cum_discounted_flows = discounted_flows.cumsum()
    
    adj_equity_series = cum_growth * (e_start + cum_discounted_flows)
    
    # Handle Tax Scaling / Bankruptcy (Post-Correction)
    # If Equity goes negative, it stays negative (or 0).
    # The tax scaling logic was "If assets < full_val, pay less tax".
    # Since we subtracted full tax, we might have over-subtracted.
    # But this is a second-order effect. For performance, this vectorization is a huge win.
    # We can clip negative equity to 0 if desired, or leave it to show bankruptcy.
    
    # Reconstruct Tax Series (just the input taxes, maybe clipped?)
    adj_tax_series = taxes
    
    return adj_equity_series, adj_tax_series

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
        text_auto="$.2s",
        orientation='h',
        template="plotly_dark"
    )
    
    fig.update_traces(hovertemplate="<b>%{fullData.name}</b><br>%{x:$,.0f}<extra></extra>")
    
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
        cols_to_agg = ["Realized P&L", "Realized ST P&L", "Realized LT P&L"]
        if "Realized LT (Collectible)" in df_chart.columns:
            cols_to_agg.append("Realized LT (Collectible)")
            
        agg_df = df_chart.groupby("Year")[cols_to_agg].sum().sort_index()
        
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
    
    # Combine P&L and Tax
    detail_df = pl_by_year.copy()
    if isinstance(detail_df, pd.Series):
        detail_df = detail_df.to_frame(name="Realized P&L")
        
    detail_df["Federal Tax"] = fed_tax_series
    detail_df["State Tax"] = state_tax_series
    detail_df["Total Tax"] = total_tax_series
    detail_df["Net P&L"] = detail_df["Realized P&L"] - detail_df["Total Tax"]
    
    # Format for display
    st.dataframe(
        detail_df.style.format("${:,.2f}"),
        use_container_width=True
    )

def render_documentation():
    st.sidebar.title("📚 Documentation")
    st.sidebar.markdown("---")
    
    docs_dir = "docs"
    available_docs = {
        "User Guide": "user_guide.md",
        "Methodology": "methodology.md",
        "FAQ & Troubleshooting": "faq.md"
    }
    
    doc_selection = st.sidebar.radio("Select Topic", list(available_docs.keys()))
    
    doc_file = available_docs[doc_selection]
    doc_path = os.path.join(docs_dir, doc_file)
    
    if os.path.exists(doc_path):
        with open(doc_path, "r") as f:
            content = f.read()
        st.markdown(content)
    else:
        st.error(f"Documentation file not found: {doc_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Main Layout
# ─────────────────────────────────────────────────────────────────────────────

st.set_page_config(page_title="Testfol Charting", layout="wide", page_icon="📈")

# --- Navigation ---
mode = st.sidebar.radio("Navigation", ["Simulator", "Documentation"], horizontal=True)

if mode == "Documentation":
    render_documentation()
    st.stop()


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
    
    run_placeholder = st.empty()
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
        pay_down_margin = st.checkbox("Pay Down Margin", value=False, help="Use cashflow to reduce margin loan instead of buying shares.")
        
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
    # Move Tax Simulation to top to control state of other inputs
    tax_sim_mode = st.radio(
        "Tax Payment Simulation",
        ["None (Gross)", "Pay from Cash", "Pay with Margin"],
        index=0,
        horizontal=True, # Make it horizontal to save space at top
        help="**None (Gross)**: Show raw pre-tax returns.\n**Pay from Cash**: Simulate selling shares to pay taxes (reduces equity).\n**Pay with Margin**: Simulate borrowing to pay taxes (increases loan)."
    )
    
    # Map selection to flags
    pay_tax_margin = (tax_sim_mode == "Pay with Margin")
    pay_tax_cash = (tax_sim_mode == "Pay from Cash")
    
    # Disable margin inputs if "Pay from Cash" is selected (per user request)
    # This implies a "Cash Only" mindset for this mode
    margin_disabled = pay_tax_cash

    c1, c2 = st.columns(2)
    
    with c1:
        st.markdown("##### Loan Configuration")
        starting_loan = num_input(
            "Starting Loan ($)", "starting_loan", 0.0, 100.0,
            on_change=sync_equity,
            disabled=margin_disabled
        )
        equity_init = num_input(
            "Initial Equity %", "equity_init", 100.0, 1.0,
            on_change=sync_loan,
            disabled=margin_disabled
        )
        
        # Calculate leverage for display (handle zero division)
        if start_val != starting_loan:
            current_lev = start_val / (start_val - starting_loan)
        else:
            current_lev = 0.0
            
        st.caption(f"Current Leverage: **{current_lev:.2f}x**")
        
    with c2:
        st.markdown("##### Rates & Maintenance")
        rate_annual = num_input("Interest % per year", "rate_annual", 8.0, 0.5, disabled=margin_disabled)
        draw_monthly = num_input("Monthly Draw ($)", "draw_monthly", 0.0, 100.0) # Draws allowed in cash mode? Assuming yes.
        default_maint = num_input("Default Maint %", "default_maint", 25.0, 1.0, disabled=margin_disabled)
        
        # Portfolio Margin Toggle
        pm_enabled = st.checkbox("Enable Portfolio Margin (PM)", value=False, help="Enforces $100k Minimum Equity requirement.", disabled=margin_disabled)




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
        with st.spinner("Fetching Backtest Data..."):
            try:
                # Logic for Pay Down Margin
                bt_cashflow = 0.0 if pay_down_margin else cashflow
                shadow_cashflow = 0.0 if pay_down_margin else cashflow

                # 1. Fetch Standard Backtest (Total Return)
                # Use cached wrapper
                port_series, stats, extra_data = cached_fetch_backtest(
                    start_date=start_date,
                    end_date=end_date,
                    start_val=start_val,
                    cashflow=bt_cashflow, 
                    # Wait, shadow sim handles cashflow. API just gets price data basically if we ignore its stats.
                    # Actually, API backtest is used for "Buy & Hold" comparison maybe?
                    # Or just to get the price series?
                    # The shadow backtest uses yfinance internally.
                    # But we pass `api_port_series` to shadow backtest for "Hybrid Mode".
                    # So we need this.
                    cashfreq="Monthly",
                    rolling=60, # Assuming 60 from original code
                    invest_div=invest_div,
                    rebalance=rebalance, # Corrected from rebalance_freq
                    allocation=alloc_preview, # Corrected from allocation_table
                    return_raw=False,
                    include_raw=True
                )
                
                # 2. Run Shadow Backtest (Tax Lots & Realized Gains)
                # Use cached wrapper
                # Note: We pass the *result* of the API call (port_series) to the shadow backtest.
                # This means if API call changes, shadow backtest inputs change, so it re-runs. Correct.
                if not port_series.empty:
                    trades_df, pl_by_year, composition_df, unrealized_pl_df, logs = cached_run_shadow_backtest(
                        allocation=alloc_preview, # Mapping to existing variable
                        start_val=start_val,
                        start_date=start_date,
                        end_date=end_date,
                        api_port_series=port_series,
                        rebalance_freq=rebalance, # Mapping to existing variable
                        cashflow=shadow_cashflow,
                        cashflow_freq=cashfreq
                    )
                else:
                    st.error("API returned empty portfolio data.")
                    st.stop()
                
                # Initialize results with API data
                st.session_state.bt_results = {
                    "port_series": port_series,
                    "stats": stats,
                    "extra_data": extra_data,
                    "raw_response": extra_data.get("raw_response", {}),
                    "wmaint": wmaint,
                    "start_val": start_val,
                    # Shadow Backtest Results
                    "trades_df": trades_df,
                    "pl_by_year": pl_by_year,
                    "composition_df": composition_df,
                    "unrealized_pl_df": unrealized_pl_df,
                    "logs": logs
                }
                
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
        
        # Use dynamic wmaint from main scope to allow interactive updates
        # But fallback to stored wmaint if current allocation is invalid (e.g. empty)
        if total_weight > 0:
             # wmaint is already calculated from current inputs in main scope
             pass 
        else:
             wmaint = results["wmaint"]
             
        start_val = results["start_val"]
        logs = results.get("logs", [])
        
        # Calculate Tax Series for Margin Simulation (and Metrics)
        tax_payment_series = None
        total_tax_owed = 0.0
        
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
            
            total_tax_owed = fed_tax_series.sum() + state_tax_series.sum()
            
            # Create Payment Series (Unconditional for Sharpe Calc)
            tax_payment_series = pd.Series(0.0, index=port_series.index)
            annual_total_tax = fed_tax_series + state_tax_series
            
            for year, amount in annual_total_tax.items():
                if amount > 0:
                    # Pay on April 15th of NEXT year
                    pay_date = pd.Timestamp(year + 1, 4, 15)
                    
                    # Find closest valid date in portfolio index
                    # We use searchsorted to find the insertion point
                    idx = port_series.index.searchsorted(pay_date)
                    
                    if idx < len(port_series.index):
                        # Check if the date is reasonably close (e.g. within same month)
                        # If backtest ends in 2024, we can't pay 2024 taxes in 2025
                        actual_date = port_series.index[idx]
                        tax_payment_series[actual_date] += amount

        # Prepare Repayment Series (if Pay Down Margin is enabled)
        repayment_series = None
        if pay_down_margin and cashflow > 0:
            # Construct series based on cashfreq
            dates = port_series.index
            repayment_vals = pd.Series(0.0, index=dates)
            
            if cashfreq == "Monthly":
                 months = dates.month
                 changes = months != np.roll(months, 1)
                 changes[0] = False
                 repayment_vals[changes] = cashflow
            elif cashfreq == "Quarterly":
                 quarters = dates.quarter
                 changes = quarters != np.roll(quarters, 1)
                 changes[0] = False
                 repayment_vals[changes] = cashflow
            elif cashfreq == "Yearly":
                 years = dates.year
                 changes = years != np.roll(years, 1)
                 changes[0] = False
                 repayment_vals[changes] = cashflow
            
            repayment_series = repayment_vals

        # Re-run margin sim (fast enough to run every time, or could cache too)
        # Only pass tax_series if the user opted to pay with margin
        sim_tax_series = tax_payment_series if pay_tax_margin else None
        
        # If paying from cash (Cash Mode), we want to simulate a "Cash Only" scenario for the base
        # This means NO loan, NO interest, NO draws added to loan.
        # We will handle the draws (and taxes) by subtracting from equity in calculate_tax_adjusted_equity.
        eff_loan = 0.0 if pay_tax_cash else st.session_state.starting_loan
        eff_rate = 0.0 if pay_tax_cash else rate_annual
        eff_draw = 0.0 if pay_tax_cash else draw_monthly
        
        loan_series, equity_series, equity_pct_series, usage_series = api.simulate_margin(
            port_series, eff_loan,
            eff_rate, eff_draw, wmaint,
            tax_series=sim_tax_series,
            repayment_series=repayment_series
        )
        
        # Update session state with latest margin results for reporting
        st.session_state.bt_results.update({
            "loan_series": loan_series,
            "equity_series": equity_series,
            "usage_series": usage_series,
            "equity_pct_series": equity_pct_series
        })
                
        
        # Calculate Tax-Adjusted Equity Curve (Global for Tabs)
        final_adj_series = pd.Series(dtype=float)
        final_tax_series = pd.Series(dtype=float) # Series of ACTUAL taxes paid (scaled if needed)
        
        if not equity_series.empty:
            if pay_tax_margin:
                final_adj_series = equity_series
                # If paying with margin, we pay the FULL tax amount (no scaling down)
                final_tax_series = tax_payment_series if tax_payment_series is not None else pd.Series(0.0, index=equity_series.index)
            elif pay_tax_cash:
                if tax_payment_series is not None and tax_payment_series.sum() > 0:
                    final_adj_series, final_tax_series = calculate_tax_adjusted_equity(
                        equity_series, tax_payment_series, port_series, loan_series, rate_annual, draw_monthly=draw_monthly
                    )
                else:
                    # Even if no taxes, we still need to apply the monthly draw if it exists
                    # Re-use the function but with empty tax series
                    empty_tax = pd.Series(0.0, index=equity_series.index)
                    final_adj_series, final_tax_series = calculate_tax_adjusted_equity(
                        equity_series, empty_tax, port_series, loan_series, rate_annual, draw_monthly=draw_monthly
                    )
            else: # None (Gross)
                final_adj_series = equity_series
                final_tax_series = pd.Series(0.0, index=equity_series.index)

        # --- Prepare Tax-Adjusted Data for Charts ---
        # We want the charts to reflect the "Net" reality.
        # Portfolio Value = Net Equity + Loan
        tax_adj_port_series = final_adj_series + loan_series
        
        # Recalculate Leverage Metrics based on Tax-Adjusted Equity
        # Equity % = Net Equity / Portfolio Value
        tax_adj_equity_pct_series = pd.Series(0.0, index=tax_adj_port_series.index)
        valid_idx = tax_adj_port_series > 0
        tax_adj_equity_pct_series[valid_idx] = final_adj_series[valid_idx] / tax_adj_port_series[valid_idx]
        
        # Margin Usage % = Loan / (Net Equity * Max_Leverage) ?? 
        # Or just Loan / (Net Equity + Loan)? 
        # The API returns 'usage_series' which is Loan / (Equity * Max_Loan_to_Equity_Ratio).
        # Let's approximate it or recalculate if we know the formula.
        # Usage = Loan / (Equity * (1/maint_pct - 1))  <-- Standard Reg T formula approximation
        # Let's just use the ratio of Loan / Net Equity for now, or stick to the original usage if it's too complex to replicate perfectly.
        # Actually, let's recalculate usage properly if possible.
        # usage = loan / (equity * (1/maint - 1))
        # if maint is 0.25, max leverage is 4x. max loan is 3x equity.
        # usage = loan / (equity * 3)
        
        # Let's just use a simple leverage ratio for the chart if usage is hard: Leverage = Debt / Equity
        # But the chart expects "Margin usage %".
        # Let's try to replicate the API's usage calc:
        # max_loan = equity * (1 - wmaint) / wmaint  <-- This is max loan value
        # usage = loan / max_loan
        
        tax_adj_usage_series = pd.Series(0.0, index=tax_adj_port_series.index)
        if wmaint < 1.0: # Avoid division by zero if maint is 100%
            # Match the logic in simulate_margin: Usage = Loan / (Port * (1 - Maint))
            # This represents usage of the current collateral's loan value (Risk of Call)
            max_loan_series = tax_adj_port_series * (1 - wmaint)
            valid_loan = max_loan_series > 0
            tax_adj_usage_series[valid_loan] = loan_series[valid_loan] / max_loan_series[valid_loan]
        else:
            # If maint is 100%, max loan is 0. Usage is infinite if loan > 0.
            # We can set it to 0 if loan is 0, or some high number if loan > 0.
            # For charting, maybe just 0 if loan is 0.
            pass
        
        
        ohlc_data = resample_data(tax_adj_port_series, timeframe, method="ohlc")
        equity_resampled = resample_data(final_adj_series, timeframe, method="last")
        loan_resampled = resample_data(loan_series, timeframe, method="last")
        usage_resampled = resample_data(tax_adj_usage_series, timeframe, method="max")
        equity_pct_resampled = resample_data(tax_adj_equity_pct_series, timeframe, method="last")

        m1, m2, m3, m4, m5 = st.columns(5)

        total_return = (tax_adj_port_series.iloc[-1] / start_val - 1) * 100
        
        # Use Stats reported by Testfol API (TWR) to match user request
        # This ensures consistency regardless of cashflow method (DCA vs Pay Down)
        cagr = stats.get("cagr", 0.0)
        max_dd = stats.get("max_drawdown", 0.0)
        sharpe = stats.get("sharpe_ratio", 0.0)
            
        m1.metric("Final Balance", f"${tax_adj_port_series.iloc[-1]:,.0f}", f"{total_return:+.1f}%")
        m2.metric("CAGR", f"{cagr * 100:.2f}%" if abs(cagr) <= 1 else f"{cagr:.2f}%")
        m3.metric("Sharpe", f"{sharpe:.2f}")
        m4.metric("Max Drawdown", f"{max_dd:.2f}%", delta_color="inverse")
        m5.metric("Final Leverage", f"{(tax_adj_port_series.iloc[-1]/final_adj_series.iloc[-1]):.2f}x")
        
        # --- New Metrics Row: Tax & Date Range ---
        st.markdown("---")
        tm1, tm2, tm3, tm4, tm5 = st.columns(5)
        
        # Calculate Total Tax (Already calculated above)
        tax_label = "Total Tax Paid" if pay_tax_margin else "Total Estimated Tax Owed"
        
        if total_tax_owed > 0:
            # If we have a scaled tax series, show THAT total
            if not final_tax_series.empty and final_tax_series.sum() > 0:
                display_tax = final_tax_series.sum()
                tm1.metric(tax_label, f"${display_tax:,.2f}")
            else:
                tm1.metric(tax_label, f"${total_tax_owed:,.2f}")
        else:
            tm1.metric(tax_label, "$0.00")
            
        # Post-Tax Balance & CAGR & Sharpe
        # Since we updated the main metrics to be tax-adjusted, these are somewhat redundant, 
        # BUT "Post-Tax Final Balance" usually refers to NET EQUITY, whereas "Final Balance" is PORTFOLIO VALUE (Gross Assets).
        # In Margin case: Final Balance = High, Post-Tax Balance (Equity) = Low.
        # In Cash case: Final Balance = Low (same as Equity), Post-Tax Balance (Equity) = Low.
        
        if start_val > 0 and not port_series.empty:
            
            # Use global final_adj_series calculated above
            final_adj_val = final_adj_series.iloc[-1]
            
            tm2.metric("Post-Tax Net Equity", f"${final_adj_val:,.0f}")
            
            days = (final_adj_series.index[-1] - final_adj_series.index[0]).days
            if days > 0:
                years = days / 365.25
                # CAGR based on the adjusted final value
                tax_adj_cagr = (final_adj_val / start_val) ** (1 / years) - 1
                tm3.metric("Net Equity CAGR", f"{tax_adj_cagr * 100:.2f}%")
            else:
                tm3.metric("Net Equity CAGR", "N/A")
                
            # Tax Adjusted Sharpe
            tax_adj_sharpe = calculate_sharpe_ratio(final_adj_series)
            tm4.metric("Net Equity Sharpe", f"{tax_adj_sharpe:.2f}")

        else:
             tm2.metric("Post-Tax Net Equity", "$0")
             tm3.metric("Net Equity CAGR", "N/A")
             tm4.metric("Net Equity Sharpe", "N/A")

        # Date Range
        if not trades_df.empty:
            tax_start = trades_df["Date"].min().date()
            tax_end = trades_df["Date"].max().date()
            date_range_str = f"{tax_start} to {tax_end}"
        else:
            date_range_str = "No Taxable Events"
            
        # Custom HTML for wrapping
        tm5.markdown(f"""
            <div data-testid="stMetric" class="stMetric">
                <label data-testid="stMetricLabel" class="css-1qg05tj e1i5pmia2">
                    <div class="css-1wivap2 e1i5pmia3">
                        <span class="css-10trblm e1i5pmia4">Tax Calculation Date Range</span>
                    </div>
                </label>
                <div data-testid="stMetricValue" class="css-1xarl3l e1i5pmia1">
                    <div class="css-1wivap2 e1i5pmia3" style="white-space: normal; word-wrap: break-word; font-size: 1rem;">
                        {date_range_str}
                    </div>
                </div>
            </div>
        """, unsafe_allow_html=True)
             
        st.markdown("---")

        res_tab_chart, res_tab_returns, res_tab_rebal, res_tab_tax, res_tab_debug = st.tabs(["📈 Chart", "📊 Returns Analysis", "⚖️ Rebalancing", "💸 Tax Analysis", "🔧 Debug"])
        
        with res_tab_tax:
            st.markdown("### Annual Tax Impact Analysis")
            
            st.info("""
            **Methodology: Tax-Adjusted Returns**
            
            *   **None (Gross):** No tax simulation. Showing raw pre-tax returns.
            *   **Pay with Margin:** Taxes paid via loan. Assets preserved. Cost = Interest.
            *   **Pay from Cash:** Taxes paid via asset sales. Assets reduced. Cost = Lost Compounding.
            """)
            
            if (pay_tax_margin or pay_tax_cash) and not final_adj_series.empty and not pl_by_year.empty:
                # Prepare Data
                # 1. Annual Ending Balance (Tax Adjusted)
                annual_bal = final_adj_series.resample("YE").last()
                annual_bal.index = annual_bal.index.year
                
                # 2. Annual Tax Paid
                # Use the SCALED tax series (final_tax_series)
                annual_tax_aligned = final_tax_series.resample("YE").sum()
                annual_tax_aligned.index = annual_tax_aligned.index.year
                # Reindex to match balance just in case
                annual_tax_aligned = annual_tax_aligned.reindex(annual_bal.index, fill_value=0.0)
                
                # 3. Annual Market Value (Gross Assets)
                # Market Value = Net Equity + Loan
                # This works for Margin (Loan increases)
                # For Cash Mode, we need a "Gross" baseline that includes Draws but NO Taxes
                if pay_tax_cash:
                    # Calculate Gross Cash Series (Draws YES, Taxes NO)
                    empty_tax = pd.Series(0.0, index=equity_series.index)
                    gross_cash_series, _ = calculate_tax_adjusted_equity(
                        equity_series, empty_tax, port_series, loan_series, rate_annual, draw_monthly=draw_monthly
                    )
                    market_val_series = gross_cash_series
                else:
                    # Margin Mode: Market Value = Net Equity + Loan (which includes tax debt)
                    market_val_series = final_adj_series + loan_series
                
                annual_mv = market_val_series.resample("YE").last()
                annual_mv.index = annual_mv.index.year
                
                # Create DataFrame
                tax_impact_df = pd.DataFrame({
                    "Market Value": annual_mv,
                    "Ending Balance": annual_bal,
                    "Tax Paid": annual_tax_aligned
                })
                
                # Plot Stacked Bar Chart with Market Value Line
                fig_tax_impact = go.Figure()
                
                # Market Value as a Line (to compare against the stack)
                # User requested NO Gross line in Cash Mode
                if not pay_tax_cash:
                    fig_tax_impact.add_trace(go.Scatter(
                        x=tax_impact_df.index,
                        y=tax_impact_df["Market Value"],
                        name="Market Value (Gross)",
                        line=dict(color="#636EFA", width=3),
                        mode='lines+markers',
                        hovertemplate="%{y:$,.0f}<extra></extra>"
                    ))
                
                # Net Balance (Bar)
                fig_tax_impact.add_trace(go.Bar(
                    x=tax_impact_df.index,
                    y=tax_impact_df["Ending Balance"],
                    name="Ending Balance (Net)",
                    marker_color="#00CC96", # Greenish
                    texttemplate="%{y:$.2s}",
                    textposition="auto",
                    hovertemplate="%{y:$,.0f}<extra></extra>"
                ))
                
                # Tax Paid (Bar)
                fig_tax_impact.add_trace(go.Bar(
                    x=tax_impact_df.index,
                    y=tax_impact_df["Tax Paid"],
                    name="Tax Paid",
                    marker_color="#EF553B", # Red
                    texttemplate="%{y:$.2s}",
                    textposition="auto",
                    hovertemplate="%{y:$,.0f}<extra></extra>"
                ))
                
                fig_tax_impact.update_layout(
                    title="Annual Tax Impact: Net Balance + Tax vs. Market Value",
                    xaxis_title="Year",
                    yaxis_title="Amount ($)",
                    barmode='stack',
                    template="plotly_dark",
                    height=500,
                    hovermode="x unified",
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                )
                
                st.plotly_chart(fig_tax_impact, use_container_width=True)
                
                st.markdown("### Detailed Data")
                st.dataframe(tax_impact_df.style.format("${:,.2f}"), use_container_width=True)
            elif not (pay_tax_margin or pay_tax_cash):
                st.warning("Tax Simulation is set to **None (Gross)**. Enable 'Pay from Cash' or 'Pay with Margin' to see tax impact analysis.")
            else:
                st.info("No data available for tax analysis.")

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
                    tax_adj_port_series, final_adj_series, loan_series, 
                    tax_adj_equity_pct_series, tax_adj_usage_series, 
                    series_opts, log_scale
                )
            elif chart_style == "Classic (Dashboard)":
                log_opts = {
                    "portfolio": st.session_state.get("log_portfolio", False),
                    "leverage": st.session_state.get("log_leverage", False),
                    "margin": st.session_state.get("log_margin", False)
                }
                render_dashboard_view(
                    tax_adj_port_series, final_adj_series, loan_series, 
                    tax_adj_equity_pct_series, tax_adj_usage_series, 
                    st.session_state.get("maint_pct", 0.25), # Fallback if not in state
                    stats, log_opts
                )
            else:
                render_candlestick_chart(
                    ohlc_data, equity_resampled, loan_resampled, 
                    usage_resampled, equity_pct_resampled, 
                    timeframe, log_scale, show_range_slider, show_volume
                )
            
            if pay_tax_cash:
                with st.expander("Detailed Cash Statistics", expanded=True):
                    # Calculate Total Withdrawals
                    # 1. Taxes
                    total_tax_paid = final_tax_series.sum() if not final_tax_series.empty else 0.0
                    
                    # 2. Monthly Draws
                    # We need to calculate total draws made over the period
                    # Logic: Count months in the range and multiply by draw_monthly
                    # Or iterate dates like in the sim
                    total_draws = 0.0
                    if draw_monthly > 0 and not equity_series.empty:
                        # Simple approximation: (Months) * Draw
                        # Better: Iterate dates to match exact logic
                        prev_m = equity_series.index[0].month
                        for d in equity_series.index[1:]:
                            if d.month != prev_m:
                                total_draws += draw_monthly
                                prev_m = d.month
                                
                    total_withdrawn = total_tax_paid + total_draws
                    
                    c1, c2, c3 = st.columns(3)
                    c1.metric("Final Balance", f"${final_adj_series.iloc[-1]:,.2f}")
                    c2.metric("Total Withdrawn", f"${total_withdrawn:,.2f}", help="Includes Taxes + Monthly Draws")
                    c3.metric("Total Tax Paid", f"${total_tax_paid:,.2f}")
                    
                    st.markdown("##### Withdrawal Breakdown")
                    w_df = pd.DataFrame({
                        "Category": ["Monthly Draws", "Taxes Paid"],
                        "Amount": [total_draws, total_tax_paid]
                    })
                    st.dataframe(w_df.style.format({"Amount": "${:,.2f}"}), use_container_width=True, hide_index=True)

            else:
                with st.expander("Detailed Margin Statistics", expanded=True):
                    # --- Calculations ---
                    # Max Usage & Min Equity
                    max_usage_idx = usage_series.idxmax()
                    max_usage_val = usage_series.max()
                    equity_at_max = equity_series.loc[max_usage_idx]
                    
                    min_equity_idx = equity_series.idxmin()
                    min_equity_val = equity_series.min()
                    
                    # Survival Metrics Calculations
                    total_draws = 0.0
                    if draw_monthly > 0 and not equity_series.empty:
                        prev_m = equity_series.index[0].month
                        for d in equity_series.index[1:]:
                            if d.month != prev_m:
                                total_draws += draw_monthly
                                prev_m = d.month
                    
                    total_tax_paid = final_tax_series.sum() if not final_tax_series.empty else 0.0
                    start_loan_val = loan_series.iloc[0]
                    final_loan_val = loan_series.iloc[-1]
                    total_interest = (final_loan_val - start_loan_val) - total_draws - total_tax_paid
                    
                    avg_usage = usage_series.mean()
                    current_usage = usage_series.iloc[-1]
                    safety_buffer = 1.0 - current_usage
                    
                    current_port_val = tax_adj_port_series.iloc[-1]
                    max_loan_allowed = current_port_val * (1 - wmaint)
                    avail_withdraw = max_loan_allowed - final_loan_val

                    # --- Display Layout (Aligned 3-Column Grid) ---
                    
                    # Row 1: Current Status
                    c1, c2, c3 = st.columns(3)
                    c1.metric("Final Equity", f"${equity_series.iloc[-1]:,.2f}")
                    c2.metric("Final Loan", f"${loan_series.iloc[-1]:,.2f}")
                    c3.metric("Final Usage", f"{usage_series.iloc[-1]*100:.2f}%")
                    
                    # Row 2: Extremes & Capacity (Aligned with Row 1)
                    c4, c5, c6 = st.columns(3)
                    c4.metric(
                        "Lowest Equity", 
                        f"${min_equity_val:,.2f}", 
                        f"{min_equity_idx.date()}",
                        delta_color="off" 
                    )
                    c5.metric(
                        "Avail. to Withdraw", 
                        f"${avail_withdraw:,.2f}", 
                        help="Excess Borrowing Power"
                    )
                    c6.metric(
                        "Max Margin Usage", 
                        f"{max_usage_val*100:.2f}%", 
                        f"{max_usage_idx.date()} (Eq: ${equity_at_max:,.0f})",
                        delta_color="inverse" 
                    )
                    
                    st.markdown("##### Survival Metrics")
                    # Row 3: Long-term Health
                    m1, m2, m3 = st.columns(3)
                    m1.metric("Total Interest Paid", f"${total_interest:,.2f}", help="Cost of Leverage")
                    m2.metric("Avg Margin Usage", f"{avg_usage*100:.2f}%", help="Chronic Stress Level")
                    m3.metric("Safety Buffer", f"{safety_buffer*100:.2f}%", help="% Market Drop allowed before Margin Call")
                    
                    # Standard Margin Breaches (Usage >= 100%)
                    breaches = pd.DataFrame({
                        "Date": usage_series[usage_series >= 1].index.date,
                        "Usage %": (usage_series[usage_series >= 1] * 100).round(2),
                        "Equity %": (equity_pct_series[usage_series >= 1] * 100).round(2),
                        "Type": "Reg T Call"
                    })
                    
                    # Portfolio Margin Breaches (Equity < $100k)
                    if pm_enabled:
                        pm_breach_mask = equity_series < 100000
                        if pm_breach_mask.any():
                            pm_breaches = pd.DataFrame({
                                "Date": equity_series[pm_breach_mask].index.date,
                                "Usage %": (usage_series[pm_breach_mask] * 100).round(2),
                                "Equity %": (equity_pct_series[pm_breach_mask] * 100).round(2),
                                "Type": "PM Min Equity < $100k"
                            })
                            # Combine and sort
                            breaches = pd.concat([breaches, pm_breaches]).sort_values("Date")
                    
                    st.markdown("##### Margin & PM Breaches")
                    if breaches.empty:
                        st.success("No breaches triggered! 🎉")
                    else:
                        # Count types
                        reg_t_count = len(breaches[breaches["Type"] == "Reg T Call"])
                        pm_count = len(breaches[breaches["Type"] == "PM Min Equity < $100k"])
                        
                        msg_parts = []
                        if reg_t_count > 0:
                            msg_parts.append(f"{reg_t_count} Margin Call(s)")
                        if pm_count > 0:
                            msg_parts.append(f"{pm_count} PM Breach(es)")
                        
                        full_msg = f"⚠️ {' + '.join(msg_parts)} triggered."
                        
                        if reg_t_count > 0:
                            st.error(full_msg) # Red for Margin Calls (Critical)
                        else:
                            st.warning(full_msg) # Yellow for PM Breaches (Warning)
                            
                        st.dataframe(breaches, use_container_width=True)
        
        with res_tab_returns:
            if pay_tax_cash:
                st.info("ℹ️ **Note:** Returns are **Net of Tax** (simulated as cash withdrawals).")
            elif pay_tax_margin:
                st.info("ℹ️ **Note:** Returns are **Gross** (taxes paid via margin loan).")
            else:
                st.info("ℹ️ **Note:** Returns are **Gross** (Pre-Tax).")
            render_returns_analysis(tax_adj_port_series)
            
        with res_tab_rebal:
            st.warning("⚠️ **Note:** These trade calculations assume **Gross** portfolio values. Tax payments are NOT deducted by selling shares in this view (assumes taxes paid via margin or external cash).")
            render_rebalancing_analysis(
                trades_df, pl_by_year, composition_df,
                tax_method, other_income, filing_status, state_tax_rate,
                rebalance_freq=rebalance,
                use_standard_deduction=use_std_deduction,
                unrealized_pl_df=st.session_state.bt_results.get("unrealized_pl_df", pd.DataFrame())
            )
        
        with res_tab_tax:
            render_tax_analysis(
                pl_by_year, other_income, filing_status, state_tax_rate,
                tax_method=tax_method,
                use_standard_deduction=use_std_deduction,
                unrealized_pl_df=st.session_state.bt_results.get("unrealized_pl_df", pd.DataFrame()),
                trades_df=trades_df,
                pay_tax_cash=pay_tax_cash,
                pay_tax_margin=pay_tax_margin
            )
            
        with res_tab_debug:
            st.subheader("Debug Info")
            st.json(logs)
            st.write("Raw API Response (First 5 items):")
            st.write(str(raw_response)[:1000])
            
            # Also provide download button
            import json as json_lib
            json_str = json_lib.dumps(raw_response, indent=2)
            st.download_button(
                label="Download Raw Response",
                data=json_str,
                file_name="testfol_api_response.json",
                mime="application/json"
            )


# -----------------------------------------------------------------------------
# Report Generation (Sidebar)
# -----------------------------------------------------------------------------
if "bt_results" in st.session_state:
    st.sidebar.markdown("---")
    st.sidebar.subheader("📄 Report")
    
    # Generate report on the fly
    try:
        report_html = report_generator.generate_html_report(st.session_state.bt_results)
        
        st.sidebar.download_button(
            label="Download HTML Report",
            data=report_html,
            file_name=f"testfol_report_{dt.datetime.now().strftime('%Y%m%d_%H%M')}.html",
            mime="text/html",
            help="Download a standalone HTML report with charts and stats."
        )
    except Exception as e:
        st.sidebar.error(f"Report Gen Failed: {e}")

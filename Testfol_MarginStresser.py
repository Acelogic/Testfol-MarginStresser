#!/usr/bin/env python3
# ─────────────────────────────────────────────────────────────────────────────
# testfol_gui_v4.py
#
# Enhanced Streamlit GUI for Testfol with dashboard view option
# ─────────────────────────────────────────────────────────────────────────────

import datetime as dt
import json
import requests

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

API_URL = "https://testfol.io/api/backtest"

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

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
    # Compute equity % based on starting loan
    st.session_state.equity_init = 100 * max(0, 1 - loan / sv)

def sync_loan():
    sv = st.session_state.start_val
    eq = st.session_state.equity_init
    # Compute starting loan based on equity %
    st.session_state.starting_loan = sv * max(0, 1 - eq / 100)

def handle_presets(key="alloc_df"):
    st.header("Portfolio presets")
    up = st.file_uploader("Load preset (JSON)", type=["json"])
    if up:
        try:
            df = pd.read_json(up)
            if set(df.columns) >= {"Ticker","Weight %","Maint %"}:
                st.session_state[key] = df
                st.success("Loaded preset")
            else:
                st.error("Invalid preset format")
        except Exception:
            st.error("Failed to read preset")
    if key in st.session_state:
        st.download_button(
            "Save preset",
            data=st.session_state[key].to_json(orient="records"),
            file_name="portfolio_preset.json",
            mime="application/json"
        )

def table_to_dicts(df: pd.DataFrame):
    df = df.dropna(subset=["Ticker"]).copy()
    alloc = {r["Ticker"].strip(): float(r["Weight %"]) for _,r in df.iterrows()}
    maint = {r["Ticker"].split("?")[0].strip(): float(r["Maint %"]) for _,r in df.iterrows()}
    return alloc, maint

def fetch_backtest(start_date, end_date, start_val, cashflow, cashfreq, rolling,
                   invest_div, rebalance, allocation):
    payload = {
        "start_date": str(start_date),
        "end_date":   str(end_date),
        "start_val":  start_val,
        "adj_inflation": False,
        "cashflow": cashflow,
        "cashflow_freq": cashfreq,
        "rolling_window": rolling,
        "backtests": [{
            "invest_dividends": invest_div,
            "rebalance_freq":   rebalance,
            "allocation":       allocation,
            "drag": 0,
            "absolute_dev": 0,
            "relative_dev": 0
        }]
    }
    r = requests.post(API_URL, json=payload, timeout=30)
    r.raise_for_status()
    resp = r.json()
    stats = resp.get("stats", {})
    if isinstance(stats, list):
        stats = stats[0] if stats else {}
    ts, vals = resp["charts"]["history"]
    dates = pd.to_datetime(ts, unit="s")
    return pd.Series(vals, index=dates, name="Portfolio"), stats

def simulate_margin(port, starting_loan, rate_annual, draw_monthly, maint_pct):
    rate_daily = (rate_annual / 100) / 252
    loan_vals, loan = [], starting_loan
    prev_m = port.index[0].month
    for d in port.index:
        loan *= 1 + rate_daily
        if draw_monthly and d.month != prev_m:
            loan += draw_monthly
            prev_m = d.month
        loan_vals.append(loan)
    loan_series = pd.Series(loan_vals, index=port.index, name="Loan")
    equity = port - loan_series
    equity_pct = (equity / port).rename("Equity %")
    usage_pct = (loan_series / (port * (1 - maint_pct))).rename("Margin usage %")
    return loan_series, equity, equity_pct, usage_pct

def render_chart(port, equity, loan, equity_pct, usage_pct, series_opts, log_scale):
    fig = go.Figure()
    TRACES = {
        "Portfolio":       (port.index, port, {"width":2}, "$%{y:,.0f}"),
        "Equity":          (equity.index, equity, {"dash":"dot"}, "$%{y:,.0f}"),
        "Loan":            (loan.index, loan, {"dash":"dot","width":1,"color":"lime"}, "$%{y:,.0f}"),
        "Margin usage %":  (usage_pct.index, usage_pct*100, {"width":2,"color":"yellow"}, "%{y:.2f}%"),
        "Equity %":        (equity_pct.index, equity_pct*100, {"dash":"dash"}, "%{y:.2f}%"),
    }
    for key in series_opts:
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

def render_dashboard(port, equity, loan, equity_pct, usage_pct, maint_pct, stats):
    """Render dashboard-style separate charts"""
    
    # Get log scale settings
    log_portfolio = st.session_state.get("log_portfolio", False)
    log_leverage = st.session_state.get("log_leverage", False)
    log_margin = st.session_state.get("log_margin", False)
    
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

def show_summary(port, equity, loan, usage_pct, stats):
    st.subheader("Summary statistics")

    # Row 1: Final outcomes
    port_metrics = [
        ("Final portfolio", port.iloc[-1], "$"),
        ("Final equity",   equity.iloc[-1], "$"),
        ("Final loan",     loan.iloc[-1], "$"),
        ("Final usage %",  usage_pct.iloc[-1]*100, "%")
    ]
    cols1 = st.columns(len(port_metrics))
    for col, (label, val, suf) in zip(cols1, port_metrics):
        text = f"{val:.2f}%" if suf == "%" else f"${val:,.2f}"
        col.metric(label, text)

    st.markdown("---")

    # Row 2: Back‐test statistics
    stat_metrics = [
        ("CAGR",           stats.get("cagr"),         "%"),
        ("Sharpe ratio",   stats.get("sharpe_ratio") or stats.get("sharpe"), ""),
        ("Max drawdown",   stats.get("max_drawdown"), "%")
    ]
    cols2 = st.columns(len(stat_metrics))
    for col, (label, val, suf) in zip(cols2, stat_metrics):
        if val is None:
            col.metric(label, "N/A")
        else:
            if suf == "%":
                pct = val * 100 if abs(val) <= 1 else val
                text = f"{pct:.2f}%"
            else:
                text = f"{val:.2f}"
            col.metric(label, text)


# ─────────────────────────────────────────────────────────────────────────────
# Main layout
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="Testfol API  |  Margin Simulator", layout="wide")
st.title("Testfol API  |  Margin Simulator")

with st.sidebar:
    st.header("Global parameters")
    c1, c2 = st.columns(2)
    start_date = c1.date_input(
        "Start date",
        value=dt.date(2012,1,1),
        min_value=dt.date(1885,1,1),
        max_value=dt.date.today()
    )
    end_date   = c2.date_input(
        "End date",
        value=dt.date.today(),
        min_value=dt.date(1885,1,1),
        max_value=dt.date.today()
    )
    start_val  = num_input(
        "Starting value", "start_val", 10000, 1000,
        on_change=sync_equity
    )
    rolling    = num_input(
        "Rolling window (months)", "rolling", 60, 1
    )
    cashflow   = num_input("Cash-flow", "cashflow", 0, 100)
    cashfreq   = st.selectbox(
        "Cash-flow frequency",
        ["Yearly","Quarterly","Monthly"]
    )
    st.divider()

    st.header("Rebalance & dividends")
    invest_div = st.checkbox("Re-invest dividends", value=True)
    rebalance  = st.selectbox(
        "Rebalance frequency",
        ["Yearly","Quarterly","Monthly"]
    )
    st.divider()

    st.header("Financing / margin")
    starting_loan = num_input(
        "Starting loan ($)", "starting_loan", 0.0, 100.0,
        on_change=sync_equity
    )
    equity_init   = num_input(
        "Initial equity %  (100=no margin)",
        "equity_init", 100.0, 1.0,
        on_change=sync_loan
    )
    st.markdown(
        f"**Loan:** ${st.session_state.starting_loan:,.2f}  —  "
        f"**Equity %:** {st.session_state.equity_init:.2f}%"
    )
    rate_annual  = num_input(
        "Interest % per year", "rate_annual", 8.0, 0.5
    )
    draw_monthly = num_input(
        "Monthly margin draw ($)", "draw_monthly", 0.0, 500.0,
        help="Borrow this amount on the 1st of each month"
    )
    st.divider()

    st.header("Chart options")
    view_mode = st.radio(
        "View mode",
        ["Combined Chart", "Dashboard View"],
        help="Choose between single chart or dashboard-style separate charts"
    )
    
    if view_mode == "Combined Chart":
        series_opts = st.multiselect(
            "Show series",
            ["Portfolio","Equity","Loan","Margin usage %","Equity %"],
            default=["Portfolio","Equity","Loan","Margin usage %","Equity %"]
        )
        log_scale = st.checkbox("Log scale (left axis)", value=False)
    else:
        st.markdown("**Dashboard log scales:**")
        log_portfolio = st.checkbox("Portfolio chart", value=False, key="log_portfolio")
        log_leverage = st.checkbox("Leverage chart", value=False, key="log_leverage")
        log_margin = st.checkbox("Margin debt chart", value=False, key="log_margin")
    st.divider()

    handle_presets()

st.subheader("Portfolio allocation + per-ticker maintenance")
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

# Render editable table
edited_df = st.data_editor(
    st.session_state.alloc_df,
    num_rows="dynamic",
    column_order=["Ticker","Weight %","Maint %"],
    column_config={
        "Weight %": st.column_config.NumberColumn(
            min_value=0.0, max_value=100.0, step=0.01, format="%.2f"
        ),
        "Maint %": st.column_config.NumberColumn(
            min_value=0.0, max_value=100.0, step=0.1, format="%.1f"
        ),
    },
    use_container_width=True,
    key="alloc_table"
)

# Use the edited dataframe for calculations
working_df = edited_df.dropna(subset=["Ticker"]).loc[lambda df: df["Ticker"].str.strip() != ""]

default_maint = num_input(
    "Default maintenance % for tickers not listed above",
    "default_maint", 25.0, 1.0
)

# Use working_df for preview calculations
alloc_preview, maint_preview = table_to_dicts(working_df)
if round(sum(alloc_preview.values()), 2) == 100:
    st.metric("Starting loan", f"${st.session_state.starting_loan:,.2f}")
    wmaint = sum(
        (wt/100) * (maint_preview.get(t.split("?")[0], default_maint)/100)
        for t, wt in alloc_preview.items()
    )
    st.metric("Weighted maint %", f"{wmaint*100:.2f}%")
else:
    st.info(f"Weights sum to {sum(alloc_preview.values()):.2f}%, must be 100% for preview")

if st.button("Run back-test", type="primary"):
    if round(sum(alloc_preview.values()), 2) != 100:
        st.error("Weights must sum to 100%.")
        st.stop()

    port, stats = fetch_backtest(
        start_date, end_date,
        st.session_state.start_val,
        cashflow, cashfreq,
        rolling, invest_div,
        rebalance, alloc_preview
    )
    maint_pct = sum(
        (wt/100) * (maint_preview.get(t.split("?")[0], default_maint)/100)
        for t, wt in alloc_preview.items()
    )
    loan_series, equity, equity_pct, usage_pct = simulate_margin(
        port, st.session_state.starting_loan,
        rate_annual, draw_monthly, maint_pct
    )
    
    # Render based on selected view mode
    if view_mode == "Combined Chart":
        render_chart(port, equity, loan_series, equity_pct, usage_pct,
                     series_opts, log_scale)
        show_summary(port, equity, loan_series, usage_pct, stats)
    else:
        render_dashboard(port, equity, loan_series, equity_pct, usage_pct, maint_pct, stats)

    # Margin breaches table (shown in both views)
    breaches = pd.DataFrame({
        "Date":     usage_pct[usage_pct>=1].index.date,
        "Usage %":  (usage_pct[usage_pct>=1]*100).round(1),
        "Equity %": (equity_pct[usage_pct>=1]*100).round(1)
    })
    st.subheader("Maintenance breaches")
    if breaches.empty:
        st.success("No margin calls 🎉")
    else:
        st.warning(f"⚠️ {len(breaches)} breach day(s)")
        st.dataframe(breaches, hide_index=True, use_container_width=True)
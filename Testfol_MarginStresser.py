#!/usr/bin/env python3
# ─────────────────────────────────────────────────────────────────────────────
# testfol_gui_v3.py
#
# Streamlit GUI for Testfol back-tests + dynamic margin simulation
# • Editable table: Ticker | Weight % | Maint %
# • Margin model:
#       – initial equity % (cash vs. loan on day-0)
#       – annual margin interest (daily compounding, 252 days/yr)
#       – optional fixed monthly margin draw ($) – live-off-portfolio
# • Calculates each trading day:
#       – equity %, margin-usage % = loan / [ P × (1 − Maint %) ]
# • Flags breaches and plots them.
# • Chart features:
#       – choose which series to plot
#       – linear/log scale, unified hover, centered legend
# • Preset management: save & load portfolio allocations
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

def show_summary(port, equity, loan, usage_pct, stats):
    st.subheader("Summary statistics")

    # ─── Row 1: Final outcomes ────────────────────────────────────────────────
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

    # ─── Row 2: Back‐test statistics ───────────────────────────────────────────
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
    series_opts = st.multiselect(
        "Show series",
        ["Portfolio","Equity","Loan","Margin usage %","Equity %"],
        default=["Portfolio","Equity","Loan","Margin usage %","Equity %"]
    )
    log_scale = st.checkbox("Log scale (left axis)", value=False)
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

# Render editable table - let st.data_editor handle its own state
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
    render_chart(port, equity, loan_series, equity_pct, usage_pct,
                 series_opts, log_scale)
    show_summary(port, equity, loan_series, usage_pct, stats)

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
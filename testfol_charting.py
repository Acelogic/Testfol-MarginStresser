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

def render_charts(ohlc_df, equity_series, loan_series, usage_series, equity_pct_series, timeframe, log_scale):
    title_map = {
        "1D": "Daily",
        "1W": "Weekly",
        "1M": "Monthly",
        "3M": "Quarterly",
        "1Y": "Yearly"
    }
    
    fig = go.Figure()

    hover_text = [
        f"Date: {d:%b %d, %Y}<br>O: {o:,.2f}<br>H: {h:,.2f}<br>L: {l:,.2f}<br>C: {c:,.2f}"
        for d, o, h, l, c in zip(ohlc_df.index, ohlc_df["Open"], ohlc_df["High"], ohlc_df["Low"], ohlc_df["Close"])
    ]

    fig.add_trace(go.Candlestick(
        x=ohlc_df.index,
        open=ohlc_df["Open"],
        high=ohlc_df["High"],
        low=ohlc_df["Low"],
        close=ohlc_df["Close"],
        name="Portfolio",
        text=hover_text,
        hoverinfo="text"
    ))

    fig.add_trace(go.Scatter(
        x=equity_series.index,
        y=equity_series,
        name="Equity",
        line=dict(color="#1DB954", width=2, dash="dot"),
        hovertemplate="$%{y:,.0f}<extra></extra>"
    ))

    fig.add_trace(go.Scatter(
        x=loan_series.index,
        y=loan_series,
        name="Loan",
        line=dict(color="#FF6B6B", width=1, dash="dot"),
        hovertemplate="$%{y:,.0f}<extra></extra>"
    ))

    fig.update_layout(
        title=dict(text=f"Portfolio Performance ({title_map.get(timeframe, timeframe)})", font=dict(size=20)),
        yaxis_title="Value ($)",
        xaxis_title="Date",
        template="plotly_dark",
        yaxis_type="log" if log_scale else "linear",
        xaxis_rangeslider_visible=False,
        height=550,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=50, r=50, t=80, b=50)
    )
    st.plotly_chart(fig, use_container_width=True)

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
        
        fig = go.Figure(data=go.Heatmap(
            z=pivot.values * 100,
            x=month_names,
            y=pivot.index,
            colorscale="RdYlGn",
            zmid=0,
            texttemplate="%{z:+.1f}%",
            hovertemplate="%{z:+.1f}%<extra></extra>",
            xgap=1, ygap=1
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
    run_placeholder = st.empty()
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
        cashfreq = st.selectbox("Frequency", ["Yearly", "Quarterly", "Monthly"], index=2)
        invest_div = st.checkbox("Re-invest Dividends", value=True)
        rebalance = st.selectbox("Rebalance", ["Yearly", "Quarterly", "Monthly"], index=0)

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
        timeframe = st.selectbox(
            "Chart Timeframe",
            ["1D", "1W", "1M", "3M", "1Y"],
            index=2
        )
    with c2:
        log_scale = st.checkbox("Logarithmic Scale", value=True)

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
                port_series, stats = api.fetch_backtest(
                    start_date, end_date, start_val,
                    cashflow, cashfreq, 60,
                    invest_div, rebalance, alloc_preview
                )
                
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
                
                m1.metric("Final Balance", f"${port_series.iloc[-1]:,.0f}", f"{total_return:+.1f}%")
                m2.metric("CAGR", f"{calculate_cagr(port_series)*100:.2f}%")
                m3.metric("Sharpe", f"{stats.get('sharpe_ratio', 0):.2f}")
                m4.metric("Max Drawdown", f"{calculate_max_drawdown(port_series)*100:.2f}%", delta_color="inverse")
                m5.metric("Final Leverage", f"{(port_series.iloc[-1]/equity_series.iloc[-1]):.2f}x")

                res_tab_chart, res_tab_returns = st.tabs(["📈 Chart", "📊 Returns Analysis"])
                
                with res_tab_chart:
                    render_charts(
                        ohlc_data, equity_resampled, loan_resampled, 
                        usage_resampled, equity_pct_resampled, 
                        timeframe, log_scale
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
                
            except Exception as e:
                st.error(f"Error running backtest: {str(e)}")

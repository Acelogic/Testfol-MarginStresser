"""Monte Carlo simulation tab rendering."""
from __future__ import annotations

import csv
import os

import numpy as np
import plotly.graph_objects as go
import streamlit as st
import pandas as pd

from app.core import monte_carlo
from app.ui import charts


def render_monte_carlo_tab(
    tab,
    results: dict,
    config: dict,
    portfolio_name: str,
    daily_rets: pd.Series,
    source_label: str,
) -> None:
    with tab:
        st.markdown("### 🔮 Monte Carlo Simulation (Historical Bootstrap)")
        st.info("Simulating **10-year future performance** based on your strategy's historical daily volatility. Assumes reinvestment of all returns.")

        if daily_rets.empty:
            st.error("No return data available for Monte Carlo.")
            return

        st.caption(f"ℹ️ Using **{source_label}**.")

        # 2. Configuration
        c_sims, c_start, c_flow = st.columns(3)
        n_sims = c_sims.slider("Scenarios", 100, 5000, 1000, 100, help="More scenarios = smoother cone", key=f"mc_n_sims_{portfolio_name}")

        def_start = results.get("start_val", 10000.0)
        sim_start = c_start.number_input("Start Value ($)", value=float(def_start), step=1000.0, key=f"mc_start_{portfolio_name}")

        cf_amt = results.get("cashflow", 0.0)
        cf_freq = results.get("cashfreq", "None")
        def_monthly = 0.0
        if cf_freq == 'Monthly': def_monthly = cf_amt
        elif cf_freq == 'Quarterly': def_monthly = cf_amt / 3
        elif cf_freq == 'Yearly': def_monthly = cf_amt / 12

        sim_monthly_add = c_flow.number_input("Monthly Add ($)", value=float(def_monthly), step=100.0, help="Monthly contribution injected into simulation", key=f"mc_monthly_{portfolio_name}")

        # Advanced Settings
        custom_mean = None
        custom_vol = None
        filter_start = None
        filter_end = None
        block_size = 1

        with st.expander("⚙️ Advanced Settings (Regimes & Scenarios)", expanded=False):
            mc_mode = st.radio(
                "Source Data / Regime",
                ["Full History (Default)", "Historical Period Filter", "Stress Scenario", "Custom Parameters"],
                help="Choose how to generate future return paths.",
                key=f"mc_mode_{portfolio_name}"
            )

            if mc_mode == "Historical Period Filter":
                min_date = daily_rets.index.min().date()
                max_date = daily_rets.index.max().date()

                c_f1, c_f2 = st.columns(2)
                filter_start = c_f1.date_input("From", value=max(min_date, pd.to_datetime("2020-01-01").date()), key=f"mc_date_from_{portfolio_name}")
                filter_end = c_f2.date_input("To", value=max_date, key=f"mc_date_to_{portfolio_name}")

            elif mc_mode == "Stress Scenario":
                scenario = st.selectbox(
                    "Select Historical Scenario",
                    ["1970s Stagflation (1973-1982)",
                     "2000 DotCom Bust (2000-2002)",
                     "2008 GFC (2007-2009)",
                     "2020 COVID Crash (Feb-Apr 2020)",
                     "2022 Inflation/Rates (2022)"],
                    key=f"mc_stress_scenario_{portfolio_name}"
                )

                if "1970s" in scenario:
                    filter_start = "1973-01-01"
                    filter_end = "1982-12-31"
                    st.caption("High Inflation, Rising Rates, Poor Real Returns.")
                elif "2000" in scenario:
                    filter_start = "2000-03-01"
                    filter_end = "2002-10-01"
                    st.caption("Tech Bubble Burst, Prolonged Bear Market.")
                elif "2008" in scenario:
                    filter_start = "2007-10-01"
                    filter_end = "2009-03-09"
                    st.caption("Systemic Financial Crisis, DEFLATIONARY shock.")
                elif "2020" in scenario:
                    filter_start = "2020-02-19"
                    filter_end = "2020-04-30"
                    st.caption("Sudden Pandemic Shock & rapid V-shaped recovery.")
                elif "2022" in scenario:
                    filter_start = "2022-01-01"
                    filter_end = "2022-12-31"
                    st.caption("Correlated Bond/Stock selloff due to Rate Hikes.")

                min_avail = daily_rets.index.min().date()
                if pd.to_datetime(filter_start).date() < min_avail:
                    st.warning(f"⚠️ Your backtest data starts on {min_avail}. The selected scenario starts earlier ({filter_start}). The simulation will only use available data.")

            elif mc_mode == "Custom Parameters":
                c_p1, c_p2 = st.columns(2)
                custom_mean = c_p1.number_input("Expected Annual Return (%)", value=7.0, step=0.5, key=f"mc_custom_return_{portfolio_name}") / 100.0
                custom_vol = c_p2.number_input("Expected Annual Volatility (%)", value=15.0, step=0.5, key=f"mc_custom_vol_{portfolio_name}") / 100.0
                st.info("Generates synthetic returns using Normal Distribution (IID). Ignores historical data patterns.")

            if mc_mode != "Custom Parameters":
                st.markdown("##### Sampling Method")
                boot_method = st.radio("Method", ["Simple Bootstrap (IID)", "Block Bootstrap"], horizontal=True, key=f"mc_boot_method_{portfolio_name}")
                if boot_method == "Block Bootstrap":
                    block_size = st.slider("Block Size (Days)", min_value=5, max_value=60, value=20, help="Larger blocks preserve longer-term market memory (volatility clustering).", key=f"mc_block_size_{portfolio_name}")
                    st.caption(f"Sampling contiguous blocks of {block_size} days.")

        # --- TABS: Standard vs Seasonal ---
        mc_tab_std, mc_tab_seas = st.tabs(["Standard Simulation (10 Yr)", "📅 Seasonal Analysis (1 Yr)"])

        with mc_tab_std:
            with st.spinner(f"Running {n_sims:,} Simulations..."):
                mc_results = monte_carlo.run_monte_carlo(
                    daily_rets,
                    n_sims=n_sims,
                    n_years=10,
                    initial_val=sim_start,
                    monthly_cashflow=sim_monthly_add,
                    filter_start_date=filter_start,
                    filter_end_date=filter_end,
                    custom_mean_annual=custom_mean,
                    custom_vol_annual=custom_vol,
                    block_size=block_size
                )

            if mc_results:
                charts.render_monte_carlo_view(mc_results, unique_id=portfolio_name)

        with mc_tab_seas:
            _render_seasonal_tab(daily_rets, sim_start, sim_monthly_add, portfolio_name)

        # Log detailed scenario results
        if mc_results:
            _log_mc_to_csv(mc_results, n_sims)


def _render_seasonal_tab(
    daily_rets: pd.Series,
    sim_start: float,
    sim_monthly_add: float,
    portfolio_name: str,
) -> None:
    st.markdown("### 📅 Typical Year Analysis (Seasonal Bootstrap)")
    st.info("This simulation builds a 'Typical Year' by sampling January returns only from historical Januaries, February entries from Februaries, etc. This reveals seasonal patterns like 'Sell in May' or 'Santa Rally'.")

    if st.button("Run Seasonal Analysis (5,000 Runs)", key=f"mc_run_seasonal_{portfolio_name}"):
        with st.spinner("Analyzing Seasonality..."):
            seas_df = monte_carlo.run_seasonal_monte_carlo(daily_rets, n_sims=5000, initial_val=sim_start, monthly_cashflow=sim_monthly_add)

            if not seas_df.empty:
                fig = go.Figure()

                x_axis = pd.bdate_range(start='2024-01-01', periods=len(seas_df))

                p90_vals = seas_df["P90"]
                p75_vals = seas_df["P75"]
                p25_vals = seas_df["P25"]
                p10_vals = seas_df["P10"]
                median_vals = seas_df["Median"]

                fig.add_trace(go.Scatter(
                    x=x_axis, y=p10_vals,
                    mode='lines',
                    line=dict(color='rgba(255, 50, 50, 0.5)', width=1, dash='dash'),
                    name='P10 (Pessimistic)',
                    hovertemplate='<b>P10</b>: $%{y:,.0f}<extra></extra>'
                ))

                fig.add_trace(go.Scatter(
                    x=x_axis, y=p25_vals,
                    mode='lines',
                    line=dict(color='rgba(255, 165, 0, 0.5)', width=1, dash='dot'),
                    name='P25 (Mod. Downside)',
                    hovertemplate='<b>P25</b>: $%{y:,.0f}<extra></extra>'
                ))

                fig.add_trace(go.Scatter(
                    x=x_axis, y=median_vals,
                    mode='lines',
                    line=dict(color='#00C8FF', width=3),
                    name='Median',
                    hovertemplate='<b>Median</b>: $%{y:,.0f}<extra></extra>'
                ))

                fig.add_trace(go.Scatter(
                    x=x_axis, y=p75_vals,
                    mode='lines',
                    line=dict(color='rgba(200, 200, 200, 0.5)', width=1, dash='dot'),
                    name='P75 (Mod. Upside)',
                    hovertemplate='<b>P75</b>: $%{y:,.0f}<extra></extra>'
                ))

                fig.add_trace(go.Scatter(
                    x=x_axis, y=p90_vals,
                    mode='lines',
                    line=dict(color='rgba(255, 215, 0, 0.5)', width=1, dash='dash'),
                    name='P90 (Optimistic)',
                    hovertemplate='<b>P90</b>: $%{y:,.0f}<extra></extra>'
                ))

                # Shading fills
                fig.add_trace(go.Scatter(
                    x=x_axis, y=p90_vals,
                    mode='lines', line=dict(width=0), showlegend=False, hoverinfo='skip'
                ))
                fig.add_trace(go.Scatter(
                    x=x_axis, y=p10_vals,
                    mode='lines', line=dict(width=0),
                    fill='tonexty', fillcolor='rgba(255, 215, 0, 0.05)',
                    showlegend=False, hoverinfo='skip'
                ))

                fig.add_trace(go.Scatter(
                    x=x_axis, y=p75_vals,
                    mode='lines', line=dict(width=0), showlegend=False, hoverinfo='skip'
                ))
                fig.add_trace(go.Scatter(
                    x=x_axis, y=p25_vals,
                    mode='lines', line=dict(width=0),
                    fill='tonexty', fillcolor='rgba(0, 200, 255, 0.1)',
                    showlegend=False, hoverinfo='skip'
                ))

                fig.update_layout(
                    title=f"Seasonal Performance Cone (Based on ${sim_start:,.0f})",
                    xaxis=dict(
                        title="Month (Typical Year)",
                        tickformat="%b",
                        hoverformat="%b %d",
                        dtick="M1"
                    ),
                    yaxis=dict(title="Portfolio Value ($)", tickprefix="$"),
                    template="plotly_dark",
                    height=500,
                    hovermode="x unified",
                    legend=dict(orientation="h", y=1.02, x=0.5, xanchor="center")
                )

                st.plotly_chart(fig, use_container_width=True)

                final_med = seas_df["Median"].iloc[-1]
                st.metric("Typical Year Ending Balance", f"${final_med:,.0f}")
            else:
                st.error("Not enough data for seasonality.")


def _log_mc_to_csv(mc_results: dict, n_sims: int) -> None:
    try:
        os.makedirs("debug_tools", exist_ok=True)
        log_path = "debug_tools/monte_carlo_scenarios.csv"

        paths_df = mc_results["paths"]
        final_vals = paths_df.iloc[-1]
        tot_inv = mc_results["metrics"]["total_invested"]

        path_metrics = mc_results.get("path_metrics", {})
        max_dds = path_metrics.get("max_dd", [0]*n_sims)
        twrs = path_metrics.get("final_twr", [1]*n_sims)

        ranks = final_vals.rank(pct=True)

        with open(log_path, "w", newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["Scenario_ID", "Final_Value", "Max_Drawdown", "CAGR_Strategy", "Total_Invested", "Percentile_Rank"])

            for i, col in enumerate(paths_df.columns):
                val = final_vals[col]
                dd = max_dds[i]
                twr_mult = twrs[i]
                cagr = (twr_mult ** (1/10)) - 1
                pct_rank = ranks[col]

                writer.writerow([i+1, round(val, 2), f"{dd:.2%}", f"{cagr:.2%}", round(tot_inv, 2), f"{pct_rank:.2%}"])

    except Exception as e:
        st.error(f"Failed to write detailed CSV: {e}")

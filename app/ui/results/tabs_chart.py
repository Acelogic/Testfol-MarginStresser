"""Chart tab with margin calcs, MA analysis, and cheat sheet subtabs."""
from __future__ import annotations

import streamlit as st
import pandas as pd

from app.ui import charts


def render_chart_tab(
    tab,
    chart_style: str,
    tax_adj_port_series: pd.Series,
    final_adj_series: pd.Series,
    loan_series: pd.Series,
    tax_adj_equity_pct_series: pd.Series,
    tax_adj_usage_series: pd.Series,
    equity_series: pd.Series,
    usage_series: pd.Series,
    equity_pct_series: pd.Series,
    effective_rate_series: pd.Series,
    ohlc_data,
    equity_resampled: pd.Series,
    loan_resampled: pd.Series,
    usage_resampled: pd.Series,
    equity_pct_resampled: pd.Series,
    effective_rate_resampled: pd.Series,
    bench_resampled: pd.Series | None,
    comp_resampled: pd.Series | None,
    port_series: pd.Series,
    component_prices: pd.DataFrame,
    portfolio_name: str,
    log_scale: bool,
    show_range_slider: bool,
    show_volume: bool,
    timeframe: str,
    wmaint: float,
    stats: dict,
    config: dict,
    pay_tax_cash: bool,
    draw_monthly: float,
    final_tax_series: pd.Series,
    start_val: float,
    rate_annual,
    pm_enabled: bool,
) -> None:
    with tab:
        chart_subtabs = st.tabs(["🧮 Margin Calcs", "📉 200DMA", "📉 150MA", "📊 Munger200WMA", "📜 Cheat Sheet"])

        with chart_subtabs[0]:
            if chart_style == "Classic (Combined)":
                series_opts = ["Portfolio", "Equity", "Loan", "Margin usage %"]
                charts.render_classic_chart(
                    tax_adj_port_series, final_adj_series, loan_series,
                    tax_adj_equity_pct_series, tax_adj_usage_series,
                    series_opts, log_scale,
                    bench_series=bench_resampled,
                    comparison_series=comp_resampled,
                    effective_rate_series=effective_rate_series
                )
            else:  # Candlestick
                charts.render_candlestick_chart(
                    ohlc_data,
                    equity_resampled,
                    loan_resampled,
                    usage_resampled,
                    equity_pct_resampled,
                    timeframe,
                    log_scale,
                    show_range_slider=show_range_slider,
                    show_volume=show_volume,
                    bench_series=bench_resampled,
                    comparison_series=comp_resampled,
                    effective_rate_series=effective_rate_resampled,
                )

            if pay_tax_cash:
                _render_cash_statistics(final_adj_series, final_tax_series, draw_monthly, equity_series)
            else:
                _render_margin_statistics(
                    tax_adj_port_series, final_adj_series, loan_series,
                    equity_series, usage_series, equity_pct_series,
                    final_tax_series, draw_monthly, wmaint, pm_enabled,
                )

        with chart_subtabs[1]:
            charts.render_ma_analysis_tab(port_series, portfolio_name, portfolio_name, window=200, show_stage_analysis=False)

        with chart_subtabs[2]:
            charts.render_ma_analysis_tab(port_series, portfolio_name, portfolio_name, window=150, show_stage_analysis=True)

        with chart_subtabs[3]:
            charts.render_munger_wma_tab(port_series, portfolio_name, portfolio_name, window=200)

        with chart_subtabs[4]:
            charts.render_cheat_sheet(
                port_series,
                portfolio_name,
                portfolio_name,
                component_data=component_prices
            )


def _render_cash_statistics(
    final_adj_series: pd.Series,
    final_tax_series: pd.Series,
    draw_monthly: float,
    equity_series: pd.Series,
) -> None:
    with st.expander("Detailed Cash Statistics", expanded=True):
        total_tax_paid = final_tax_series.sum() if not final_tax_series.empty else 0.0

        total_draws = 0.0
        if draw_monthly > 0 and not equity_series.empty:
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
        import pandas as _pd
        w_df = _pd.DataFrame({
            "Category": ["Monthly Draws", "Taxes Paid"],
            "Amount": [total_draws, total_tax_paid]
        })
        st.dataframe(w_df.style.format({"Amount": "${:,.2f}"}), use_container_width=True, hide_index=True)


def _render_margin_statistics(
    tax_adj_port_series: pd.Series,
    final_adj_series: pd.Series,
    loan_series: pd.Series,
    equity_series: pd.Series,
    usage_series: pd.Series,
    equity_pct_series: pd.Series,
    final_tax_series: pd.Series,
    draw_monthly: float,
    wmaint: float,
    pm_enabled: bool,
) -> None:
    with st.expander("Detailed Margin Statistics", expanded=True):
        if usage_series.empty:
            return

        max_usage_idx = usage_series.idxmax()
        max_usage_val = usage_series.max()
        equity_at_max = equity_series.loc[max_usage_idx]

        min_equity_idx = equity_series.idxmin()
        min_equity_val = equity_series.min()

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

        # Row 1: Current Status
        c1, c2, c3 = st.columns(3)
        c1.metric("Final Equity", f"${equity_series.iloc[-1]:,.2f}")
        c2.metric("Final Loan", f"${loan_series.iloc[-1]:,.2f}")
        c3.metric("Final Usage", f"{usage_series.iloc[-1]*100:.2f}%")

        # Row 2: Extremes & Capacity
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
                breaches = pd.concat([breaches, pm_breaches]).sort_values("Date")

        st.markdown("##### Margin & PM Breaches")
        if breaches.empty:
            st.success("No breaches triggered! 🎉")
        else:
            reg_t_count = len(breaches[breaches["Type"] == "Reg T Call"])
            pm_count = len(breaches[breaches["Type"] == "PM Min Equity < $100k"])

            msg_parts = []
            if reg_t_count > 0:
                msg_parts.append(f"{reg_t_count} Margin Call(s)")
            if pm_count > 0:
                msg_parts.append(f"{pm_count} PM Breach(es)")

            full_msg = f"⚠️ {' + '.join(msg_parts)} triggered."

            if reg_t_count > 0:
                st.error(full_msg)
            else:
                st.warning(full_msg)

            st.dataframe(breaches, use_container_width=True)

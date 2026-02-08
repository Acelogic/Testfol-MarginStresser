"""Results package — render() orchestrator with tab delegation."""
from __future__ import annotations

import streamlit as st
import pandas as pd
import numpy as np
import datetime as dt

from app.common import utils
from app.core import calculations, tax_library
from app.services import testfol_api as api
from app.reporting import report_generator
from app.ui import charts, xray_view

from app.ui.results.tabs_chart import render_chart_tab
from app.ui.results.tabs_tax import render_tax_impact_tab
from app.ui.results.tabs_monte_carlo import render_monte_carlo_tab
from app.ui.results.tabs_debug import render_debug_tab


def render(results: dict, config: dict, portfolio_name: str = "", clip_start_date=None) -> None:
    """
    Renders the results section, including metrics and charts.
    """

    # Extract Data from Results
    port_series = results["port_series"]
    stats = results["stats"]
    portfolio_name = results.get("name", "Portfolio")
    component_prices = results.get("component_prices", pd.DataFrame())


    # --- Clip Data Logic (Sync with Chart) ---
    original_start_date = results.get("start_date")
    twr_series = results.get("twr_series")

    if clip_start_date and not port_series.empty:
        # Avoid clipping if clip_start is before port start
        if clip_start_date > port_series.index[0]:
            port_series = port_series[port_series.index >= clip_start_date]

            # Recalculate Stats for clipped period
            if not port_series.empty:
                # Use TWR Series for stats if available (Correct for Cashflows)
                target_series = port_series
                if twr_series is not None and not twr_series.empty:
                    twr_clipped = twr_series[twr_series.index >= clip_start_date]
                    if not twr_clipped.empty:
                        target_series = twr_clipped

                stats = calculations.generate_stats(target_series)

    # Initialize optional chart variables to prevent UnboundLocalError
    fig_tax_impact = None

    if port_series.empty or not isinstance(port_series.index, pd.DatetimeIndex):
        st.error("No valid simulation data generated (or data completely clipped). Check inputs or API connection.")
        return

    trades_df = results["trades_df"]
    pl_by_year = results["pl_by_year"]
    composition_df = results.get("composition_df", pd.DataFrame())
    raw_response = results["raw_response"]

    start_val = port_series.iloc[0] # RE-BASE start val to the clipped start

    wmaint = results.get("wmaint", config.get('wmaint', 0.25))

    # Tax/Margin Config
    pay_tax_margin = config.get('pay_tax_margin', False)
    pay_tax_cash = config.get('pay_tax_cash', False)
    other_income = config.get('other_income', 0.0)
    filing_status = config.get('filing_status', 'Single')
    tax_method = config.get('tax_method', '2025_fixed')
    use_std_deduction = config.get('use_std_deduction', True)
    state_tax_rate = config.get('state_tax_rate', 0.0)

    rate_annual = config.get('rate_annual', 8.0)
    draw_monthly = config.get('draw_monthly', 0.0)
    starting_loan = config.get('starting_loan', 0.0)

    cashflow = config.get('cashflow', 0.0)
    cashfreq = config.get('cashfreq', "Monthly")
    pay_down_margin = config.get('pay_down_margin', False)

    chart_style = config.get('chart_style', "Classic (Combined)")
    timeframe = config.get('timeframe', "1M")
    log_scale = config.get('log_scale', True)
    show_range_slider = config.get('show_range_slider', True)
    show_volume = config.get('show_volume', True)

    pm_enabled = config.get('pm_enabled', False)

    # Extract Benchmark (Standard Comparison or Custom)
    bench_series = results.get("bench_series", None)
    bench_stats = results.get("bench_stats")

    # Clip Benchmark if exists
    if bench_series is not None and clip_start_date:
         bench_series = bench_series[bench_series.index >= clip_start_date]
         if not bench_series.empty:
             bench_stats = calculations.generate_stats(bench_series)

    logs = results.get("logs", [])

    # Calculate Tax Series for Margin Simulation (and Metrics)
    tax_payment_series = None
    total_tax_owed = 0.0

    # Initialize Tax Series (Defaults)
    fed_tax_series = pd.Series(dtype=float)
    state_tax_series = pd.Series(dtype=float)

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
    else:
        # No realized P&L = No Tax
        fed_tax_series = pd.Series(0.0, index=[dt.date.today().year]) # Dummy index
        state_tax_series = pd.Series(0.0, index=[dt.date.today().year])

    total_tax_owed = fed_tax_series.sum() + state_tax_series.sum()

    # Create Payment Series (Unconditional for Sharpe Calc)
    tax_payment_series = pd.Series(0.0, index=port_series.index)
    annual_total_tax = fed_tax_series + state_tax_series

    for year, amount in annual_total_tax.items():
        if amount > 0:
            # Pay on April 15th of NEXT year
            pay_date = pd.Timestamp(year + 1, 4, 15)

            idx = port_series.index.searchsorted(pay_date)

            if idx < len(port_series.index):
                actual_date = port_series.index[idx]
                tax_payment_series[actual_date] += amount

    # Prepare Repayment Series (if Pay Down Margin is enabled)
    repayment_series = None
    if pay_down_margin and cashflow > 0:
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

    # Re-run margin sim
    sim_tax_series = tax_payment_series if pay_tax_margin else None

    eff_loan = 0.0 if pay_tax_cash else starting_loan
    eff_rate = 0.0 if pay_tax_cash else rate_annual
    eff_draw = 0.0 if pay_tax_cash else draw_monthly

    loan_series, equity_series, equity_pct_series, usage_series, effective_rate_series = api.simulate_margin(
        port_series, eff_loan,
        eff_rate, eff_draw, wmaint,
        tax_series=sim_tax_series,
        repayment_series=repayment_series
    )

    # Update session state with latest margin results for reporting
    results.update({
        "loan_series": loan_series,
        "equity_series": equity_series,
        "usage_series": usage_series,
        "equity_pct_series": equity_pct_series,
        "effective_rate_series": effective_rate_series
    })


    # Calculate Tax-Adjusted Equity Curve (Global for Tabs)
    final_adj_series = pd.Series(dtype=float)
    final_tax_series = pd.Series(dtype=float) # Series of ACTUAL taxes paid (scaled if needed)

    if not equity_series.empty:
        if pay_tax_margin:
            final_adj_series = equity_series
            final_tax_series = tax_payment_series if tax_payment_series is not None else pd.Series(0.0, index=equity_series.index)
        elif pay_tax_cash:
            if tax_payment_series is not None and tax_payment_series.sum() > 0:
                final_adj_series, final_tax_series = calculations.calculate_tax_adjusted_equity(
                    equity_series, tax_payment_series, port_series, loan_series, rate_annual, draw_monthly=draw_monthly
                )
            else:
                empty_tax = pd.Series(0.0, index=equity_series.index)
                final_adj_series, final_tax_series = calculations.calculate_tax_adjusted_equity(
                    equity_series, empty_tax, port_series, loan_series, rate_annual, draw_monthly=draw_monthly
                )
        else: # None (Gross)
            final_adj_series = equity_series
            final_tax_series = pd.Series(0.0, index=equity_series.index)

    # --- Prepare Tax-Adjusted Data for Charts ---
    tax_adj_port_series = final_adj_series + loan_series

    # Retrieve benchmark if available
    bench_series = results.get("bench_series")
    bench_resampled = None
    if bench_series is not None:
            bench_aligned = bench_series.reindex(tax_adj_port_series.index).ffill().fillna(0)
            bench_resampled = utils.resample_data(bench_aligned, timeframe, method="last")

    # Retrieve Comparison Benchmark (Standard Rebalance) if available
    comp_series = results.get("comparison_series")
    comp_resampled = None
    if comp_series is not None:
             comp_aligned = comp_series.reindex(tax_adj_port_series.index).ffill().fillna(0)
             comp_resampled = utils.resample_data(comp_aligned, timeframe, method="last")
             if comp_series.name is None:
                 comp_resampled.name = "Standard (Yearly)"
             else:
                 comp_resampled.name = comp_series.name

    # Recalculate Leverage Metrics based on Tax-Adjusted Equity
    tax_adj_equity_pct_series = pd.Series(0.0, index=tax_adj_port_series.index)
    valid_idx = tax_adj_port_series > 0
    tax_adj_equity_pct_series[valid_idx] = final_adj_series[valid_idx] / tax_adj_port_series[valid_idx]

    tax_adj_usage_series = pd.Series(0.0, index=tax_adj_port_series.index)
    if wmaint < 1.0:
        max_loan_series = tax_adj_port_series * (1 - wmaint)
        valid_loan = max_loan_series > 0
        tax_adj_usage_series[valid_loan] = loan_series[valid_loan] / max_loan_series[valid_loan]


    ohlc_data = utils.resample_data(tax_adj_port_series, timeframe, method="ohlc")
    equity_resampled = utils.resample_data(final_adj_series, timeframe, method="last")
    loan_resampled = utils.resample_data(loan_series, timeframe, method="last")
    usage_resampled = utils.resample_data(tax_adj_usage_series, timeframe, method="max")
    equity_pct_resampled = utils.resample_data(tax_adj_equity_pct_series, timeframe, method="last")
    effective_rate_resampled = utils.resample_data(effective_rate_series, timeframe, method="last")



    # Update Range Caption to reflect actual displayed data
    if not port_series.empty:
        start_str = port_series.index[0].strftime('%Y-%m-%d')
        end_str = port_series.index[-1].strftime('%Y-%m-%d')
        display_range = f"{start_str} to {end_str} (Synced)"
    else:
        display_range = results.get('sim_range', 'N/A')

    st.caption(f"📅 **Backtest Range:** {display_range}")
    m1, m2, m3, m4, m5 = st.columns(5)

    # Comparison Logic
    if bench_series is not None:
         try:
             aligned_pf = tax_adj_port_series.reindex(bench_series.index).ffill()
             diff_val = aligned_pf.iloc[-1] - bench_series.iloc[-1]
             diff_pct = (diff_val / bench_series.iloc[-1]) * 100
             st.info(f"**Comparison Active**: {portfolio_name} vs {bench_series.name if bench_series.name else 'Benchmark'} | Diff: ${diff_val:,.2f} ({diff_pct:+.2f}%)")
         except (IndexError, KeyError, ZeroDivisionError, TypeError):
             pass

    total_return = (tax_adj_port_series.iloc[-1] / start_val - 1) * 100

    # Use Stats reported by Testfol API (TWR)
    cagr = stats.get("cagr", 0.0)
    max_dd = stats.get("max_drawdown", 0.0)
    sharpe = stats.get("sharpe", 0.0)

    # Retrieve Benchmark Stats
    bench_stats = results.get("bench_stats")

    # Calculate CAGR display value
    cagr_display = cagr * 100 if abs(cagr) <= 1 else cagr

    # Calculate Tax-Adjusted metrics
    if start_val > 0 and not port_series.empty:
        final_adj_val = final_adj_series.iloc[-1]
        days = (final_adj_series.index[-1] - final_adj_series.index[0]).days
        if days > 0:
            years = days / 365.25
            tax_adj_cagr = (final_adj_val / start_val) ** (1 / years) - 1
        else:
            tax_adj_cagr = 0.0
        tax_adj_sharpe = calculations.calculate_sharpe_ratio(final_adj_series)
    else:
        final_adj_val = 0.0
        tax_adj_cagr = 0.0
        tax_adj_sharpe = 0.0

    # Calculate Differences if Benchmark Exists
    active_bench_series = bench_series if bench_series is not None else results.get("comparison_series")
    active_bench_stats = bench_stats if bench_stats is not None else results.get("comparison_stats")

    if active_bench_stats:
        b_cagr = active_bench_stats.get("cagr", 0.0)
        b_sharpe = active_bench_stats.get("sharpe", 0.0)
        b_dd = active_bench_stats.get("max_drawdown", 0.0)

        b_cagr_display = b_cagr * 100 if abs(b_cagr) <= 1 else b_cagr
        diff_cagr_display = cagr_display - b_cagr_display
        diff_sharpe = sharpe - b_sharpe
        diff_dd = max_dd - b_dd

        bench_label = "Bench"
        if active_bench_series is not None and hasattr(active_bench_series, 'name') and active_bench_series.name:
            bench_label = active_bench_series.name.replace("Benchmark", "").replace("Standard", "").strip(" ()")
            if not bench_label: bench_label = "Bench"

        # Primary Metrics Row - Gross (Pre-Tax) from API
        m1.metric("Portfolio Value", f"${tax_adj_port_series.iloc[-1]:,.0f}", f"{total_return:+.1f}%")
        m2.metric("Gross CAGR", f"{cagr_display:.2f}%", f"{diff_cagr_display:+.2f}% vs {bench_label}", help="Pre-tax return from Testfol API")
        m3.metric("Gross Sharpe", f"{sharpe:.2f}", f"{diff_sharpe:+.2f} vs {bench_label}", help="Pre-tax risk-adjusted return")
        m4.metric("Max Drawdown", f"{max_dd:.2f}%", f"{diff_dd:+.2f}% vs {bench_label}", delta_color="inverse")
        m5.metric("Leverage", f"{(tax_adj_port_series.iloc[-1]/final_adj_series.iloc[-1]):.2f}x")

        # Detailed Comparison Table
        bench_end_val = active_bench_series.iloc[-1] if active_bench_series is not None and not active_bench_series.empty else 0
        strategy_end_val = tax_adj_port_series.iloc[-1]

        comp_label = active_bench_series.name if active_bench_series is not None and hasattr(active_bench_series, 'name') and active_bench_series.name else "Benchmark"

        comp_data = {
            "Metric": ["Ending Value", "CAGR", "Sharpe", "Max Drawdown", "Std Dev"],
            portfolio_name: [f"${strategy_end_val:,.0f}", f"{cagr_display:.2f}%", f"{sharpe:.2f}", f"{max_dd:.2f}%", f"{stats.get('volatility',0)*100:.2f}%"],
            comp_label: [f"${bench_end_val:,.0f}", f"{b_cagr_display:.2f}%", f"{b_sharpe:.2f}", f"{b_dd:.2f}%", f"{active_bench_stats.get('volatility',0)*100:.2f}%"],
            "Diff": [f"${strategy_end_val - bench_end_val:+,.0f}", f"{diff_cagr_display:+.2f}%", f"{diff_sharpe:+.2f}", f"{diff_dd:+.2f}%", ""]
        }
        comp_df = pd.DataFrame(comp_data)
        with st.expander(f"📊 Detailed {portfolio_name} vs {comp_label} Statistics", expanded=False):
            st.dataframe(comp_df, hide_index=True, use_container_width=True)

    else:
        # Standard View (No Benchmark) - Gross (Pre-Tax) from API
        m1.metric("Portfolio Value", f"${tax_adj_port_series.iloc[-1]:,.0f}", f"{total_return:+.1f}%")
        m2.metric("Gross CAGR", f"{cagr_display:.2f}%", help="Pre-tax return from Testfol API")
        m3.metric("Gross Sharpe", f"{sharpe:.2f}", help="Pre-tax risk-adjusted return")
        m4.metric("Max Drawdown", f"{max_dd:.2f}%", delta_color="inverse")
        m5.metric("Leverage", f"{(tax_adj_port_series.iloc[-1]/final_adj_series.iloc[-1]):.2f}x")

    # --- Secondary Metrics in Expander (Only show when tax simulation is active) ---
    tax_sim_active = pay_tax_margin or pay_tax_cash

    if tax_sim_active:
        with st.expander("💰 Tax & Post-Tax Details", expanded=False):
            st.caption(f"📅 **Shadow Data Range:** {results.get('shadow_range', 'N/A')}")

            sm1, sm2, sm3, sm4 = st.columns(4)

            # Tax Info
            tax_label = "Total Tax Paid" if pay_tax_margin else "Est. Tax Owed"
            if total_tax_owed > 0:
                if not final_tax_series.empty and final_tax_series.sum() > 0:
                    display_tax = final_tax_series.sum()
                else:
                    display_tax = total_tax_owed
            else:
                display_tax = 0.0

            sm1.metric(tax_label, f"${display_tax:,.0f}")
            sm2.metric("Post-Tax Equity", f"${final_adj_val:,.0f}", help="Equity after tax simulation")
            sm3.metric("Post-Tax CAGR", f"{tax_adj_cagr * 100:.2f}%", help="Growth rate after taxes")
            sm4.metric("Post-Tax Sharpe", f"{tax_adj_sharpe:.2f}", help="Risk-adjusted return after taxes")

            unpaid_liability = total_tax_owed - display_tax
            if unpaid_liability > 1:
                st.caption(f"ℹ️ **Timing Difference:** Total Tax Paid (\${display_tax:,.0f}) is lower than Total Tax Owed (\${total_tax_owed:,.0f}) because taxes are typically paid on **April 15th of the following year**. The tax bill for the final simulation year (\${unpaid_liability:,.0f}) is technically owed (Accrued) but the payment date falls **after** the simulation ends, so it was never deducted from your cash.")

    st.markdown("---")

    # =========================================================================
    # Tabs
    # =========================================================================
    res_tab_chart, res_tab_returns, res_tab_rebal, res_tab_tax, res_tab_xray, res_tab_mc, res_tab_debug = st.tabs(
        ["📈 Chart", "📊 Returns Analysis", "⚖️ Rebalancing", "💸 Tax Analysis", "🔍 X-Ray", "🔮 Monte Carlo", "🔧 Debug"]
    )

    # --- Tax Impact Tab ---
    render_tax_impact_tab(
        res_tab_tax,
        pl_by_year=pl_by_year,
        config=config,
        port_series=port_series,
        equity_series=equity_series,
        loan_series=loan_series,
        final_adj_series=final_adj_series,
        annual_total_tax=annual_total_tax,
        tax_payment_series=tax_payment_series,
        pay_tax_margin=pay_tax_margin,
        pay_tax_cash=pay_tax_cash,
        rate_annual=rate_annual,
        draw_monthly=draw_monthly,
        starting_loan=starting_loan,
        wmaint=wmaint,
        repayment_series=repayment_series,
        twr_series=results.get("twr_series"),
        stats=stats,
    )

    # --- Debug Tab ---
    render_debug_tab(res_tab_debug, logs, raw_response, portfolio_name)

    # --- Chart Tab ---
    render_chart_tab(
        res_tab_chart,
        chart_style=chart_style,
        tax_adj_port_series=tax_adj_port_series,
        final_adj_series=final_adj_series,
        loan_series=loan_series,
        tax_adj_equity_pct_series=tax_adj_equity_pct_series,
        tax_adj_usage_series=tax_adj_usage_series,
        equity_series=equity_series,
        usage_series=usage_series,
        equity_pct_series=equity_pct_series,
        effective_rate_series=effective_rate_series,
        ohlc_data=ohlc_data,
        equity_resampled=equity_resampled,
        loan_resampled=loan_resampled,
        usage_resampled=usage_resampled,
        equity_pct_resampled=equity_pct_resampled,
        effective_rate_resampled=effective_rate_resampled,
        bench_resampled=bench_resampled,
        comp_resampled=comp_resampled,
        port_series=port_series,
        component_prices=component_prices,
        portfolio_name=portfolio_name,
        log_scale=log_scale,
        show_range_slider=show_range_slider,
        show_volume=show_volume,
        timeframe=timeframe,
        wmaint=wmaint,
        stats=stats,
        config=config,
        pay_tax_cash=pay_tax_cash,
        draw_monthly=draw_monthly,
        final_tax_series=final_tax_series,
        start_val=start_val,
        rate_annual=rate_annual,
        pm_enabled=pm_enabled,
    )

    # --- Returns Tab (inline, small) ---
    with res_tab_returns:
        if pay_tax_cash:
            st.info("ℹ️ **Note:** Returns are **Net of Tax** (simulated as cash withdrawals).")
        elif pay_tax_margin:
            st.info("ℹ️ **Note:** Returns are **Gross** (taxes paid via margin loan).")
        else:
            st.info("ℹ️ **Note:** Returns are **Gross** (Pre-Tax).")

        charts.render_returns_analysis(
            tax_adj_port_series,
            bench_series=bench_resampled,
            comparison_series=comp_resampled,
            unique_id=portfolio_name,
            portfolio_name=portfolio_name,
            component_data=component_prices,
            raw_port_series=port_series
        )

    # --- Rebalancing Tab (inline, small) ---
    with res_tab_rebal:
        st.warning("⚠️ **Note:** These trade calculations assume **Gross** portfolio values. Tax payments are NOT deducted by selling shares in this view (assumes taxes paid via margin or external cash).")
        rebal_freq_for_chart = config.get('rebalance', 'Yearly')

        if not composition_df.empty:
            for d in composition_df["Date"].unique():
                if d in tax_adj_port_series.index:
                    target_total = tax_adj_port_series.loc[d]
                    current_total = composition_df[composition_df["Date"] == d]["Value"].sum()
                    if current_total > 0:
                        ratio = target_total / current_total
                        composition_df.loc[composition_df["Date"] == d, "Value"] *= ratio

        charts.render_rebalancing_analysis(
            trades_df, pl_by_year, composition_df,
            tax_method, other_income, filing_status, state_tax_rate,
            rebalance_freq=rebal_freq_for_chart,
            use_standard_deduction=use_std_deduction,
            unrealized_pl_df=results.get("unrealized_pl_df", pd.DataFrame()),
            custom_freq=config.get('custom_freq', 'Yearly'),
            unique_id=portfolio_name
        )

    # --- Tax Analysis (charts) within same tax tab ---
    with res_tab_tax:
        charts.render_tax_analysis(
            pl_by_year, other_income, filing_status, state_tax_rate,
            tax_method=tax_method,
            use_standard_deduction=use_std_deduction,
            unrealized_pl_df=results.get("unrealized_pl_df", pd.DataFrame()),
            trades_df=trades_df,
            pay_tax_cash=pay_tax_cash,
            pay_tax_margin=pay_tax_margin
        )

    # --- X-Ray Tab (inline, small) ---
    with res_tab_xray:
        if not composition_df.empty:
            latest_date = composition_df['Date'].max()
            latest_df = composition_df[composition_df['Date'] == latest_date]

            total_val = latest_df['Value'].sum()
            if total_val > 0:
                alloc_map = dict(zip(latest_df['Ticker'], latest_df['Value'] / total_val))
                xray_view.render_xray(alloc_map, portfolio_name=portfolio_name)
            else:
                st.info("No allocation data found for X-Ray.")
        else:
            st.info("No composition data available for X-Ray.")

    # --- Monte Carlo Tab ---
    # Prepare daily_rets and source_label before delegation
    extended_series = results.get("port_series")
    tax_twr_series = results.get("twr_series")

    daily_rets = pd.Series(dtype=float)
    source_label = "Unknown"

    use_extended = False

    if extended_series is not None and not extended_series.empty:
        if tax_twr_series is not None and not tax_twr_series.empty:
             if extended_series.index[0] < tax_twr_series.index[0]:
                 use_extended = True
                 source_label = "Extended Chart Data (Simulated)"
             else:
                 daily_rets = tax_twr_series.pct_change()
                 source_label = "Tax TWR Data (Real)"
        else:
             use_extended = True
             source_label = "Extended Chart Data (Simulated)"
    elif tax_twr_series is not None and not tax_twr_series.empty:
         daily_rets = tax_twr_series.pct_change()
         source_label = "Tax TWR Data (Real)"

    if use_extended:
         daily_rets = extended_series.pct_change()

    render_monte_carlo_tab(
        res_tab_mc,
        results=results,
        config=config,
        portfolio_name=portfolio_name,
        daily_rets=daily_rets,
        source_label=source_label,
    )

    # -------------------------------------------------------------------------
    # Post-Tabs (Charts Controls & Report)
    # -------------------------------------------------------------------------
    st.divider()
    col_c1, col_c2, col_c3 = st.columns([2, 1, 1])

    # Report Generation (Sidebar)
    if results:
        st.sidebar.markdown("---")
        st.sidebar.subheader("📄 Report")

        try:
            report_data = st.session_state.get('results_list', results)
            if isinstance(report_data, list) and not report_data:
                report_data = results

            report_html = report_generator.generate_html_report(report_data)

            btn_label = "Download Combined Report (HTML)" if isinstance(report_data, list) and len(report_data) > 1 else "Download HTML Report"

            st.sidebar.download_button(
                label=btn_label,
                data=report_html,
                file_name=f"testfol_report_{dt.datetime.now().strftime('%Y%m%d_%H%M')}.html",
                mime="text/html",
                key=f"dl_report_{portfolio_name}",
                help="Download a standalone HTML report with charts and stats."
            )
        except Exception as e:
            st.sidebar.error(f"Report Gen Failed: {e}")

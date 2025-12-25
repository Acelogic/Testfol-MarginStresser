import streamlit as st
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import datetime as dt
import os
import csv
import json
from app.common import utils
from app.core import run_shadow_backtest, calculations, tax_library
from app.core import monte_carlo # Import Monte Carlo module
from app.services import testfol_api as api
from app.reporting import report_generator
from app.ui import charts

def render(results, config, portfolio_name=""):
    """
    Renders the results section, including metrics and charts.
    
    Args:
        results (dict): The results dictionary from the backtest.
        config (dict): The configuration dictionary used for the backtest.
    """
        
    # Extract Data from Results
    
    port_series = results["port_series"]
    stats = results["stats"]
    portfolio_name = results.get("name", "Portfolio")
    
    # Initialize optional chart variables to prevent UnboundLocalError
    fig_tax_impact = None
    
    if port_series.empty or not isinstance(port_series.index, pd.DatetimeIndex):
        st.error("No valid simulation data generated. Check inputs or API connection.")
        return
        
    trades_df = results["trades_df"]
    pl_by_year = results["pl_by_year"]
    composition_df = results.get("composition_df", pd.DataFrame())
    raw_response = results["raw_response"]
    
    # Extract Config Values
    # We fallback to results defaults if necessary, but config should match
    start_val = results["start_val"]
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
    
    pm_enabled = config.get('pm_enabled', False)
    
    # Extract Benchmark (Standard Comparison or Custom)
    bench_series = results.get("bench_series", None)
    
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
            # If paying with margin, we pay the FULL tax amount (no scaling down)
            final_tax_series = tax_payment_series if tax_payment_series is not None else pd.Series(0.0, index=equity_series.index)
        elif pay_tax_cash:
            if tax_payment_series is not None and tax_payment_series.sum() > 0:
                final_adj_series, final_tax_series = calculations.calculate_tax_adjusted_equity(
                    equity_series, tax_payment_series, port_series, loan_series, rate_annual, draw_monthly=draw_monthly
                )
            else:
                # Even if no taxes, we still need to apply the monthly draw if it exists
                # Re-use the function but with empty tax series
                empty_tax = pd.Series(0.0, index=equity_series.index)
                final_adj_series, final_tax_series = calculations.calculate_tax_adjusted_equity(
                    equity_series, empty_tax, port_series, loan_series, rate_annual, draw_monthly=draw_monthly
                )
        else: # None (Gross)
            final_adj_series = equity_series
            final_tax_series = pd.Series(0.0, index=equity_series.index)
    
    # --- Prepare Tax-Adjusted Data for Charts ---
    # We want the charts to reflect the "Net" reality.
    # Portfolio Value = Net Equity + Loan
    tax_adj_port_series = final_adj_series + loan_series
    
    # Retrieve benchmark if available
    bench_series = results.get("bench_series")
    bench_resampled = None
    if bench_series is not None:
            # Resample/Align benchmark to match portfolio index or timeframe
            # Usually aligning to tax_adj_port_series
            bench_aligned = bench_series.reindex(tax_adj_port_series.index, method="ffill").fillna(0)
            # Resample for charts handled below
            bench_resampled = utils.resample_data(bench_aligned, timeframe, method="last")
    
    # Retrieve Comparison Benchmark (Standard Rebalance) if available
    comp_series = results.get("comparison_series")
    comp_resampled = None
    if comp_series is not None:
             comp_aligned = comp_series.reindex(tax_adj_port_series.index, method="ffill").fillna(0)
             comp_resampled = utils.resample_data(comp_aligned, timeframe, method="last")
             if comp_series.name is None:
                 comp_resampled.name = "Standard (Yearly)"
             else:
                 comp_resampled.name = comp_series.name
    
    # Recalculate Leverage Metrics based on Tax-Adjusted Equity
    # Equity % = Net Equity / Portfolio Value
    tax_adj_equity_pct_series = pd.Series(0.0, index=tax_adj_port_series.index)
    valid_idx = tax_adj_port_series > 0
    tax_adj_equity_pct_series[valid_idx] = final_adj_series[valid_idx] / tax_adj_port_series[valid_idx]
    
    tax_adj_usage_series = pd.Series(0.0, index=tax_adj_port_series.index)
    if wmaint < 1.0: # Avoid division by zero if maint is 100%
        # Match the logic in simulate_margin: Usage = Loan / (Port * (1 - Maint))
        # This represents usage of the current collateral's loan value (Risk of Call)
        max_loan_series = tax_adj_port_series * (1 - wmaint)
        valid_loan = max_loan_series > 0
        tax_adj_usage_series[valid_loan] = loan_series[valid_loan] / max_loan_series[valid_loan]
    else:
        pass
    
    
    ohlc_data = utils.resample_data(tax_adj_port_series, timeframe, method="ohlc")
    equity_resampled = utils.resample_data(final_adj_series, timeframe, method="last")
    loan_resampled = utils.resample_data(loan_series, timeframe, method="last")
    usage_resampled = utils.resample_data(tax_adj_usage_series, timeframe, method="max")
    equity_pct_resampled = utils.resample_data(tax_adj_equity_pct_series, timeframe, method="last")
    effective_rate_resampled = utils.resample_data(effective_rate_series, timeframe, method="last")
    

    
    st.caption(f"📅 **Backtest Range:** {results.get('sim_range', 'N/A')}")
    m1, m2, m3, m4, m5 = st.columns(5)
    
    # Comparison Logic
    if bench_series is not None:
         try:
             aligned_pf = tax_adj_port_series.reindex(bench_series.index, method='ffill')
             diff_val = aligned_pf.iloc[-1] - bench_series.iloc[-1]
             diff_pct = (diff_val / bench_series.iloc[-1]) * 100
             st.info(f"**Comparison Active**: Strategy vs {bench_series.name if bench_series.name else 'Benchmark'} | Diff: ${diff_val:,.2f} ({diff_pct:+.2f}%)")
         except:
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
            # Use short name for label if possible
            bench_label = active_bench_series.name.replace("Benchmark", "").replace("Standard", "").strip(" ()")
            if not bench_label: bench_label = "Bench"
        
        # Primary Metrics Row - Gross (Pre-Tax) from API
        m1.metric("Portfolio Value", f"${tax_adj_port_series.iloc[-1]:,.0f}", f"{total_return:+.1f}%")
        m2.metric("Gross CAGR", f"{cagr_display:.2f}%", f"{diff_cagr_display:+.2f}% vs {bench_label}", help="Pre-tax return from Testfol API")
        m3.metric("Gross Sharpe", f"{sharpe:.2f}", f"{diff_sharpe:+.2f} vs {bench_label}", help="Pre-tax risk-adjusted return")
        m4.metric("Max Drawdown", f"{max_dd:.2f}%", f"{diff_dd:+.2f}% vs {bench_label}", delta_color="inverse")
        m5.metric("Leverage", f"{(tax_adj_port_series.iloc[-1]/final_adj_series.iloc[-1]):.2f}x")
        
        # Detailed Comparison Table
        # Get benchmark ending value
        bench_end_val = active_bench_series.iloc[-1] if active_bench_series is not None and not active_bench_series.empty else 0
        strategy_end_val = tax_adj_port_series.iloc[-1]
        
        comp_label = active_bench_series.name if active_bench_series is not None and hasattr(active_bench_series, 'name') and active_bench_series.name else "Benchmark"
        
        comp_data = {
            "Metric": ["Ending Value", "CAGR", "Sharpe", "Max Drawdown", "Std Dev"],
            "Strategy": [f"${strategy_end_val:,.0f}", f"{cagr_display:.2f}%", f"{sharpe:.2f}", f"{max_dd:.2f}%", f"{stats.get('volatility',0)*100:.2f}%"],
            comp_label: [f"${bench_end_val:,.0f}", f"{b_cagr_display:.2f}%", f"{b_sharpe:.2f}", f"{b_dd:.2f}%", f"{active_bench_stats.get('volatility',0)*100:.2f}%"],
            "Diff": [f"${strategy_end_val - bench_end_val:+,.0f}", f"{diff_cagr_display:+.2f}%", f"{diff_sharpe:+.2f}", f"{diff_dd:+.2f}%", ""]
        }
        comp_df = pd.DataFrame(comp_data)
        with st.expander(f"📊 Detailed Strategy vs {comp_label} Statistics", expanded=False):
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
            
            # Explain discrepancy between Owed (Chart) and Paid (Metric)
            unpaid_liability = total_tax_owed - display_tax
            if unpaid_liability > 1: # Threshold for noise
                st.caption(f"ℹ️ **Timing Difference:** Total Tax Paid (\${display_tax:,.0f}) is lower than Total Tax Owed (\${total_tax_owed:,.0f}) because taxes are typically paid on **April 15th of the following year**. The tax bill for the final simulation year (\${unpaid_liability:,.0f}) is technically owed (Accrued) but the payment date falls **after** the simulation ends, so it was never deducted from your cash.")
            
    st.markdown("---")
    
    res_tab_chart, res_tab_returns, res_tab_rebal, res_tab_tax, res_tab_mc, res_tab_debug = st.tabs(["📈 Chart", "📊 Analysis", "⚖️ Rebalancing", "💸 Tax Analysis", "🔮 Monte Carlo", "🔧 Debug"])
    
    with res_tab_tax:
        st.markdown("### Annual Tax Impact Analysis")
        
        st.info("""
        **Methodology: Tax-Adjusted Returns**
        
        *   **None (Gross):** No tax simulation. Showing raw pre-tax returns.
        *   **Pay with Margin:** Taxes paid via loan. Assets preserved. Cost = Interest.
        *   **Pay from Cash:** Taxes paid via asset sales. Assets reduced. Cost = Lost Compounding.
        """)

        # --- Data Integrity Check (Chart vs Tax Data) ---
        twr_series = results.get("twr_series")
        if twr_series is not None and not twr_series.empty:
            # 1. Determine Common Timeframe
            start_dt = twr_series.index[0]
            end_dt = twr_series.index[-1]
            years = (end_dt - start_dt).days / 365.25
            
            # 2. Calculate Shadow CAGR
            shadow_cagr = (twr_series.iloc[-1] / twr_series.iloc[0]) ** (1/years) - 1
            
            # 3. Calculate "Subset" API CAGR (Aligned to Shadow Dates)
            api_cagr = 0.0
            if not port_series.empty:
                try:
                    p_start = port_series.asof(start_dt)
                    p_end = port_series.asof(end_dt)
                    if pd.notna(p_start) and pd.notna(p_end) and p_start > 0:
                        api_cagr = (p_end / p_start) ** (1/years) - 1
                except:
                    api_cagr = stats.get("cagr", 0.0)
                    if api_cagr > 1.0: api_cagr /= 100.0

            # 4. Comparison
            cf_amt = config.get('cashflow', 0.0)
            diff = abs(shadow_cagr - api_cagr)
            
            # Show Validation Expander
            with st.expander("🔎 Data Validation (Chart vs Tax Data)", expanded=False):
                st.caption(f"Comparing performance over overlapped period: {start_dt.strftime('%Y-%m-%d')} to {end_dt.strftime('%Y-%m-%d')}")
                
                c1, c2, c3 = st.columns(3)
                c1.metric("Chart CAGR (Testfol)", f"{api_cagr:.2%}")
                c2.metric("Tax CAGR (yFinance)", f"{shadow_cagr:.2%}")
                
                delta_color = "normal"
                if diff < 0.002: delta_color = "normal" 
                elif diff < 0.005: delta_color = "off" 
                else: delta_color = "inverse" 
                
                c3.metric("Drift", f"{diff:.2%}", delta="Match" if diff < 0.005 else "Mismatch", delta_color=delta_color)
                
                # Explanatory notes
                if abs(cf_amt) > 10:
                    st.caption("Note: Small drift is expected when Cashflows are active (MWR vs TWR).")
                elif diff > 0.01:
                    st.warning("Significant drift detected! 'yfinance' price data may differ from Testfol's source (e.g., missing dividends, bad splits). Tax calculations may be less accurate.")
                
                # Hybrid mode explanation
                sim_engine = config.get('sim_engine', 'standard')
                if sim_engine == 'hybrid':
                    st.info("""
**Hybrid Mode: Why Drift Occurs**

Your chart uses Testfol's extended simulated data (e.g., SPYSIM for SPY back to the 1980s). However, tax calculations use **real yFinance data only**, which may start later (e.g., real ZROZ data starts in 2009).

When yFinance data starts later than your chart, the tax engine initializes your portfolio at that later date using the chart's value at that time as your cost basis. This ensures taxes are grounded in reality rather than synthetic models.

**Example:** If your chart shows $500k in 2009 but uses simulated data for 1980–2009, your taxes will treat 2009 as your acquisition date with a $500k cost basis.
                    """.strip())

        if (pay_tax_margin or pay_tax_cash) and not final_adj_series.empty and not pl_by_year.empty:
            # Prepare Data
            # 1. Annual Ending Balance (Tax Adjusted)
            annual_bal = final_adj_series.resample("YE").last()
            annual_bal.index = annual_bal.index.year
            
            # 2. Annual Tax INCURRED (for the impact chart, we show tax based on gains that year)
            # annual_total_tax is indexed by year (int)
            annual_tax_aligned = annual_total_tax
            # Reindex to match balance just in case
            annual_tax_aligned = annual_tax_aligned.reindex(annual_bal.index, fill_value=0.0)
            
            # 3. Annual Market Value (Gross Assets)
            # Market Value = Net Equity + Loan
            # This works for Margin (Loan increases)
            # For Cash Mode, we need a "Gross" baseline that includes Draws but NO Taxes
            if pay_tax_cash:
                # Calculate Gross Cash Series (Draws YES, Taxes NO)
                empty_tax = pd.Series(0.0, index=equity_series.index)
                gross_cash_series, _ = calculations.calculate_tax_adjusted_equity(
                    equity_series, empty_tax, port_series, loan_series, rate_annual, draw_monthly=draw_monthly
                )
                market_val_series = gross_cash_series
            else:
                # Margin Mode: Market Value = Pre-Tax Net Equity (Net Equity if NO tax debt existed)
                # We reuse the simulate_margin function but with tax_series=None
                gross_margin_loan, gross_margin_equity, _, _, _ = api.simulate_margin(
                    port_series, starting_loan, rate_annual, draw_monthly, wmaint,
                    tax_series=None, repayment_series=repayment_series
                )
                market_val_series = gross_margin_equity
            
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
            
            # Market Value as a Line (Baseline: What your wealth would be if taxes didn't exist)
            fig_tax_impact.add_trace(go.Scatter(
                x=tax_impact_df.index,
                y=tax_impact_df["Market Value"],
                name="Pre-Tax Wealth (Baseline)",
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
                title="Annual Tax Impact: Net Wealth vs. Pre-Tax Baseline",
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
            series_opts = ["Portfolio", "Equity", "Loan", "Margin usage %"]
            charts.render_classic_chart(
                tax_adj_port_series, final_adj_series, loan_series, 
                tax_adj_equity_pct_series, tax_adj_usage_series, 
                series_opts, log_scale,
                bench_series=bench_resampled,
                comparison_series=comp_resampled,
                effective_rate_series=effective_rate_series
            )
        elif chart_style == "Classic (Dashboard)":
            log_opts = config.get('log_opts', {})
            charts.render_dashboard_view(
                tax_adj_port_series, final_adj_series, loan_series, 
                tax_adj_equity_pct_series, tax_adj_usage_series, 
                wmaint, stats, log_opts,
                bench_series=bench_resampled,
                comparison_series=comp_resampled,
                start_val=start_val,
                rate_annual=rate_annual
            )
        else: # Candlestick
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
                comparison_series=comp_resampled
            )       
        if pay_tax_cash:
            with st.expander("Detailed Cash Statistics", expanded=True):
                # Calculate Total Withdrawals
                # 1. Taxes
                total_tax_paid = final_tax_series.sum() if not final_tax_series.empty else 0.0
                
                # 2. Monthly Draws
                total_draws = 0.0
                if draw_monthly > 0 and not equity_series.empty:
                    # Iterate dates to match exact logic
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
                if not usage_series.empty:
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
    
        charts.render_returns_analysis(
            tax_adj_port_series, 
            bench_series=bench_resampled,
            comparison_series=comp_resampled,
            unique_id=portfolio_name
        )
        
    with res_tab_rebal:
        st.warning("⚠️ **Note:** These trade calculations assume **Gross** portfolio values. Tax payments are NOT deducted by selling shares in this view (assumes taxes paid via margin or external cash).")
        # Determine correct frequency for chart view
        rebal_freq_for_chart = config.get('rebalance', 'Yearly')
        if rebal_freq_for_chart == "Custom":
            pass # We handle Custom logic inside render_rebalancing_analysis now
            
        # Scale composition_df to match tax_adj_port_series (Source of Truth for Metrics)
        if not composition_df.empty:
            # We must use .copy() to avoid modifying session state unexpectedly if needed, 
            # though composition_df is already a local variable here.
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
        
    with res_tab_mc:
        st.markdown("### 🔮 Monte Carlo Simulation (Historical Bootstrap)")
        st.info("Simulating **10-year future performance** based on your strategy's historical daily volatility. Assumes reinvestment of all returns.")
        
        
        # 1. Get Returns Data source
        # Priority:
        # A. Extended Portfolio Data (api_port_series) - Best for Long History (ZROZSIM back to 1960s)
        # B. TWR Series (Shadow Backtest) - Best for Accuracy (excludes cashflows) but might be short (2009+) in Hybrid Mode
        
        # We prefer Extended (A) for Monte Carlo generally, so we can capture regimes.
        # But if Cashflows are heavy, (A) is dirty (MWR).
        # Compromise: Use (A) if available for full history.
        
        # Actually, results['port_series'] IS the API extended series (Simulated).
        # results['twr_series'] is the shadow/tax series (Real/Short in Hybrid).
        
        # Let's default to port_series (Extended) to enable 1970s scenarios.
        extended_series = results.get("port_series")
        tax_twr_series = results.get("twr_series")
        
        daily_rets = pd.Series(dtype=float)
        source_label = "Unknown"
        
        # Decide which series to use
        # Logic: Use Extended if it covers more history than TWR
        use_extended = False
        
        if extended_series is not None and not extended_series.empty:
            if tax_twr_series is not None and not tax_twr_series.empty:
                 # Check start dates
                 if extended_series.index[0] < tax_twr_series.index[0]:
                     use_extended = True
                     source_label = "Extended Chart Data (Simulated)"
                 else:
                     # They start same time, prefer TWR (Cleaner)
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
             # Warn if cashflows exist
             if results.get("cashflow", 0.0) != 0:
                 st.caption(f"ℹ️ using **{source_label}** to maximize history. Note: Returns include cashflow effects (MWR).")
             else:
                 st.caption(f"ℹ️ Using **{source_label}**.")
        else:
             st.caption(f"ℹ️ Using **{source_label}**.")

        if daily_rets.empty:
             st.error("No return data available for Monte Carlo.")
             st.stop()
            
        # 2. Configuration
        c_sims, c_start, c_flow = st.columns(3)
        n_sims = c_sims.slider("Scenarios", 100, 5000, 1000, 100, help="More scenarios = smoother cone", key=f"mc_n_sims_{portfolio_name}")
        
        # Start Value Default
        def_start = results.get("start_val", 10000.0)
        sim_start = c_start.number_input("Start Value ($)", value=float(def_start), step=1000.0, key=f"mc_start_{portfolio_name}")
        
        # Cashflow Default (Normalize to Monthly)
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
            # Source Mode
            mc_mode = st.radio(
                "Source Data / Regime",
                ["Full History (Default)", "Historical Period Filter", "Stress Scenario", "Custom Parameters"],
                help="Choose how to generate future return paths.",
                key=f"mc_mode_{portfolio_name}"
            )
            
            # Mode Logic
            if mc_mode == "Historical Period Filter":
                # Get min/max dates from data
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
                
                # Preset Scenarios
                if "1970s" in scenario:
                    # Note: Original data may not go back this far!
                    # Ideally we'd warn, but let the engine filter handle empty data
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
                    
                # Warning about data availability
                min_avail = daily_rets.index.min().date()
                if pd.to_datetime(filter_start).date() < min_avail:
                    st.warning(f"⚠️ Your backtest data starts on {min_avail}. The selected scenario starts earlier ({filter_start}). The simulation will only use available data.")
                    
            elif mc_mode == "Custom Parameters":
                c_p1, c_p2 = st.columns(2)
                custom_mean = c_p1.number_input("Expected Annual Return (%)", value=7.0, step=0.5, key=f"mc_custom_return_{portfolio_name}") / 100.0
                custom_vol = c_p2.number_input("Expected Annual Volatility (%)", value=15.0, step=0.5, key=f"mc_custom_vol_{portfolio_name}") / 100.0
                st.info("Generates synthetic returns using Normal Distribution (IID). Ignores historical data patterns.")
            
            # Bootstrap Method (Only if using History)
            if mc_mode != "Custom Parameters":
                st.markdown("##### Sampling Method")
                boot_method = st.radio("Method", ["Simple Bootstrap (IID)", "Block Bootstrap"], horizontal=True, key=f"mc_boot_method_{portfolio_name}")
                if boot_method == "Block Bootstrap":
                    block_size = st.slider("Block Size (Days)", min_value=5, max_value=60, value=20, help="Larger blocks preserve longer-term market memory (volatility clustering).", key=f"mc_block_size_{portfolio_name}")
                    st.caption(f"Sampling contiguous blocks of {block_size} days.")
    
        # --- TABS: Standard vs Seasonal ---
        mc_tab_std, mc_tab_seas = st.tabs(["Standard Simulation (10 Yr)", "📅 Seasonal Analysis (1 Yr)"])
        
        with mc_tab_std:
            # 3. Run Simulation (Standard)
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
             st.markdown("### 📅 Typical Year Analysis (Seasonal Bootstrap)")
             st.info("This simulation builds a 'Typical Year' by sampling January returns only from historical Januaries, February entries from Februaries, etc. This reveals seasonal patterns like 'Sell in May' or 'Santa Rally'.")
             
             if st.button("Run Seasonal Analysis (5,000 Runs)", key=f"mc_run_seasonal_{portfolio_name}"):
                 with st.spinner("Analyzing Seasonality..."):
                     # Uses the same source data as main MC (e.g. Extended History if avail)
                     seas_df = monte_carlo.run_seasonal_monte_carlo(daily_rets, n_sims=5000, initial_val=sim_start, monthly_cashflow=sim_monthly_add)
                     
                     if not seas_df.empty:
                         # Plot

                         fig = go.Figure()
                         
                         # Create a dummy date range for a typical year (using 2024 as generic leap year)
                         # This allows Plotly to show dates in the tooltip header automatically.
                         x_axis = pd.bdate_range(start='2024-01-01', periods=len(seas_df))
                         
                         # Values are already in DOLLARS from the engine
                         p90_vals = seas_df["P90"]
                         p75_vals = seas_df["P75"]
                         p25_vals = seas_df["P25"]
                         p10_vals = seas_df["P10"]
                         median_vals = seas_df["Median"]
                         
                         # Custom Hover Template
                         # Note: In 'x unified' hovermode, the LAST added trace appears at the TOP.
                         # We want P90 at top, so we add P10 first, then P25... finishing with P90.
                         
                         # 1. P10 (Pessimistic) - Bottom of List
                         fig.add_trace(go.Scatter(
                             x=x_axis, y=p10_vals,
                             mode='lines', 
                             line=dict(color='rgba(255, 50, 50, 0.5)', width=1, dash='dash'), # Red
                             name='P10 (Pessimistic)',
                             hovertemplate='<b>P10</b>: $%{y:,.0f}<extra></extra>'
                         ))

                         # 2. P25
                         fig.add_trace(go.Scatter(
                             x=x_axis, y=p25_vals,
                             mode='lines', 
                             line=dict(color='rgba(255, 165, 0, 0.5)', width=1, dash='dot'), # Orange
                             name='P25 (Mod. Downside)',
                             hovertemplate='<b>P25</b>: $%{y:,.0f}<extra></extra>'
                         ))

                         # 3. Median (Main)
                         fig.add_trace(go.Scatter(
                             x=x_axis, y=median_vals,
                             mode='lines', 
                             line=dict(color='#00C8FF', width=3), # Cyan
                             name='Median',
                             hovertemplate='<b>Median</b>: $%{y:,.0f}<extra></extra>'
                         ))

                         # 4. P75
                         fig.add_trace(go.Scatter(
                             x=x_axis, y=p75_vals,
                             mode='lines', 
                             line=dict(color='rgba(200, 200, 200, 0.5)', width=1, dash='dot'),
                             name='P75 (Mod. Upside)',
                             hovertemplate='<b>P75</b>: $%{y:,.0f}<extra></extra>'
                         ))

                         # 5. P90 (Optimistic) - Top of List
                         fig.add_trace(go.Scatter(
                             x=x_axis, y=p90_vals,
                             mode='lines', 
                             line=dict(color='rgba(255, 215, 0, 0.5)', width=1, dash='dash'), # Gold dashed
                             name='P90 (Optimistic)',
                             hovertemplate='<b>P90</b>: $%{y:,.0f}<extra></extra>'
                         ))
                         
                         # Shading (Fills) using invisible traces if needed, or fill='tonexty' on lines
                         # Let's add separate fill traces to keep lines clean in legend
                         # Fill 10-90
                         fig.add_trace(go.Scatter(
                             x=x_axis, y=p90_vals,
                             mode='lines', line=dict(width=0), showlegend=False, hoverinfo='skip'
                         ))
                         fig.add_trace(go.Scatter(
                             x=x_axis, y=p10_vals,
                             mode='lines', line=dict(width=0),
                             fill='tonexty', fillcolor='rgba(255, 215, 0, 0.05)', # Very faint gold/yellow
                             showlegend=False, hoverinfo='skip'
                         ))
                         
                         # Fill 25-75 (Darker center)
                         fig.add_trace(go.Scatter(
                             x=x_axis, y=p75_vals,
                             mode='lines', line=dict(width=0), showlegend=False, hoverinfo='skip'
                         ))
                         fig.add_trace(go.Scatter(
                             x=x_axis, y=p25_vals,
                             mode='lines', line=dict(width=0),
                             fill='tonexty', fillcolor='rgba(0, 200, 255, 0.1)', # Cyan tint
                             showlegend=False, hoverinfo='skip'
                         ))
                         
                         fig.update_layout(
                             title=f"Seasonal Performance Cone (Based on ${sim_start:,.0f})",
                             xaxis=dict(
                                 title="Month (Typical Year)",
                                 tickformat="%b",      # Axis labels: Jan, Feb...
                                 hoverformat="%b %d",  # Tooltip header: Jan 01, Jan 02...
                                 dtick="M1"            # Force monthly ticks
                             ),
                             yaxis=dict(title="Portfolio Value ($)", tickprefix="$"),
                             template="plotly_dark",
                             height=500,
                             hovermode="x unified",
                             legend=dict(orientation="h", y=1.02, x=0.5, xanchor="center")
                         )
                         
                         st.plotly_chart(fig, use_container_width=True)
                         
                         # Stats
                         final_med = seas_df["Median"].iloc[-1]
                         st.metric("Typical Year Ending Balance", f"${final_med:,.0f}")
                     else:
                         st.error("Not enough data for seasonality.")
        
        # Log detailed scenario results to .csv file
        try:
            os.makedirs("debug_tools", exist_ok=True)
            log_path = "debug_tools/monte_carlo_scenarios.csv"
            
            # Extract data
            paths_df = mc_results["paths"]
            final_vals = paths_df.iloc[-1]
            tot_inv = mc_results["metrics"]["total_invested"]
            
            # New Metrics
            path_metrics = mc_results.get("path_metrics", {})
            max_dds = path_metrics.get("max_dd", [0]*n_sims)
            twrs = path_metrics.get("final_twr", [1]*n_sims)
            
            # Calculate Percentiles (Rank within distribution)
            # pct=True gives 0.0 to 1.0
            ranks = final_vals.rank(pct=True)
            
            with open(log_path, "w", newline='') as f:
                writer = csv.writer(f)
                # CSV Header
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
    

        
    with res_tab_debug:
        st.subheader("Debug Info")
        
        st.divider()
        st.json(logs)
        st.write("Raw API Response (First 5 items):")
        st.write(str(raw_response)[:1000])
        
        # Also provide download button
        json_str = json.dumps(raw_response, indent=2)
        st.download_button(
            label="Download Raw Response",
            data=json_str,
            file_name=f"testfol_api_response_{portfolio_name}.json",
            mime="application/json",
            key=f"dl_json_{portfolio_name}"
        )
    
    # -------------------------------------------------------------------------
    # 5. Charts
    # -------------------------------------------------------------------------
    st.divider()
    # Chart Controls
    col_c1, col_c2, col_c3 = st.columns([2, 1, 1])
    
    # -----------------------------------------------------------------------------
    # Report Generation (Sidebar)
    # -----------------------------------------------------------------------------
    if results:
        st.sidebar.markdown("---")
        st.sidebar.subheader("📄 Report")
        
        # Generate report on the fly
        try:
            # Use combined results logic
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

import plotly.graph_objects as go
import plotly.express as px
import streamlit as st
import pandas as pd
import numpy as np
import os

from app.core import tax_library

@st.cache_data(show_spinner=False)
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
    
    detail_df = pl_by_year.copy()
    if isinstance(detail_df, pd.Series):
        detail_df = detail_df.to_frame(name="Realized P&L")
        
    detail_df["Federal Tax"] = fed_tax_series
    detail_df["State Tax"] = state_tax_series
    detail_df["Total Tax"] = total_tax_series
    detail_df["Net P&L"] = detail_df["Realized P&L"] - detail_df["Total Tax"]
    
    st.dataframe(
        detail_df.style.format("${:,.2f}"),
        use_container_width=True
    )


# -----------------------------------------------------------------------------
# Monte Carlo Visualization
# -----------------------------------------------------------------------------
def render_monte_carlo_view(mc_results, unique_id=None):
    """
    Renders the Monte Carlo 'Cone of Uncertainty' chart.
    """
    if not mc_results or mc_results.get("percentiles").empty:
        st.info("Insufficient data for Monte Carlo simulation.")
        return

    df = mc_results["percentiles"]
    m = mc_results["metrics"]
    initial_val = m["initial_val"]
    
    # 1. Metrics Header
    # Row 1: Performance Range
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Median Final Value", f"${m['median_final']:,.0f}", help="50th Percentile Outcome")
    c2.metric("Median CAGR", f"{m['cagr_median']:.1%}", help="Annualized return of the median path")
    
    multiplier = m['median_final'] / m.get('total_invested', initial_val)
    c3.metric("Growth Multiple", f"{multiplier:.1f}x", help="Final Value / Total Invested Capital (Initial + Monthly Adds)")
    c4.metric("CAGR Range (P10-P90)", f"{m['cagr_p10']:.1%} - {m['cagr_p90']:.1%}", help="Pessimistic to Optimistic Annual Return")

    # Row 2: Risk Analysis
    st.markdown("##### ⚠️ Risk Analysis")
    r1, r2, r3, r4 = st.columns(4)
    
    r1.metric("Median Max Drawdown", f"{m['max_dd_median']:.1%}", help="Expected peak-to-trough decline")
    r2.metric("Worst Case DD (P90)", f"{m['max_dd_p90']:.1%}", help="90th Percentile Max Drawdown (Severe Crash)")
    r3.metric("Chance of Loss", f"{m['prob_loss']:.1%}", help="Probability of ending lower than starting value")
    r4.metric("Best Case DD (P10)", f"{m['max_dd_p10']:.1%}", help="10th Percentile Max Drawdown (Mild Correction)")
    
    # 2. Options
    c1, c2 = st.columns([1, 4])
    key_suffix = f"_{unique_id}" if unique_id else ""
    use_log = c1.toggle("Log Scale", help="Use logarithmic scale to see percentage changes better", key=f"mc_log_scale{key_suffix}")
    show_paths = c1.toggle("Show Paths", help="Display 100 random individual simulation paths (Spaghetti Chart)", key=f"mc_show_paths{key_suffix}")
    
    # 3. Plotly Fan Chart
    fig = go.Figure()
    
    x = df.index
    
    # Optional: Spaghetti Paths (Behind the cone)
    if show_paths and "paths" in mc_results:
        paths_df = mc_results["paths"]
        # Sample up to 100 paths (Visual sampling cap for performance)
        n_sample = min(100, paths_df.shape[1])
        if n_sample > 0:
            sample_cols = np.random.choice(paths_df.columns, n_sample, replace=False)
            sample_paths = paths_df[sample_cols]
            
            for col in sample_cols:
                fig.add_trace(go.Scatter(
                    x=x, y=sample_paths[col],
                    mode='lines',
                    line=dict(width=1, color='rgba(100, 255, 255, 0.15)'), # Faint cyan (increased opacity)
                    showlegend=False,
                    hoverinfo='skip' # Don't hover on spaghetti
                ))
    
    # --- Background Fill (Invisible Helper Traces) ---
    # Draw P10 then P90 to create the shaded cone "behind" everything
    fig.add_trace(go.Scatter(
        x=x, y=df["P10"],
        mode='lines',
        line=dict(width=0),
        showlegend=False,
        hoverinfo='skip'
    ))
    fig.add_trace(go.Scatter(
        x=x, y=df["P90"],
        mode='lines',
        line=dict(width=0),
        fill='tonexty',
        fillcolor='rgba(255, 215, 0, 0.15)', # Faint Gold
        showlegend=False,
        hoverinfo='skip'
    ))

    # --- Helper & Pre-calc ---
    def fmt_currency_custom(val):
        if abs(val) >= 1e9: return f"${val/1e9:.2f}B" # 2 decimals for Billions
        if abs(val) >= 1e6: return f"${val/1e6:.1f}M" # 1 decimal for Millions
        if abs(val) >= 1e3: return f"${val/1e3:.0f}k"
        return f"${val:.0f}"
        
    def calculate_nice_ticks(min_v, max_v, is_log=False):
        """Generate human-friendly tick values for axes."""
        if is_log:
            # Powers of 10
            start_exp = np.floor(np.log10(max(1, min_v)))
            end_exp = np.ceil(np.log10(max(1, max_v)))
            ticks = 10 ** np.arange(start_exp, end_exp + 1)
            # Filter to relevant range
            ticks = ticks[(ticks >= min_v * 0.5) & (ticks <= max_v * 1.5)]
            return ticks
        else:
            # Simple linear grid (5-6 ticks)
            return np.linspace(0, max_v, num=6)

    # Pre-calculate custom labels for tooltips
    chart_cols = ["Median", "P10", "P90", "P25", "P75"]
    for col in chart_cols:
        df[f"{col}_chk"] = df[col].apply(fmt_currency_custom)
        
    # Calculate Axis Ticks (Force 'B' suffix)
    y_max_main = df["P90"].max()
    y_min_main = df["P10"].min()
    main_ticks = calculate_nice_ticks(y_min_main, y_max_main, use_log)
    main_tickdata = [fmt_currency_custom(x) for x in main_ticks]

    # --- Visible Lines (Ordered Bottom-to-Top for Z-Order) ---
    
    # 1. P10 (Red) - Bottom
    fig.add_trace(go.Scatter(
        x=x, y=df["P10"],
        mode='lines',
        line=dict(width=1, color='rgba(255, 100, 100, 0.5)', dash='dash'),
        name='P10',
        customdata=df["P10_chk"],
        hovertemplate="<b>P10 (Pessimistic)</b>: %{customdata}<extra></extra>",
        showlegend=True
    ))

    # 2. P25 (Orange)
    fig.add_trace(go.Scatter(
        x=x, y=df["P25"],
        mode='lines',
        line=dict(width=1, color='rgba(255, 165, 0, 0.5)', dash='dot'),
        name='P25',
        customdata=df["P25_chk"],
        hovertemplate="<b>P25 (Mod. Downside)</b>: %{customdata}<extra></extra>"
    ))

    # 3. Median (Cyan)
    fig.add_trace(go.Scatter(
        x=x, y=df["Median"],
        mode='lines',
        line=dict(color='cyan', width=2),
        name='Median',
        customdata=df["Median_chk"],
        hovertemplate="<b>Median</b>: %{customdata}<extra></extra>"
    ))

    # 4. P75 (Silver)
    fig.add_trace(go.Scatter(
        x=x, y=df["P75"],
        mode='lines',
        line=dict(width=1, color='silver', dash='dot'),
        name='P75',
        customdata=df["P75_chk"],
        hovertemplate="<b>P75 (Mod. Upside)</b>: %{customdata}<extra></extra>"
    ))

    # 5. P90 (Gold) - Top
    fig.add_trace(go.Scatter(
        x=x, y=df["P90"],
        mode='lines',
        line=dict(width=1, color='#FFD700', dash='dash'), 
        name='P90',
        customdata=df["P90_chk"],
        hovertemplate="<b>P90 (Optimistic)</b>: %{customdata}<extra></extra>"
    ))
    
    fig.update_layout(
        title=f"Monte Carlo Simulation (10 Years)",
        yaxis_title="Portfolio Value ($)",
        xaxis_title="Date",
        hovermode="x unified", # Force sort by Value (descending)
        height=600,
        yaxis=dict(
            type='log' if use_log else 'linear', 
            tickformat='$.2s', # Compact currency format
            gridcolor='rgba(128,128,128,0.2)'
        ),
        xaxis=dict(gridcolor='rgba(128,128,128,0.2)'),
        template="plotly_dark",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    st.caption("""
    **Methodology:** Historical Bootstrap using your strategy's daily volatility.
    *   **Median Path:** The central outcome (50% probability).
    *   **Outer Cone (Shaded):** The broad range of probable outcomes (**10th** to **90th** percentile).
    *   **Inner Lines (Dashed):** The "likely" zone (**25th** to **75th** percentile), capturing the middle 50% of scenarios.
    *   **Visualization:** For browser performance, we display a random sample of **up to 100 paths** (spaghetti), but metrics use all simulated scenarios.
    *   **Assumptions:** Projection based on the 'Start Value' and 'Monthly Add' configured above. Reinvests all dividends/returns.
    """)
    
    # 4. Distribution Chart (Histogram) (Clipped to P95 for better focus)
    st.markdown("##### 📊 Distribution of Outcomes (Year 10)")
    path_finals = mc_results["paths"].iloc[-1]
    
    # Clip outliers for visualization (Show up to P95)
    p95_val = np.percentile(path_finals, 95)
    
    # --- Manual Binning for Robust Scale Handling ---
    if use_log:
        # Log Mode: Bin in log-space, then map back to dollars
        safe_data = path_finals[path_finals > 0] 
        log_data = np.log10(safe_data)
        counts, bin_edges_log = np.histogram(log_data, bins=100)
        
        # Calculate widths in linear domain (critical for go.Bar on log axis)
        bin_edges = 10**bin_edges_log
        widths = np.diff(bin_edges)
        ctrs = (bin_edges[:-1] + bin_edges[1:]) / 2
        
        x_plot = ctrs
        y_plot = counts
        bar_widths = widths
        
        xaxis_config = dict(
            tickformat='$.2s', 
            gridcolor='rgba(128,128,128,0.2)',
            type='log'
        )
    else:
        # Linear Mode: Bin the clipped data
        data_clipped = path_finals[path_finals <= p95_val * 1.05]
        data_clipped = data_clipped[data_clipped >= 0] 
        
        counts, bin_edges = np.histogram(data_clipped, bins=100)
        widths = np.diff(bin_edges)
        x_plot = (bin_edges[:-1] + bin_edges[1:]) / 2
        y_plot = counts
        bar_widths = widths
        
        xaxis_config = dict(
            tickformat='$.2s', 
            gridcolor='rgba(128,128,128,0.2)',
            type='linear',
            range=[0, p95_val * 1.05] 
        )
    
    # Calculate Manual Ticks for Histogram (to fix 'G')
    hist_min = np.min(x_plot) if use_log else 0
    hist_max = np.max(x_plot) if use_log else p95_val * 1.05
    hist_ticks = calculate_nice_ticks(hist_min, hist_max, use_log)
    hist_tickdata = [fmt_currency_custom(x) for x in hist_ticks]
    
    # Enable manual ticks
    xaxis_config["tickmode"] = "array"
    xaxis_config["tickvals"] = hist_ticks
    xaxis_config["ticktext"] = hist_tickdata

    fig_hist = go.Figure()
    
    # Render as Explicit Bar Chart (for Separator Lines)
    hist_custom_labels = [fmt_currency_custom(v) for v in x_plot]
    
    fig_hist.add_trace(go.Bar(
        x=x_plot, y=y_plot,
        width=bar_widths, # Explicit widths for log scale correctness
        marker_color='rgba(0, 150, 150, 0.6)', # Darker Cyan Fill
        marker_line_color='rgba(200, 255, 255, 0.8)', # Bright separation lines
        marker_line_width=1,
        name='Frequency',
        customdata=hist_custom_labels,
        hovertemplate="<b>Value</b>: %{customdata}<br><b>Count</b>: %{y}<extra></extra>"
    ))
    
    # Add Percentile Lines
    df_pct = mc_results["percentiles"]
    
    # Stagger labels to avoid overlap (3 tiers)
    annotations = [
        (m['median_final'], 'cyan', 'Median', 1.20),      # Top Tier
        (df_pct['P25'].iloc[-1], 'orange', 'P25', 1.12),  # Mid Tier
        (df_pct['P75'].iloc[-1], 'silver', 'P75', 1.12),  # Mid Tier
        (df_pct['P10'].iloc[-1], 'red', 'P10', 1.05),     # Low Tier
        (df_pct['P90'].iloc[-1], '#FFD700', 'P90', 1.05)  # Low Tier
    ]
    
    # Calculate max frequency for line height scaling
    max_freq = np.max(y_plot) if len(y_plot) > 0 else 1
    
    for val, color, label, y_pos in annotations:
        # Use dot for P25/P75 to match main chart, dash for others
        dash_style = "dot" if "P25" in label or "P75" in label else "dash"
        
        # 1. Interactive Line (Trace) for Tooltip
        fmt_val = fmt_currency_custom(val)
        fig_hist.add_trace(go.Scatter(
            x=[val, val], 
            y=[0, max_freq], # Draw line up to the peak frequency
            mode='lines',
            line=dict(color=color, width=2, dash=dash_style),
            name=label,
            hovertemplate=f"<b>{label}</b>: {fmt_val}<extra></extra>", # Use pre-formatted string
            showlegend=False
        ))
        
        # 2. Text Label (Annotation)
        fig_hist.add_annotation(
            x=val, y=y_pos, yref="paper", # Staggered positions
            text=f"<b>{label}</b>", 
            showarrow=False,
            font=dict(color=color, size=11),
            yshift=0
        )

    fig_hist.update_layout(
        height=350,
        xaxis_title="Final Portfolio Value ($)",
        yaxis_title="Frequency (Scenarios)",
        template="plotly_dark",
        bargap=0.1,
        xaxis=xaxis_config,
        yaxis=dict(gridcolor='rgba(128,128,128,0.2)'),
        margin=dict(t=60), # More space for 3-tier annotations
        hovermode="x" # Snap to x-axis (easier to hit lines)
    )
    
    st.plotly_chart(fig_hist, use_container_width=True)


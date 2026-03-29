import plotly.graph_objects as go
import plotly.express as px
import streamlit as st
import pandas as pd
import numpy as np
import datetime as dt

from app.common.utils import color_return
from app.core import tax_library, calculations

# 20 colors ordered so that adjacent entries have maximum perceptual contrast.
# Alternates between warm/cool and light/dark to avoid neighbor confusion
# even with 13+ tickers stacked or overlaid.
_DISTINCT_COLORS = [
    "#636EFA",  # blue
    "#EF553B",  # red
    "#00CC96",  # teal
    "#FFA15A",  # orange
    "#AB63FA",  # purple
    "#FECB52",  # yellow
    "#19D3F3",  # cyan
    "#C4451C",  # rust
    "#B6E880",  # lime
    "#FF6692",  # hot pink
    "#325A9B",  # navy
    "#86CE00",  # chartreuse
    "#FF97FF",  # magenta
    "#85660D",  # brown
    "#1CFFCE",  # aquamarine
    "#FEAF16",  # amber
    "#782AB6",  # violet
    "#F8A19F",  # salmon
    "#1CBE4F",  # emerald
    "#DEA0FD",  # lavender
]


def _build_color_map(tickers: list[str]) -> dict[str, str]:
    """Assign a distinct color to each ticker, consistent across charts."""
    return {t: _DISTINCT_COLORS[i % len(_DISTINCT_COLORS)] for i, t in enumerate(tickers)}


def _hex_to_rgba(hex_color: str, alpha: float = 0.55) -> str:
    """Convert '#RRGGBB' to 'rgba(R,G,B,alpha)'."""
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"

def render_rebalance_sankey(trades_df, view_freq="Yearly", unique_id=None):
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
    elif view_freq == "Per Event":
        df["Period"] = df["Date"].dt.strftime('%Y-%m-%d')
        
    # Period Selection
    periods = sorted(df["Period"].unique(), reverse=True)
    key_suffix = f"_{unique_id}" if unique_id else ""
    selected_period = st.selectbox("Select Period for Flow", periods, index=0, key=f"rebal_period_selector{key_suffix}")
    
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

@st.cache_data(show_spinner=False)
def render_portfolio_composition(composition_df, view_freq="Yearly"):
    if composition_df.empty:
        return

    st.subheader("Portfolio Composition")
    
    # Ensure sorted by date
    df = composition_df.sort_values(["Date", "Ticker"])
    
    # Filtering Logic based on View Frequency to avoid summing multiple snapshots in one bar
    if view_freq == "Yearly":
        # Keep only the last snapshot available for each Year
        last_dates = df.groupby(df['Date'].dt.year)['Date'].max()
        df = df[df['Date'].isin(last_dates)]
    elif view_freq == "Quarterly":
        # Keep last snapshot for each Quarter
        last_dates = df.groupby(df['Date'].dt.to_period('Q'))['Date'].max()
        df = df[df['Date'].isin(last_dates)]
    elif view_freq == "Monthly":
        # Keep last snapshot for each Month
        last_dates = df.groupby(df['Date'].dt.to_period('M'))['Date'].max()
        df = df[df['Date'].isin(last_dates)]
    elif view_freq == "Per Event":
        # Keep every unique rebalance snapshot date
        pass

    # Format Date for the axis labeling (categorical)
    df['Date Label'] = df['Date'].dt.strftime('%Y-%m-%d')
    
    # Calculate Total Value per date for hover display
    totals = df.groupby('Date Label')['Value'].sum().rename('Total Value')
    df = df.merge(totals, on='Date Label')

    df = df.sort_values(["Date Label", "Value"], ascending=[True, True])
    
    # Build consistent color map for tickers
    unique_tickers = sorted(df["Ticker"].unique())
    cmap = _build_color_map(unique_tickers)

    fig = px.bar(
        df,
        y="Date Label",
        x="Value",
        color="Ticker",
        color_discrete_map=cmap,
        title=f"Portfolio Value by Asset (Pre-Rebalance, {view_freq})",
        text_auto="$.2s",
        orientation='h',
        template="plotly_dark",
        custom_data=["Total Value"]
    )
    
    fig.update_traces(
        hovertemplate="<b>%{fullData.name}</b><br>Asset Value: %{x:$,.0f}<br>Total Portfolio: %{customdata[0]:$,.0f}<extra></extra>"
    )
    
    fig.update_layout(
        xaxis_title="Value ($)",
        yaxis_title="Rebalance Date",
        legend_title="Asset",
        height=min(1200, max(400, len(df['Date Label'].unique()) * 30)), # Dynamic height
        yaxis=dict(type='category', categoryorder='category ascending') # Recent at top (Y-axis ascending puts largest/latest at top)
    )
    
    st.plotly_chart(fig, use_container_width=True)

    # --- Data Table View ---
    with st.expander(f"📋 Portfolio Composition Details ({view_freq})", expanded=False):
        # Pivot the data for a clean table view: Dates as rows, Tickers as columns
        table_df = df.pivot(index='Date Label', columns='Ticker', values='Value').fillna(0)
        
        # Add Total Column
        table_df['Total'] = table_df.sum(axis=1)
        
        # Sort by date descending (latest at top)
        table_df = table_df.sort_index(ascending=False)
        
        # Format for display
        st.dataframe(
            table_df.style.format("${:,.0f}"),
            use_container_width=True
        )


def render_portfolio_allocation(
    component_prices: pd.DataFrame,
    allocation: dict,
    composition_df: pd.DataFrame,
    start_val: float,
    unique_id: str = "",
    port_series: pd.Series | None = None,
    rebal_config: dict | None = None,
):
    """Render a 100% stacked area chart showing portfolio allocation drift over time."""
    if component_prices.empty or not allocation:
        return

    # Map full tickers (e.g. AAPL?L=2) to base tickers used in component_prices columns
    # Also extract leverage and expense ratio modifiers per base ticker
    weights: dict[str, float] = {}
    leverage: dict[str, float] = {}
    expense_ratio: dict[str, float] = {}
    for full_tk, weight in allocation.items():
        base = full_tk.split("?")[0]
        weights[base] = weights.get(base, 0) + weight
        # Parse modifiers
        if "?" in full_tk:
            query = full_tk.split("?", 1)[1]
            for pair in query.split("&"):
                if "=" in pair:
                    k, v = pair.split("=", 1)
                    try:
                        if k.upper() == "L":
                            leverage[base] = float(v)
                        elif k.upper() in ("E", "D"):
                            expense_ratio[base] = float(v)
                    except ValueError:
                        pass

    available = [t for t in weights if t in component_prices.columns]
    if not available:
        return

    raw_prices = component_prices[available].dropna(how="all").ffill().dropna(how="any")
    if raw_prices.empty or len(raw_prices) < 2:
        return

    # Apply leverage and expense ratio to daily returns to get modified prices
    daily_returns = raw_prices.pct_change()
    modified_returns = daily_returns.copy()
    for t in available:
        lev = leverage.get(t, 1.0)
        er = expense_ratio.get(t, 0.0)
        daily_er = (er / 100.0) / 252.0 if er > 0 else 0.0
        if lev != 1.0 or daily_er > 0:
            modified_returns[t] = daily_returns[t] * lev - daily_er

    # Reconstruct modified price series from returns
    prices = (1 + modified_returns).cumprod() * raw_prices.iloc[0]
    prices.iloc[0] = raw_prices.iloc[0]  # anchor first row

    total_weight = sum(weights[t] for t in available)
    if total_weight <= 0:
        return

    # Get rebalance dates from composition_df, filling gaps with synthetic dates
    # generated from the portfolio's rebalance config.
    # The shadow backtest may only have composition from when real ETF data starts
    # (e.g. ZROZ from 2009), but component_prices (SIM) go back further.
    rebal_dates: list[pd.Timestamp] = []
    if not composition_df.empty:
        rebal_dates = sorted(pd.to_datetime(composition_df["Date"].unique()))

    # If composition doesn't cover the full price range, generate synthetic
    # rebalance dates from portfolio config for the gap period.
    rc = rebal_config or {}
    idx = prices.index
    if rebal_dates and (rebal_dates[0] - idx[0]).days > 730:
        freq = rc.get("freq", "Yearly")
        rebal_month = rc.get("month", 1)
        rebal_day = rc.get("day", 1)

        gap_end = rebal_dates[0]
        synthetic: list[pd.Timestamp] = []
        if freq == "Monthly":
            for yr in range(idx[0].year, gap_end.year + 1):
                for mo in range(1, 13):
                    target = pd.Timestamp(yr, mo, min(rebal_day, 28))
                    if idx[0] < target < gap_end:
                        loc = idx.searchsorted(target)
                        if 0 < loc < len(idx):
                            synthetic.append(idx[loc])
        elif freq == "Quarterly":
            for yr in range(idx[0].year, gap_end.year + 1):
                for mo in [1, 4, 7, 10]:
                    target = pd.Timestamp(yr, mo, min(rebal_day, 28))
                    if idx[0] < target < gap_end:
                        loc = idx.searchsorted(target)
                        if 0 < loc < len(idx):
                            synthetic.append(idx[loc])
        else:  # Yearly
            for yr in range(idx[0].year + 1, gap_end.year + 1):
                target = pd.Timestamp(yr, rebal_month, min(rebal_day, 28))
                if idx[0] < target < gap_end:
                    loc = idx.searchsorted(target)
                    if 0 < loc < len(idx):
                        synthetic.append(idx[loc])
        rebal_dates = sorted(set(synthetic + rebal_dates))

    # Build segment boundaries: [start, rebal_1, rebal_2, ..., end]
    seg_starts = [prices.index[0]]
    for rd in rebal_dates:
        # Snap to nearest trading day in prices
        idx_loc = prices.index.searchsorted(rd)
        if 0 < idx_loc < len(prices.index):
            snapped = prices.index[idx_loc]
            if snapped > seg_starts[-1]:
                seg_starts.append(snapped)

    # Compute daily position values segment by segment
    all_positions = []
    for i, seg_start in enumerate(seg_starts):
        seg_end = seg_starts[i + 1] if i + 1 < len(seg_starts) else prices.index[-1]

        if i + 1 < len(seg_starts):
            seg_prices = prices.loc[seg_start:seg_end].iloc[:-1]  # exclude next segment start
        else:
            seg_prices = prices.loc[seg_start:]

        if seg_prices.empty:
            continue

        # Determine total value at segment start
        if i == 0:
            total_val = start_val
        else:
            # Use end of previous segment's total value
            prev_end = all_positions[-1].iloc[-1] if all_positions else None
            total_val = prev_end.sum() if prev_end is not None else start_val

        # Vectorized: position(t) = (alloc_value / start_price) * price(t)
        start_prices = seg_prices.iloc[0]
        seg_pos = pd.DataFrame(index=seg_prices.index, columns=available, dtype=float)
        for t in available:
            if start_prices[t] > 0:
                alloc_val = total_val * (weights[t] / total_weight)
                seg_pos[t] = alloc_val * (seg_prices[t] / start_prices[t])
            else:
                seg_pos[t] = 0.0

        all_positions.append(seg_pos)

    if not all_positions:
        return

    positions = pd.concat(all_positions)
    # Remove any duplicate indices (shouldn't happen but safety)
    positions = positions[~positions.index.duplicated(keep="first")]

    row_totals = positions.sum(axis=1)

    # Scale positions to match actual gross portfolio value (port_series).
    # The position tracking from component_prices approximates leveraged growth,
    # but may diverge from the actual backtest due to data source differences,
    # dividend handling, etc.  Scaling preserves the relative allocation
    # proportions while anchoring the total to the real portfolio value.
    if port_series is not None and not port_series.empty:
        aligned_port = port_series.reindex(positions.index).ffill().bfill()
        scale = aligned_port / row_totals.replace(0, np.nan)
        scale = scale.fillna(1.0)
        for t in available:
            positions[t] = positions[t] * scale
        row_totals = positions.sum(axis=1)

    # Build consistent color map for all charts below (sorted to match composition bar chart)
    cmap = _build_color_map(sorted(available))

    key_suffix = f"_{unique_id}" if unique_id else ""

    # --- Correlation Overlay ---
    corr_window_options = {
        "1 month": 21, "3 months": 63, "6 months": 126,
        "1 year": 252, "2 years": 504, "3 years": 756, "5 years": 1260,
    }
    show_correlation = len(available) >= 2

    if show_correlation:
        corr_window_label = st.selectbox(
            "Correlation Window",
            list(corr_window_options.keys()),
            index=3,  # default "1 year"
            key=f"corr_window{key_suffix}",
            help=(
                "**Purple line** — Rolling average pairwise correlation between all assets.\n\n"
                "• **+100** = assets move perfectly together (no diversification)\n"
                "• **0** = no relationship between asset movements\n"
                "• **-100** = assets move in opposite directions (ideal diversification)\n\n"
                "**Red shading** = periods where ALL assets are declining simultaneously "
                "(diversification failure)."
            ),
        )
        corr_window = corr_window_options[corr_window_label]

        # Compute rolling pairwise correlations and average them
        ret = modified_returns[available].dropna()
        pairs = [(available[i], available[j]) for i in range(len(available)) for j in range(i+1, len(available))]
        pair_corrs = pd.DataFrame(index=ret.index)
        for a, b in pairs:
            pair_corrs[f"{a}_{b}"] = ret[a].rolling(corr_window, min_periods=corr_window // 2).corr(ret[b])
        avg_corr = pair_corrs.mean(axis=1).reindex(positions.index)

        # Detect all-assets-declining periods (use shorter of corr window or 63 days)
        decline_window = min(corr_window, 63)
        all_declining_mask = pd.Series(True, index=ret.index)
        for t in available:
            rolling_ret = ret[t].rolling(decline_window, min_periods=decline_window // 2).sum()
            all_declining_mask = all_declining_mask & (rolling_ret < 0)
        all_declining_mask = all_declining_mask.reindex(positions.index, fill_value=False)

    # --- Component Performance Chart (line chart showing each position's actual value) ---
    st.subheader("Component Performance")

    fig_lines = go.Figure()

    # Total portfolio line (market value = sum of component positions)
    fig_lines.add_trace(go.Scatter(
        x=row_totals.index,
        y=row_totals,
        name="Total Portfolio",
        mode="lines",
        line=dict(width=2.5, color="white", dash="dot"),
        hovertemplate="Total: $%{y:,.0f}<extra></extra>",
    ))

    for t in available:
        fig_lines.add_trace(go.Scatter(
            x=positions.index,
            y=positions[t],
            name=t,
            mode="lines",
            line=dict(width=1.5, color=cmap[t]),
            hovertemplate=f"{t}: $%{{y:,.0f}}<extra></extra>",
        ))

    # Rebalance markers with labels
    for rd in seg_starts[1:]:
        rd_ms = int(pd.Timestamp(rd).timestamp() * 1000)
        fig_lines.add_shape(
            type="line",
            x0=rd_ms, x1=rd_ms, y0=0, y1=1,
            xref="x", yref="paper",
            line=dict(dash="dot", color="rgba(255,255,255,0.25)", width=1),
        )
        fig_lines.add_annotation(
            x=rd_ms, y=1,
            xref="x", yref="paper",
            text="Rebal",
            showarrow=False,
            font=dict(size=8, color="rgba(255,255,255,0.5)"),
            textangle=-90,
            yanchor="bottom",
        )

    # Red shaded regions for all-declining periods
    if show_correlation and all_declining_mask.any():
        in_region = False
        region_start = None
        for date, declining in all_declining_mask.items():
            if declining and not in_region:
                region_start = date
                in_region = True
            elif not declining and in_region:
                fig_lines.add_vrect(
                    x0=region_start, x1=date,
                    fillcolor="rgba(239, 68, 68, 0.10)",
                    line=dict(color="rgba(239, 68, 68, 0.3)", width=1),
                    layer="below",
                )
                in_region = False
        if in_region and region_start is not None:
            fig_lines.add_vrect(
                x0=region_start, x1=positions.index[-1],
                fillcolor="rgba(239, 68, 68, 0.10)",
                line=dict(color="rgba(239, 68, 68, 0.3)", width=1),
                layer="below",
            )

    # Correlation line on secondary y-axis
    if show_correlation:
        fig_lines.add_trace(go.Scatter(
            x=avg_corr.index,
            y=avg_corr * 100,
            name="Avg Correlation",
            mode="lines",
            line=dict(width=1.5, color="#a78bfa"),
            yaxis="y2",
            opacity=0.85,
            hovertemplate="Correlation: %{y:.0f}<extra></extra>",
        ))

        # Dashed zero line for correlation axis
        fig_lines.add_shape(
            type="line",
            x0=0, x1=1, y0=0, y1=0,
            xref="paper", yref="y2",
            line=dict(dash="dash", color="rgba(167, 139, 250, 0.3)", width=1),
        )

        if all_declining_mask.any():
            fig_lines.add_trace(go.Scatter(
                x=[None], y=[None],
                name="All Declining",
                mode="markers",
                marker=dict(size=10, color="rgba(239, 68, 68, 0.5)", symbol="square"),
                showlegend=True,
            ))

    layout_kwargs = dict(
        yaxis=dict(
            title="Position Value ($)",
            type="log",
            tickprefix="$",
            tickformat=",",
            gridcolor="rgba(255,255,255,0.1)",
            minor=dict(showgrid=True, gridcolor="rgba(255,255,255,0.05)"),
        ),
        xaxis=dict(title="Date", gridcolor="rgba(255,255,255,0.1)"),
        template="plotly_dark",
        height=500,
        showlegend=True,
        hovermode="x unified",
        margin=dict(l=80, r=80, t=20, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=-0.25, xanchor="center", x=0.5),
    )
    if show_correlation:
        layout_kwargs["yaxis2"] = dict(
            title="Avg Correlation",
            overlaying="y",
            side="right",
            range=[-105, 105],
            showgrid=False,
            tickvals=[-100, -50, 0, 50, 100],
            titlefont=dict(color="#a78bfa"),
            tickfont=dict(color="#a78bfa"),
            zeroline=False,
        )
    fig_lines.update_layout(**layout_kwargs)
    st.plotly_chart(fig_lines, use_container_width=True, key=f"comp_perf{key_suffix}")

    # --- Per-Asset Correlation Breakdown ---
    if show_correlation and len(available) >= 3:
        # Classify assets: equity-like vs diversifiers
        _equity_keywords = {
            "SPY", "SPYSIM", "VOO", "VOOSIM", "VTI", "VTISIM", "QQQ", "QQQSIM",
            "SSO", "SSOSIM", "UPRO", "UPROSIM", "TQQQ", "TQQQSIM",
            "NDXMEGASIM", "NDXSIM", "NDX30SIM",
            "VUG", "VUGSIM", "VTV", "VTVSIM", "VO", "VOSIM",
            "VB", "VBSIM", "IWM", "IWMSIM",
        }
        equity_assets = [t for t in available if t.upper() in _equity_keywords]
        diversifiers = [t for t in available if t.upper() not in _equity_keywords]

        if equity_assets and diversifiers:
            with st.expander("Per-Asset Correlation vs Equity Core", expanded=False):
                # Compute equity core return (weight-averaged)
                eq_total_w = sum(weights.get(t, 1) for t in equity_assets)
                equity_ret = sum(
                    ret[t] * (weights.get(t, 1) / eq_total_w) for t in equity_assets
                )

                # Compute each diversifier's rolling correlation to equity core
                fig_breakdown = go.Figure()
                for t in diversifiers:
                    corr_vs_eq = ret[t].rolling(corr_window, min_periods=corr_window // 2).corr(equity_ret)
                    corr_vs_eq = corr_vs_eq.reindex(positions.index) * 100
                    fig_breakdown.add_trace(go.Scatter(
                        x=corr_vs_eq.index,
                        y=corr_vs_eq,
                        name=f"{t} ↔ Equities",
                        mode="lines",
                        line=dict(width=1.5, color=cmap.get(t, "#94a3b8")),
                        hovertemplate=f"{t} corr: %{{y:.0f}}<extra></extra>",
                    ))

                # Zero line
                fig_breakdown.add_shape(
                    type="line", x0=0, x1=1, y0=0, y1=0,
                    xref="paper", yref="y",
                    line=dict(dash="dash", color="rgba(255,255,255,0.3)", width=1),
                )

                fig_breakdown.update_layout(
                    yaxis=dict(
                        title="Correlation to Equity Core",
                        range=[-105, 105],
                        tickvals=[-100, -50, 0, 50, 100],
                        gridcolor="rgba(255,255,255,0.1)",
                    ),
                    xaxis=dict(title="Date", gridcolor="rgba(255,255,255,0.1)"),
                    template="plotly_dark",
                    height=350,
                    showlegend=True,
                    hovermode="x unified",
                    margin=dict(l=80, r=20, t=20, b=40),
                    legend=dict(orientation="h", yanchor="bottom", y=-0.3, xanchor="center", x=0.5),
                )
                st.plotly_chart(fig_breakdown, use_container_width=True, key=f"corr_breakdown{key_suffix}")
                st.caption(
                    "Rolling correlation of each diversifier vs the equity-weighted core. "
                    "**Positive** = moving with equities (pinned). "
                    "**Negative** = moving against equities (diversifying). "
                    f"Equity core: {', '.join(equity_assets)}."
                )

    # --- Percentage Stacked Area Chart ---
    pct = positions.div(row_totals, axis=0) * 100
    pct = pct.fillna(0)

    st.subheader("Portfolio Allocation")

    fig = go.Figure()
    for t in available:
        fig.add_trace(go.Scatter(
            x=pct.index,
            y=pct[t],
            name=t,
            mode="lines",
            stackgroup="one",
            fillcolor=_hex_to_rgba(cmap[t], 0.55),
            line=dict(width=0.5, color=cmap[t]),
            hovertemplate=f"{t}: %{{y:.2f}}%<extra></extra>",
        ))

    # Vertical lines at each rebalance date
    for rd in seg_starts[1:]:  # skip the first (portfolio start)
        rd_ms = int(pd.Timestamp(rd).timestamp() * 1000)
        fig.add_shape(
            type="line",
            x0=rd_ms, x1=rd_ms, y0=0, y1=100,
            xref="x", yref="y",
            line=dict(dash="dot", color="rgba(255,255,255,0.25)", width=1),
        )
        fig.add_annotation(
            x=rd_ms, y=100,
            xref="x", yref="y",
            text="Rebalance",
            showarrow=False,
            font=dict(size=8, color="rgba(255,255,255,0.5)"),
            textangle=-90,
            yanchor="bottom",
        )

    fig.update_layout(
        yaxis=dict(
            title="Allocation",
            ticksuffix="%",
            range=[0, 100],
        ),
        template="plotly_dark",
        height=450,
        showlegend=True,
        hovermode="x unified",
        margin=dict(l=60, r=20, t=20, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=-0.25, xanchor="center", x=0.5),
    )

    key_suffix = f"_{unique_id}" if unique_id else ""
    st.plotly_chart(fig, use_container_width=True, key=f"port_alloc_area{key_suffix}")


def render_rebalancing_analysis(trades_df, pl_by_year, composition_df, tax_method, other_income, filing_status, state_code, rebalance_freq="Yearly", use_standard_deduction=True, unrealized_pl_df=None, custom_freq="Yearly", unique_id=None, component_prices=None, allocation=None, start_val=10000, retirement_income=None, retirement_year=None, port_series=None, rebal_config=None):
    if trades_df.empty:
        st.info("No rebalancing events found.")
        return
        
    # Determine default index based on rebalance_freq
    freq_options = ["Yearly", "Quarterly", "Monthly", "Per Event"]
    try:
        default_idx = freq_options.index(rebalance_freq)
    except ValueError:
        if rebalance_freq == "Custom":
            # Try to match the custom frequency (Yearly/Quarterly/Monthly)
            if custom_freq in freq_options:
                default_idx = freq_options.index(custom_freq)
            else:
                default_idx = 3 # Default to "Per Event"
        else:
            default_idx = 0
        
    key_suffix = f"_{unique_id}" if unique_id else ""
    
    # View Frequency Selector
    view_freq = st.selectbox(
        "View Frequency", 
        freq_options, 
        index=default_idx,
        key=f"rebal_view_freq{key_suffix}"
    )

    # Optional "Mag 7 Fund" Grouping
    group_mag7 = st.toggle("Enable Mag 7 Grouping", value=False, key=f"rebal_mag7{key_suffix}", help="Groups AAPL, MSFT, GOOG, AMZN, NVDA, META, TSLA, and AVGO into a single 'Mag 7' fund.")
    
    # Process Composition Data for Mag 7 Grouping
    comp_df_to_plot = composition_df.copy()
    if group_mag7 and not comp_df_to_plot.empty:
        # Define sets
        mag7_standard = ["MSFT", "TSLA", "GOOG", "AAPL", "NVDA", "META", "AMZN"]
        mag7_plus_avgo = mag7_standard + ["AVGO"]
        
        # Helper to categorize tickers
        def get_group(ticker):
            # Check for leverage
            is_lev = "?L=2" in ticker
            
            # Clean base ticker for checking (handle ?L=2 and other suffixes if any)
            base = ticker.split("?")[0]
            
            # Check if it starts with any known root (e.g. GOOGL matches GOOG)
            # We must be careful not to match random things, but for these specific tickers likely safe
            
            # 1. QQQU Check: Mag 7 + AVGO (Leveraged)
            # User said: "When you see it [Mag7+AVGO] with L2 consider the tag 'QQQU'"
            if is_lev:
                # Check root match against Mag 7 + AVGO
                match = False
                if base in mag7_plus_avgo: match = True
                else:
                    for r in mag7_plus_avgo:
                        if base.startswith(r): 
                            match = True
                            break
                if match: return "QQQU"

            # 2. Mag 7 Check: Standard Mag 7 (Unleveraged, NO AVGO)
            # User said: "Mag 7 without AVGO ... without ?L=2 consider it labeled 'Mag 7'"
            else:
                # Check root match against Standard Mag 7 ONLY
                match = False
                if base in mag7_standard: match = True
                else: 
                     for r in mag7_standard:
                        if base.startswith(r):
                            match = True
                            break
                if match: return "Mag 7"
            
            # Default: No Group
            return None

        # Apply grouping
        comp_df_to_plot["Group"] = comp_df_to_plot["Ticker"].apply(get_group)
        
        # Split into grouped and ungrouped
        grouped_rows = comp_df_to_plot[comp_df_to_plot["Group"].notna()]
        ungrouped_rows = comp_df_to_plot[comp_df_to_plot["Group"].isna()].drop(columns=["Group"])
        
        if not grouped_rows.empty:
            # Aggregate by Date AND Group
            grouped_agg = grouped_rows.groupby(["Date", "Group"])["Value"].sum().reset_index()
            grouped_agg = grouped_agg.rename(columns={"Group": "Ticker"})
            
            # Combine back
            comp_df_to_plot = pd.concat([ungrouped_rows, grouped_agg], ignore_index=True).sort_values("Date")
        else:
            # Cleanup if nothing matched
             comp_df_to_plot = comp_df_to_plot.drop(columns=["Group"])
    
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
            unrealized_yearly = unrealized_pl_df.resample("YE").last()
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
        
    elif view_freq == "Per Event":
        # Use exact dates of each trade event
        cols_to_agg = ["Realized P&L", "Realized ST P&L", "Realized LT P&L"]
        if "Realized LT (Collectible)" in df_chart.columns:
            cols_to_agg.append("Realized LT (Collectible)")
            
        agg_df = df_chart.groupby("Date")[cols_to_agg].sum().sort_index()
        
        # Merge Unrealized P&L (Match to exact event dates)
        if unrealized_pl_df is not None and not unrealized_pl_df.empty:
            # We align unrealized P&L to the exact rebalance dates
            unrealized_aligned = unrealized_pl_df.reindex(agg_df.index).ffill().fillna(0.0)
            agg_df = agg_df.join(unrealized_aligned[["Unrealized P&L"]], how="outer").fillna(0.0)
            
        x_axis = agg_df.index
        
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
            
            # Add Unrealized traces
            if "Unrealized P&L" in agg_df.columns and agg_df["Unrealized P&L"].abs().sum() > 0:
                 # Cumulative Unrealized P&L (bar)
                 fig.add_trace(go.Bar(
                    x=x_axis,
                    y=agg_df["Unrealized P&L"],
                    name="Unrealized P&L (Deferred)",
                    marker_color="#636EFA",
                    opacity=0.6,
                    hovertemplate="%{y:$,.0f}"
                 ))
                 # Total Gain = Realized + Unrealized Change per period
                 yoy_change = agg_df["Unrealized P&L"].diff()
                 yoy_change.iloc[0] = agg_df["Unrealized P&L"].iloc[0]
                 total_gain = agg_df["Realized P&L"] + yoy_change
                 fig.add_trace(go.Scatter(
                    x=x_axis,
                    y=total_gain,
                    name="Total Gain (Realized + Unrealized Δ)",
                    mode="lines+markers",
                    line=dict(color="#FECB52", width=2.5, dash="dot"),
                    marker=dict(size=7),
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
            xaxis_title={"Yearly": "Year", "Quarterly": "Quarter", "Monthly": "Month", "Per Event": "Event"}.get(view_freq, view_freq),
            template="plotly_dark",
            showlegend=True,
            height=400,
            hovermode="x unified"
        )
        st.plotly_chart(fig, use_container_width=True)
        
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


    # Portfolio Allocation (100% stacked area — drift between rebalances)
    if component_prices is not None and allocation is not None:
        render_portfolio_allocation(
            component_prices,
            allocation,
            composition_df,
            start_val=start_val,
            unique_id=unique_id or "",
            port_series=port_series,
            rebal_config=rebal_config,
        )

    # Portfolio Composition
    render_portfolio_composition(comp_df_to_plot, view_freq=view_freq)

    # Sankey Diagram
    render_rebalance_sankey(trades_df, view_freq=view_freq, unique_id=unique_id)
    
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
            elif view_freq == "Per Event":
                df_details["Period"] = df_details["Date"].dt.strftime('%Y-%m-%d')

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

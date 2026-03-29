# Correlation Overlay Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a rolling average pairwise correlation line and all-assets-declining red shading to the component performance chart.

**Architecture:** All changes inside `render_portfolio_allocation()` in `app/ui/charts/rebalancing.py`. Uses the `modified_returns` DataFrame already computed in the function. No new files or modules.

**Tech Stack:** pandas (rolling correlation), Plotly (second y-axis, vrect shapes), Streamlit (selectbox)

**Spec:** `docs/superpowers/specs/2026-03-28-correlation-overlay-design.md`

---

### Task 1: Add correlation dropdown and overlay to component performance chart

**Files:**
- Modify: `app/ui/charts/rebalancing.py` (inside `render_portfolio_allocation()`, lines ~415–474)

- [ ] **Step 1: Add the correlation window dropdown before the chart**

Find the line (around line 415):
```python
    fig_lines.add_trace(go.Scatter(
        x=positions.index,
        y=row_totals,
        name="Total Portfolio",
```

Insert BEFORE this line (after positions are computed but before traces are added, around line 413):

```python
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
            key=f"corr_window_{unique_id}" if unique_id else "corr_window",
        )
        corr_window = corr_window_options[corr_window_label]

        # Compute rolling pairwise correlations and average them
        ret = modified_returns[available].dropna()
        pairs = [(available[i], available[j]) for i in range(len(available)) for j in range(i+1, len(available))]
        pair_corrs = pd.DataFrame(index=ret.index)
        for a, b in pairs:
            pair_corrs[f"{a}_{b}"] = ret[a].rolling(corr_window, min_periods=corr_window // 2).corr(ret[b])
        avg_corr = pair_corrs.mean(axis=1).reindex(positions.index)

        # Detect all-assets-declining periods (rolling sum of returns < 0 for every asset)
        all_declining_mask = pd.Series(True, index=ret.index)
        for t in available:
            rolling_ret = ret[t].rolling(corr_window, min_periods=corr_window // 2).sum()
            all_declining_mask = all_declining_mask & (rolling_ret < 0)
        all_declining_mask = all_declining_mask.reindex(positions.index, fill_value=False)
```

- [ ] **Step 2: Add red shaded regions for all-declining periods**

After the code from Step 1, add:

```python
        # Build contiguous all-declining date ranges for red shading
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
        # Close final region if still open
        if in_region and region_start is not None:
            fig_lines.add_vrect(
                x0=region_start, x1=positions.index[-1],
                fillcolor="rgba(239, 68, 68, 0.10)",
                line=dict(color="rgba(239, 68, 68, 0.3)", width=1),
                layer="below",
            )
```

Note: This code needs to go AFTER `fig_lines` is created (the `go.Figure()` call) but before `fig_lines.update_layout()`. Looking at the current code, `fig_lines` is created around line 410 with `fig_lines = go.Figure()`. The vrect shapes need `fig_lines` to exist. So the red shading code should be placed after the component traces are added (after the `for t in available:` loop around line 435) but before `update_layout`.

Actually, looking more carefully at the flow: the dropdown must come before the figure is rendered by `st.plotly_chart`, and the correlation calculation + vrect shapes need `fig_lines` to exist. The cleanest placement is:

1. Dropdown + calculation: right after `positions` DataFrame is built (around line 413, before any traces)
2. Traces added to fig_lines as usual (existing code)
3. Red shading vrects + correlation trace: after component traces, before `update_layout`

Revising: place the **dropdown + calculation** from Step 1 right after positions are computed (before `fig_lines = go.Figure()`). Then place the **red shading** and **correlation trace** after the rebalance markers loop (after line 454) and before `update_layout` (line 457).

- [ ] **Step 3: Add the correlation line trace**

After the red shading code (and after the rebalance markers loop), add:

```python
    # Add correlation trace on secondary y-axis
    if show_correlation:
        fig_lines.add_trace(go.Scatter(
            x=avg_corr.index,
            y=avg_corr,
            name="Avg Correlation",
            mode="lines",
            line=dict(width=1.5, color="#a78bfa"),
            yaxis="y2",
            opacity=0.85,
            hovertemplate="Correlation: %{y:.2f}<extra></extra>",
        ))

        # Add legend entry for red shading
        if all_declining_mask.any():
            fig_lines.add_trace(go.Scatter(
                x=[None], y=[None],
                name="All Declining",
                mode="markers",
                marker=dict(size=10, color="rgba(239, 68, 68, 0.5)", symbol="square"),
                showlegend=True,
            ))
```

- [ ] **Step 4: Update the layout with second y-axis**

Modify the existing `fig_lines.update_layout()` call. Change:

```python
    fig_lines.update_layout(
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
        margin=dict(l=80, r=20, t=20, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=-0.25, xanchor="center", x=0.5),
    )
```

to:

```python
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
            range=[-1.05, 1.05],
            showgrid=False,
            tickvals=[-1, -0.5, 0, 0.5, 1],
            titlefont=dict(color="#a78bfa"),
            tickfont=dict(color="#a78bfa"),
            zeroline=True,
            zerolinecolor="rgba(167, 139, 250, 0.3)",
            zerolinewidth=1,
        )
    fig_lines.update_layout(**layout_kwargs)
```

Note: `margin.r` changed from 20 to 80 to make room for the right y-axis labels.

- [ ] **Step 5: Verify the app runs**

Run: `streamlit run testfol_charting.py --server.port 8501`

Open in browser, run a backtest with 2+ assets (e.g. SPYSIM 60% / GLDSIM 20% / ZROZSIM 20%), go to Rebalancing tab. Verify:
- Correlation dropdown appears above the component performance chart
- Purple correlation line on right y-axis (-1 to 1)
- Red shaded regions visible during periods when all assets decline
- Changing the dropdown window updates the chart
- Single-asset portfolios: no dropdown, no correlation line

- [ ] **Step 6: Commit**

```bash
git add app/ui/charts/rebalancing.py
git commit -m "feat: add rolling correlation overlay to component performance chart"
```

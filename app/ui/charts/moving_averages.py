import plotly.graph_objects as go
import streamlit as st
import pandas as pd

from app.core import calculations

def render_ma_analysis_tab(port_series, portfolio_name, unique_id, window=200, show_stage_analysis=True):
    """
    Renders the Moving Average Analysis tab content.
    """
    st.subheader(f"{portfolio_name} {window}-Day Moving Average Analysis")
    
    # Controls
    c_ctrl1, c_ctrl2 = st.columns(2)
    # Ensure keys are unique per window AND per portfolio instance
    key_suffix = f"{unique_id}_{window}" if unique_id else f"{window}"
    
    merge_tol = c_ctrl1.slider(
        "Merge Events Tolerance (Days)", 
        min_value=0, max_value=30, value=14, step=1,
        key=f"ma_merge_{key_suffix}",
        help=f"**Merge Tolerance**: Ignores short recoveries. If the price recovers above {window}MA for fewer than X days before dropping again, it is considered a single continuous 'Under' event. Useful for filtering out fake breakouts."
    )
    min_days = c_ctrl2.slider(
        "Signal Filter (Min Days)", 
        min_value=0, max_value=90, value=14, step=1,
        key=f"ma_min_{key_suffix}",
        help=f"**Signal Filter**: Excludes short-lived drops below the {window}MA (noise). Events shorter than X days will be hidden from the analysis table and statistics."
    )

    # Calculate Stats (Reactive)
    # Use the generalized analyze_ma function
    dma_series, events_df = calculations.analyze_ma(port_series, window=window, tolerance_days=merge_tol)
    
    # Calculate Stage (New)
    stage_series, slope_series, _ = calculations.analyze_stage(port_series, ma_window=window)
    
    if dma_series is None or dma_series.dropna().empty: 
            st.info(f"Insufficient data to calculate {window}MA (need >{window} days).")
            return

    # Just in case events_df is empty but we have DMA
    filtered_events = pd.DataFrame()  # Initialize for chart use
    if events_df.empty:
        st.info(f"Price has never been below {window}MA in this period.")
    else:
        # Apply Min Days Filter for Display/Stats
        filtered_events = events_df[events_df["Duration (Days)"] >= min_days]
    
    # Chart
    fig = go.Figure()
    
    # Base Price (Blue)
    fig.add_trace(go.Scatter(
        x=port_series.index, y=port_series,
        name="Price",
        line=dict(color='#2E86C1', width=2),
        hovertemplate="Price: $%{y:,.0f}<extra></extra>"
    ))
    
    # Price Below MA (Red Overlay)
    price_below = port_series.copy()
    # Mask values where Price >= DMA (keep only Below)
    price_below[port_series >= dma_series] = None 
    
    fig.add_trace(go.Scatter(
        x=price_below.index, y=price_below,
        name=f"Below {window}MA",
        line=dict(color='#FFD700', width=2),
        hovertemplate="Price: $%{y:,.0f}<extra></extra>",
        showlegend=False # Cleaner legend
    ))

    fig.add_trace(go.Scatter(
        x=dma_series.index, y=dma_series,
        name=f"{window}MA",
        line=dict(color='#E74C3C', width=1.5),
        hovertemplate=f"{window}MA: $%{{y:,.0f}}<extra></extra>"
    ))
    
    # Add Peak markers - Show ALL from events_df, grey out non-filtered
    if not events_df.empty and "Peak Date" in events_df.columns:
        # Get filtered peak dates for comparison
        filtered_peak_dates = set(filtered_events["Peak Date"].dropna()) if not filtered_events.empty else set()
        
        # Get all peaks from raw events
        peak_data = events_df[["Peak Date", "Bottom to Peak (%)"]].dropna(subset=["Peak Date"])
        if not peak_data.empty:
            for _, row in peak_data.iterrows():
                d = row["Peak Date"]
                rally = row["Bottom to Peak (%)"]
                if d in port_series.index:
                    is_filtered = d in filtered_peak_dates
                    rally_val = rally if pd.notna(rally) else 0
                    filtered_label = "" if is_filtered else " (filtered)"
                    fig.add_trace(go.Scatter(
                        x=[d], 
                        y=[port_series.loc[d]],
                        mode='markers',
                        name="Peak" if is_filtered else "Peak (Filtered)",
                        legendgroup="peak" if is_filtered else "peak_filtered",
                        showlegend=False,
                        visible=True if is_filtered else 'legendonly',
                        marker=dict(
                            symbol='diamond',
                            size=10 if is_filtered else 7,
                            color='#00CC96' if is_filtered else 'rgba(100, 100, 100, 0.5)',
                            line=dict(width=1, color='white' if is_filtered else 'grey')
                        ),
                        hovertemplate=f"Peak: $%{{y:,.0f}} (+{rally_val:.1f}%)<br>%{{x|%b %d, %Y}}{filtered_label}<extra></extra>"
                    ))
        
        # Add legend entries with legendgroup for toggle
        fig.add_trace(go.Scatter(x=[None], y=[None], mode='markers', name="Peak",
            legendgroup="peak",
            marker=dict(symbol='diamond', size=10, color='#00CC96', line=dict(width=1, color='white'))))
        fig.add_trace(go.Scatter(x=[None], y=[None], mode='markers', name="Peak (Filtered)",
            legendgroup="peak_filtered", visible='legendonly',
            marker=dict(symbol='diamond', size=7, color='rgba(100, 100, 100, 0.5)', line=dict(width=1, color='grey'))))
    
    # Add Bottom markers - Show ALL from events_df, grey out non-filtered
    if not events_df.empty and "Bottom Date" in events_df.columns:
        # Get filtered bottom dates for comparison
        filtered_bottom_dates = set(filtered_events["Bottom Date"].dropna()) if not filtered_events.empty else set()
        
        # Get all bottoms from raw events
        bottom_data = events_df[["Bottom Date", "Max Depth (%)"]].dropna(subset=["Bottom Date"])
        if not bottom_data.empty:
            for _, row in bottom_data.iterrows():
                d = row["Bottom Date"]
                depth = row["Max Depth (%)"]
                if d in port_series.index:
                    is_filtered = d in filtered_bottom_dates
                    depth_val = depth if pd.notna(depth) else 0
                    filtered_label = "" if is_filtered else " (filtered)"
                    fig.add_trace(go.Scatter(
                        x=[d], 
                        y=[port_series.loc[d]],
                        mode='markers',
                        name="Bottom" if is_filtered else "Bottom (Filtered)",
                        legendgroup="bottom" if is_filtered else "bottom_filtered",
                        showlegend=False,
                        visible=True if is_filtered else 'legendonly',
                        marker=dict(
                            symbol='triangle-down',
                            size=10 if is_filtered else 7,
                            color='#EF553B' if is_filtered else 'rgba(100, 100, 100, 0.5)',
                            line=dict(width=1, color='white' if is_filtered else 'grey')
                        ),
                        hovertemplate=f"Bottom: $%{{y:,.0f}} ({depth_val:.1f}%)<br>%{{x|%b %d, %Y}}{filtered_label}<extra></extra>"
                    ))
        
        # Add legend entries with legendgroup for toggle
        fig.add_trace(go.Scatter(x=[None], y=[None], mode='markers', name="Bottom",
            legendgroup="bottom",
            marker=dict(symbol='triangle-down', size=10, color='#EF553B', line=dict(width=1, color='white'))))
        fig.add_trace(go.Scatter(x=[None], y=[None], mode='markers', name="Bottom (Filtered)",
            legendgroup="bottom_filtered", visible='legendonly',
            marker=dict(symbol='triangle-down', size=7, color='rgba(100, 100, 100, 0.5)', line=dict(width=1, color='grey'))))
    fig.update_layout(
        title=f"Price vs {window}MA",
        template="plotly_dark",
        height=500,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        xaxis_title="Date",
        xaxis=dict(range=[port_series.index[0], port_series.index[-1]]),  # Force full date range
        yaxis_title="Price ($)",
        yaxis_type="log",
        hovermode="x unified"
    )
    st.plotly_chart(fig, use_container_width=True, key=f"ma_chart_{key_suffix}")

    if not events_df.empty:
        # Summary Metrics (Based on FILTERED events)
        # Note: If filtered_events is empty, we handle that
        
        total_days = (port_series.index[-1] - port_series.index[0]).days
        if not filtered_events.empty: # Use filtered_events here to respect the variable (was hardcoded)
            days_under = filtered_events["Duration (Days)"].sum()
            pct_under = (days_under / total_days) * 100 if total_days > 0 else 0
            
            longest_event_idx = filtered_events["Duration (Days)"].idxmax()
            longest_event = filtered_events.loc[longest_event_idx]
            
            l_dur = longest_event['Duration (Days)']
            l_depth = longest_event['Max Depth (%)']
            
            # Calculate median Recovery Days (Start to Recovered)
            if "Recovery Days" in filtered_events.columns:
                recovered_events = filtered_events[filtered_events["Recovery Days"].notna()]
                if not recovered_events.empty:
                    median_recovery = recovered_events["Recovery Days"].median()
                else:
                    median_recovery = None
            else:
                median_recovery = None

            # Calculate Median Rally Days (Bottom -> Peak)
            if "Days Bottom to Peak" in filtered_events.columns:
                rally_events = filtered_events[filtered_events["Days Bottom to Peak"].notna()]
                if not rally_events.empty:
                    median_rally_days = rally_events["Days Bottom to Peak"].median()
                else:
                   median_rally_days = None
            else:
                median_rally_days = None

            # Calculate Median Rally % (Bottom -> Peak)
            if "Bottom to Peak (%)" in filtered_events.columns:
                rally_pct_events = filtered_events[filtered_events["Bottom to Peak (%)"].notna()]
                if not rally_pct_events.empty:
                    median_rally_pct = rally_pct_events["Bottom to Peak (%)"].median()
                else:
                    median_rally_pct = None
            else:
                median_rally_pct = None
            
            # Calculate median max depth and range
            median_depth = filtered_events["Max Depth (%)"].median()
            min_depth = filtered_events["Max Depth (%)"].min()  # Most negative = deepest
            max_depth = filtered_events["Max Depth (%)"].max()  # Least negative = shallowest
            total_breaches = len(filtered_events)

            # Calculate median Days to ATH (only for events that reached ATH)
            if "Days to ATH" in filtered_events.columns:
                ath_events = filtered_events[filtered_events["Days to ATH"].notna()]
                if not ath_events.empty:
                    median_days_to_ath = ath_events["Days to ATH"].median()
                else:
                    median_days_to_ath = None
            else:
                median_days_to_ath = None
        else:
            days_under = 0
            pct_under = 0
            l_dur = 0
            l_depth = 0
            median_recovery = None
            median_rally_days = None
            median_rally_pct = None
            median_depth = None
            min_depth = None
            max_depth = None
            total_breaches = 0
            median_days_to_ath = None
        
        # Check current status and calculate depth metrics
        last_price = port_series.iloc[-1]
        last_dma = dma_series.iloc[-1]
        current_depth = None
        current_depth_rank = None
        is_below = False
        status_text = ""
        status_delta = ""

        if events_df.empty:
            if last_price >= last_dma:
                status_text = "🟢 Above"
                status_delta = f"Never below {window}MA"
            else:
                status_text = "🔴 Below"
                status_delta = f"First breach of {window}MA"
                is_below = True
        elif events_df.iloc[-1]["Status"] == "Ongoing":
            last_event = events_df.iloc[-1]
            status_text = "🔴 Below"
            status_delta = f"{last_event['Duration (Days)']}d under {window}MA"
            is_below = True
            # Calculate current depth
            start_date = last_event["Start Date"]
            event_prices = port_series[start_date:]
            if not event_prices.empty:
                start_price_val = event_prices.iloc[0]
                min_price = event_prices.min()
                current_depth = ((min_price - start_price_val) / start_price_val) * 100
                # Calculate depth rank
                if total_breaches > 0 and min_depth is not None:
                    all_depths = filtered_events["Max Depth (%)"].dropna().tolist()
                    sorted_depths = sorted(all_depths)
                    current_depth_rank = 1
                    for d in sorted_depths:
                        if current_depth <= d:
                            break
                        current_depth_rank += 1
        elif last_price < last_dma:
            status_text = "🔴 Below"
            status_delta = f"Just crossed {window}MA"
            is_below = True
        else:
            last_event = events_df.iloc[-1]
            last_end = last_event["End Date"]
            if pd.notna(last_end):
                days_above = (port_series.index[-1] - last_end).days
                status_text = "🟢 Above"
                status_delta = f"{days_above}d over {window}MA"
            else:
                status_text = "🟢 Above"
                status_delta = f"{window}MA"

        # Calculate recovery stats from similar depths (for below MA state)
        recovery_rate_similar = None
        num_similar = 0
        num_recovered = 0
        med_recovery_similar = None
        max_recovery_similar = None

        if is_below and current_depth is not None and total_breaches >= 1:
            similar_or_deeper = filtered_events[filtered_events["Max Depth (%)"] <= current_depth]
            num_similar = len(similar_or_deeper)
            if num_similar > 0:
                recovered = similar_or_deeper[similar_or_deeper["Status"] == "Recovered"]
                num_recovered = len(recovered)
                recovery_rate_similar = (num_recovered / num_similar) * 100
                if num_recovered > 0 and "Recovery Days" in recovered.columns:
                    recovery_days = recovered["Recovery Days"].dropna()
                    if not recovery_days.empty:
                        med_recovery_similar = recovery_days.median()
                        max_recovery_similar = recovery_days.max()

        # Row 1: Current State (4 cols)
        r1c1, r1c2, r1c3, r1c4 = st.columns(4)
        r1c1.metric("Status", status_text, status_delta)

        if current_depth is not None:
            rank_text = f"Rank #{current_depth_rank}/{total_breaches}" if current_depth_rank else ""
            r1c2.metric("Current Depth", f"{current_depth:.1f}%", rank_text)
        elif median_depth is not None:
            depth_range = f"Range: {min_depth:.0f}% to {max_depth:.0f}%" if min_depth is not None else ""
            r1c2.metric("Med. Depth", f"{median_depth:.1f}%", depth_range)

        r1c3.metric("Time Under", f"{pct_under:.1f}%", f"{days_under}d total")
        r1c4.metric("Longest", f"{l_dur:.0f}d", f"Depth: {l_depth:.1f}%")

        # Row 2: Recovery Outlook (4 cols)
        r2c1, r2c2, r2c3, r2c4 = st.columns(4)

        if is_below and recovery_rate_similar is not None:
            # Show recovery outlook for current situation
            r2c1.metric("Recovery Rate", f"{recovery_rate_similar:.0f}%", f"{num_recovered}/{num_similar} similar")
            if med_recovery_similar is not None:
                r2c2.metric("Similar Recovery", f"{med_recovery_similar:.0f}d", f"Max: {max_recovery_similar:.0f}d")
            elif median_recovery is not None:
                r2c2.metric("Med. Recovery", f"{median_recovery:.0f}d", "Start→Even")
            if median_rally_pct is not None and median_rally_days is not None:
                r2c3.metric("Med. Rally", f"{median_rally_pct:.1f}%", f"{median_rally_days:.0f}d Bottom→Peak")
            elif median_rally_pct is not None:
                r2c3.metric("Rally Gain", f"{median_rally_pct:.1f}%", "Bottom→Peak")
            if median_days_to_ath is not None:
                r2c4.metric("To ATH", f"{median_days_to_ath:.0f}d", "Cross→New High")
        else:
            # Show general historical stats
            if median_recovery is not None:
                r2c1.metric("Med. Recovery", f"{median_recovery:.0f}d", "Start→Even")
            if median_rally_pct is not None and median_rally_days is not None:
                r2c2.metric("Med. Rally", f"{median_rally_pct:.1f}%", f"{median_rally_days:.0f}d Bottom→Peak")
            elif median_rally_days is not None:
                r2c2.metric("Med. Rally", f"{median_rally_days:.0f}d", "Bottom→Peak")
            if median_days_to_ath is not None:
                r2c3.metric("To ATH", f"{median_days_to_ath:.0f}d", "Cross→New High")
            r2c4.metric("# Breaches", f"{total_breaches}", "Historical events")

        # Stage Analysis Display
        if show_stage_analysis and stage_series is not None and not stage_series.empty:
            current_stage = stage_series.iloc[-1]
            current_slope = slope_series.iloc[-1]
            
            # Determine color/icon for Stage
            # Stage 1 (Basing) = cautiously bullish (coming out of decline)
            # Stage 3 (Topping) = cautiously bearish (coming off advance)
            if "Stage 2" in current_stage:
                s_color = "🟢"
            elif "Stage 4" in current_stage:
                s_color = "🔴"
            elif "Stage 1" in current_stage and "1/3" not in current_stage:
                s_color = "🟡"  # Basing - neutral but post-decline
            elif "Stage 3" in current_stage:
                s_color = "🟠"  # Topping - warning after advance
            else:
                s_color = "⚪"  # Indeterminate
            
            # Trend Text
            if current_slope > 0.001: trend_txt = "Rising ↗️"
            elif current_slope < -0.001: trend_txt = "Falling ↘️"
            else: trend_txt = "Flat ➡️"
            
            st.markdown("---")
            sc1, sc2 = st.columns(2)
            sc1.metric("Weinstein Stage Est.", f"{s_color} {current_stage}")
            sc2.metric(f"{window}MA Trend", trend_txt, f"Slope: {current_slope:.2%}")

            with st.expander("ℹ️ About Weinstein Market Stages"):
                st.markdown("""
                **Stan Weinstein's 4 Stages** *(from "Secrets for Profiting in Bull and Bear Markets")*:
                
                | Stage | Name | MA Trend | Price vs MA | Implication |
                |-------|------|----------|-------------|-------------|
                | 🟡 **1** | **Basing** | Flat (after falling) | Near/around | Accumulation. Bottoming process. |
                | 🟢 **2** | **Advancing** | Rising | Above | Bull market. Strong uptrend. |
                | 🟠 **3** | **Topping** | Flat (after rising) | Near/around | Distribution. Trend exhaustion. |
                | 🔴 **4** | **Declining** | Falling | Below | Bear market. Strong downtrend. |
                
                **Sub-Phases:**
                -   **Stage 2 (Correction):** Price dips below *rising* MA. Often a buying dip if trend intact.
                -   **Stage 4 (Bear Rally):** Price pops above *falling* MA. Often a "bull trap" or selling opportunity.
                
                **This Implementation:**
                -   Uses **{window}-day MA** (Weinstein used 30-week/~150-day on weekly charts).
                -   **Adaptive threshold**: "Flat" is relative to the asset's recent volatility.
                -   **Stage 1 vs 3**: Distinguished by *prior trend* (1 follows decline, 3 follows advance).
                -   **5-day smoothing**: Reduces daily noise/whipsaw.
                """.replace("{window}", str(window)))
        
        # Events Table
        st.subheader(f"Periods Under {window}MA")
        
        with st.expander("ℹ️ Understanding the Metrics"):
            st.markdown(f"""
**Row 1 - Current State:**
| Metric | Meaning |
|--------|---------|
| **Status** | Current position: 🟢 Above or 🔴 Below the {window}MA, with duration |
| **Current Depth** | *(When below MA)* Current drawdown from breach start, with rank among all historical breaches (1 = deepest) |
| **Med. Depth** | *(When above MA)* Typical (median) drawdown, with historical range |
| **Time Under** | Total % of the period spent below the MA, with total days |
| **Longest** | The single longest breach, with its max depth |

**Row 2 - Recovery Outlook** *(when below MA)*:
| Metric | Meaning |
|--------|---------|
| **Recovery Rate** | % of historical breaches at similar depth that recovered, with count |
| **Similar Recovery** | Median recovery time from breaches of similar or greater depth, with max |
| **Med. Rally** | Typical rally gain from bottom to peak, with duration |
| **To ATH** | Typical days from MA crossover to new all-time high |

**Row 2 - Historical Stats** *(when above MA)*:
| Metric | Meaning |
|--------|---------|
| **Med. Recovery** | Typical days from breach start to breakeven (price back to start) |
| **Med. Rally** | Typical rally gain from bottom to peak, with duration |
| **To ATH** | Typical days from MA crossover to new all-time high |
| **# Breaches** | Total historical breach events |

**Table Columns:**
| Column | Meaning |
|--------|---------|
| **Start / End** | When the price dropped below / recovered above the {window}MA |
| **Days Under MA** | Total calendar days spent below the MA |
| **Max Depth** | Price drawdown from event start to the lowest point (`(Bottom - Start) / Start`) |
| **Breakeven Gain** | % gain from the **bottom** to the **start price** (breakeven). Shows how much the price rallied just to get back to even. |
| **Post-MA Rally** | % gain from **recovery date** (MA crossover) to the subsequent peak |
| **Post-MA Rally Days** | Days from **recovery date** to the subsequent peak (Duration of post-recovery rally) |
| **Price Recovery Days** | Calendar days from **event start** to first date price recovers to **start price** (breakeven). For ongoing events, shows days elapsed so far. |
| **True Recovery** | Calendar days from **event start** until **BOTH** price ≥ start price **AND** price > MA. This is when you're truly recovered—at breakeven and back in an uptrend. |
| **Entry MA** | The MA value when price first dropped below it (your "entry" into the drawdown) |
| **Exit MA** | The MA value when price recovered above it (your "exit" from the drawdown) |
| **MA Δ%** | How much the MA changed during the event: `(Exit MA - Entry MA) / Entry MA`. Negative = MA fell (making it easier to cross back above even if price hasn't fully recovered) |
| **Rally Days** | Calendar days from **lowest price** to **subsequent peak** (Duration of the rally) |
| **Full Rally %** | % gain from the **lowest price** to the subsequent peak (full rebound) |
| **Days to ATH** | Days from **MA crossover** until price makes a **new all-time high** (vs pre-drawdown ATH) |
| **Status** | `Recovered` = crossed back above MA, `Ongoing` = still below (shown with 🟠 highlight) |
| **Pattern** | Recovery shape classification (see below) |

**What is "Peak"?**
> The **subsequent peak** is the highest price reached between the MA recovery date and either:
> - The start of the *next* drawdown event (next time price drops below MA), or
> - The end of the data (if no subsequent drawdown occurred).
>
> This represents the **local high** during the rally—not necessarily a new all-time high. 
> To see when the price made a *new ATH*, check the **Days to ATH** column.

**Recovery Patterns** *(classified using median thresholds from this dataset)*:

| Pattern | Criteria | What It Means |
|---------|----------|---------------|
| ⚡ **V-Shape** | Fast (≤ median days) + Strong (≥ median %) | **Best case.** Sharp selloff met with aggressive buying. Market quickly finds a floor and rockets higher. Often seen after panic selling or capitulation events. |
| 📈 **Grind** | Slow (> median days) + Strong (> median %) | **Patience rewarded.** Base-building recovery that eventually delivers strong returns. Requires holding through volatility but ends well. |
| 🐌 **Choppy** | Slow (> median days) + Weak (< median %) | **Frustrating.** Extended period of sideways action with minimal payoff. May indicate structural weakness or regime change. |
| 📉 **Weak** | Fast (≤ median days) + Weak (< median %) | **Dead cat bounce.** Quick but shallow recovery that doesn't recoup losses. Often followed by more downside. |

*Thresholds are relative to this asset's history—what's "fast" for bonds differs from stocks.*
            """)
        
        display_df = filtered_events.copy() # Use filtered_events here
        if not display_df.empty:
            # Formatting Dates
            display_df["Start"] = display_df["Start Date"].dt.date
            display_df["End"] = display_df["End Date"].apply(lambda x: x.date() if pd.notna(x) else "Ongoing")
            if "Peak Date" in display_df.columns:
                display_df["Peak Date"] = display_df["Peak Date"].apply(lambda x: x.date() if pd.notna(x) else "-")
            
            # Add Recovery Pattern to Status for recovered events
            # Calculate BEFORE renaming columns to access raw fields
            def classify_recovery(row):
                status = row.get("Status", "")
                if "Recovered" not in str(status):
                    return status
                
                days = row.get("Days Bottom to Peak") # Use Rally signal duration
                pct = row.get("Bottom to Peak (%)")
                
                if pd.isna(days) or pd.isna(pct):
                    return status
                
                # Use median thresholds for classification
                # Note: These medians should be calculated on the whole set
                median_days = filtered_events["Days Bottom to Peak"].median()
                median_pct = filtered_events["Bottom to Peak (%)"].median()
                
                short_days = days <= median_days
                high_pct = pct >= median_pct
                
                if short_days and high_pct:
                    pattern = "⚡ V-Shape"
                elif not short_days and high_pct:
                    pattern = "📈 Grind"
                elif not short_days and not high_pct:
                    pattern = "🐌 Choppy"
                else:  # short_days and low_pct
                    pattern = "📉 Weak"
                
                return pattern
            
            display_df["Pattern"] = display_df.apply(classify_recovery, axis=1)

            # Selection & Renaming for cleaner UI
            cols_map = {
                "Start": "Start",
                "End": "End",
                "Duration (Days)": "Days Under MA",
                "Max Depth (%)": "Max Depth",
                "Bottom to Recovery (%)": "Breakeven Gain",  # Actual rally from bottom to breakeven
                "Subsequent Peak (%)": "Post-MA Rally",
                "Post-MA Rally Days": "Post-MA Rally Days",
                "Bottom to Peak (%)": "Full Rally %",
                "Days Bottom to Peak": "Rally Days",
                "Recovery Days": "Price Recovery Days",
                "Days to ATH": "Days to ATH",
                "Status": "Status",
                "Pattern": "Pattern",
                # MA Context for accuracy (MA is a moving target)
                "Entry MA": "Entry MA",
                "Exit MA": "Exit MA",
                "MA Change (%)": "MA Δ%",
                "True Recovery Days": "True Recovery"
            }

            # Ensure columns exist before selecting
            final_cols = [c for c in cols_map.keys() if c in display_df.columns or c in ["Start", "End"]]

            display_df = display_df[final_cols].rename(columns=cols_map)
            display_df = display_df.sort_values("Start", ascending=False)

            # Add emoji indicators to Status for visual highlighting (since column_config doesn't support row styling)
            display_df["Status"] = display_df["Status"].apply(
                lambda x: f"🟡 {x}" if "Ongoing" in str(x) or "Current" in str(x) else x
            )

            # Toggle for MA context columns
            show_ma_context = st.checkbox(
                "Show MA Context",
                value=False,
                help="Show Entry MA, Exit MA, MA Δ%, and True Recovery columns. These help explain why 'Days Under MA' can differ from 'Price Recovery Days' since the MA moves during the drawdown.",
                key=f"ma_context_{unique_id}_{window}"
            )

            # Reorder columns - Main metrics first, then optionally MA context columns
            if show_ma_context:
                final_display_cols = [
                    "Start", "End", "Days Under MA", "Max Depth", "Breakeven Gain",
                    "Price Recovery Days", "True Recovery",  # True Recovery = Price + MA recovered
                    "Entry MA", "Exit MA", "MA Δ%",  # MA Context
                    "Full Rally %", "Rally Days", "Post-MA Rally", "Post-MA Rally Days",
                    "Days to ATH", "Status", "Pattern"
                ]
            else:
                final_display_cols = [
                    "Start", "End", "Days Under MA", "Max Depth", "Breakeven Gain",
                    "Price Recovery Days",
                    "Full Rally %", "Rally Days", "Post-MA Rally", "Post-MA Rally Days",
                    "Days to ATH", "Status", "Pattern"
                ]
            final_display_cols = [c for c in final_display_cols if c in display_df.columns]
            display_df = display_df[final_display_cols]

            # Column tooltips for hover explanations
            column_config = {
                "Start": st.column_config.DateColumn("Start", help="Date when price dropped below the MA", format="YYYY-MM-DD"),
                "End": st.column_config.DateColumn("End", help="Date when price recovered above the MA (or 'Ongoing')", format="YYYY-MM-DD"),
                "Days Under MA": st.column_config.NumberColumn("Days Under MA", help="Total calendar days spent below the MA", format="%.0f"),
                "Max Depth": st.column_config.NumberColumn("Max Depth", help="Price drawdown from event start to the lowest point", format="%.2f%%"),
                "Breakeven Gain": st.column_config.NumberColumn("Breakeven Gain", help="% gain needed from the bottom to get back to start price (breakeven)", format="%.1f%%"),
                "Price Recovery Days": st.column_config.NumberColumn("Price Recovery Days", help="Calendar days from event start to breakeven. For ongoing events, shows days elapsed so far.", format="%.0f"),
                "True Recovery": st.column_config.NumberColumn("True Recovery", help="Days until BOTH price ≥ start price AND price > MA (truly recovered: at breakeven + in uptrend)", format="%.0f"),
                "Entry MA": st.column_config.NumberColumn("Entry MA", help="MA value when price first dropped below it", format="$%.2f"),
                "Exit MA": st.column_config.NumberColumn("Exit MA", help="MA value when price recovered above it", format="$%.2f"),
                "MA Δ%": st.column_config.NumberColumn("MA Δ%", help="How much MA changed during event. Negative = MA fell, making crossover easier even if price hasn't fully recovered.", format="%+.1f%%"),
                "Full Rally %": st.column_config.NumberColumn("Full Rally %", help="% gain from the lowest price to the subsequent peak (or current price for ongoing)", format="%.1f%%"),
                "Rally Days": st.column_config.NumberColumn("Rally Days", help="Calendar days from lowest price to subsequent peak", format="%.0f"),
                "Post-MA Rally": st.column_config.NumberColumn("Post-MA Rally", help="% gain from MA crossover to the subsequent peak", format="%.1f%%"),
                "Post-MA Rally Days": st.column_config.NumberColumn("Post-MA Rally Days", help="Days from MA crossover to the subsequent peak", format="%.0f"),
                "Days to ATH": st.column_config.NumberColumn("Days to ATH", help="Days from MA crossover until price makes a new all-time high", format="%.0f"),
                "Status": st.column_config.TextColumn("Status", help="🟡 Ongoing = still below MA, Recovered = crossed back above MA"),
                "Pattern": st.column_config.TextColumn("Pattern", help="Recovery shape classification based on duration and rally strength"),
            }

            st.dataframe(
                display_df,
                column_config=column_config,
                use_container_width=True,
                hide_index=True
            )

            # Entry Strategy Comparison Table
            st.subheader(f"{portfolio_name} Entry Strategy Comparison (vs SPYSIM)")

            with st.expander("ℹ️ Understanding the Metrics"):
                st.markdown("""
**Strategy:** Buy at the **maximum depth** (lowest point) during each MA breach, sell when price recovers above the MA. Compare returns to buying SPY at the same time.

**Summary Metrics:**
| Metric | Meaning |
|--------|---------|
| **Total Events** | Number of completed (recovered) breach events analyzed |
| **Win Rate** | % of events where buying at max-depth beat buying SPY |
| **Avg Alpha** | Average outperformance vs SPY across all events |
| **Median Alpha** | Typical (median) outperformance vs SPY |

**Table Columns:**
| Column | Meaning |
|--------|---------|
| **Start / End** | When price dropped below / recovered above the MA |
| **Days/Weeks** | Duration of the breach event |
| **Max Depth** | Maximum drawdown from breach start price during this event |
| **Depth Rank** | Depth rank among all breaches (1 = deepest in history) |
| **Return** | Portfolio return from max-depth entry to recovery |
| **SPY Return** | SPYSIM return for the same window (max-depth to recovery) |
| **Alpha** | Outperformance vs SPY (Return - SPY Return). Green = beat SPY, Red = underperformed |
                """)

            comparison_df = calculations.compare_breach_events(
                port_series,
                window=window,
                tolerance_days=merge_tol
            )

            # Also get ongoing event if exists (from the full events_df)
            _, all_events_df = calculations.analyze_ma(port_series, window=window, tolerance_days=merge_tol)
            ongoing_event = None
            if not all_events_df.empty:
                last_event = all_events_df.iloc[-1]
                if last_event["Status"] == "Ongoing":
                    ongoing_event = last_event

            # Summary Statistics Row (only from recovered events)
            if not comparison_df.empty or ongoing_event is not None:
                total_recovered = len(comparison_df) if not comparison_df.empty else 0

                # Extract alpha column with NA handling
                maxdepth_alpha = comparison_df["Max-Depth Entry Alpha (%)"].dropna() if not comparison_df.empty else pd.Series(dtype=float)

                # Calculate statistics
                maxdepth_wins = (maxdepth_alpha > 0).sum() if len(maxdepth_alpha) > 0 else 0
                maxdepth_win_rate = (maxdepth_wins / len(maxdepth_alpha) * 100) if len(maxdepth_alpha) > 0 else 0
                maxdepth_avg = maxdepth_alpha.mean() if len(maxdepth_alpha) > 0 else 0
                maxdepth_median = maxdepth_alpha.median() if len(maxdepth_alpha) > 0 else 0

                # Display metrics row
                m1, m2, m3, m4, m5 = st.columns(5)
                m1.metric("Total Events", total_recovered + (1 if ongoing_event is not None else 0))
                m2.metric("Win Rate", f"{maxdepth_win_rate:.1f}%", help="% of recovered events where max-depth entry beat SPY")
                m3.metric("Avg Alpha", f"{maxdepth_avg:+.1f}%", help="Average outperformance vs SPY (recovered events)")
                m4.metric("Median Alpha", f"{maxdepth_median:+.1f}%", help="Median outperformance vs SPY (recovered events)")
                m5.metric("Events Analyzed", f"{len(maxdepth_alpha)}/{total_recovered}" + (" +1 ongoing" if ongoing_event is not None else ""))

                st.markdown("---")

                # Prepare display DataFrame
                comp_display = comparison_df.copy() if not comparison_df.empty else pd.DataFrame()

                # Add ongoing event to display if exists
                if ongoing_event is not None:
                    ongoing_row = {
                        "Start Date": ongoing_event["Start Date"],
                        "End Date": pd.NaT,
                        "Duration (Days)": ongoing_event["Duration (Days)"],
                        "Max Depth (%)": ongoing_event["Max Depth (%)"],
                        "Max-Depth Entry Return (%)": None,
                        "SPYSIM Max-Depth Return (%)": None,
                        "Max-Depth Entry Alpha (%)": None,
                        "Status": "🟠 Ongoing"
                    }
                    ongoing_df = pd.DataFrame([ongoing_row])
                    comp_display = pd.concat([ongoing_df, comp_display], ignore_index=True)

                if not comp_display.empty:
                    # Convert date columns to .date for cleaner display
                    comp_display["Start"] = comp_display["Start Date"].dt.date
                    comp_display["End"] = comp_display["End Date"].apply(lambda x: x.date() if pd.notna(x) else "Ongoing")
                    comp_display["Duration"] = comp_display["Duration (Days)"]

                    # Calculate Depth Rank across ALL events including ongoing (1 = deepest)
                    if "Max Depth (%)" in comp_display.columns:
                        comp_display["Depth Rank"] = comp_display["Max Depth (%)"].rank(method='min').astype(int)

                    # Select and order columns for display (max-depth entry only)
                    display_cols = [
                        "Start", "End", "Duration", "Max Depth (%)", "Depth Rank",
                        "Max-Depth Entry Return (%)", "SPYSIM Max-Depth Return (%)",
                        "Max-Depth Entry Alpha (%)"
                    ]
                    comp_display = comp_display[[c for c in display_cols if c in comp_display.columns]]

                    # Sort by Start descending (most recent first)
                    comp_display = comp_display.sort_values("Start", ascending=False)

                    # Color styling function for alpha columns
                    def color_alpha(val):
                        if pd.isna(val):
                            return ''
                        color = '#00CC96' if val >= 0 else '#EF553B'
                        return f'color: {color}'

                    # Apply styling to alpha column
                    alpha_cols_present = [c for c in ["Max-Depth Entry Alpha (%)"] if c in comp_display.columns]
                    styled_df = comp_display.style.map(color_alpha, subset=alpha_cols_present)

                    # Column config with tooltips
                    comp_column_config = {
                        "Start": st.column_config.DateColumn("Start", help="Date when price dropped below the MA", format="YYYY-MM-DD"),
                        "End": st.column_config.TextColumn("End", help="Date when price recovered above the MA (or 'Ongoing')"),
                        "Duration": st.column_config.NumberColumn("Days", help="Total calendar days of the breach event", format="%.0f"),
                        "Max Depth (%)": st.column_config.NumberColumn("Max Depth", help="Maximum drawdown from breach start price during this event", format="%.1f%%"),
                        "Depth Rank": st.column_config.NumberColumn("Depth Rank", help="Depth rank among all breaches (1 = deepest)", format="%d"),
                        "Max-Depth Entry Return (%)": st.column_config.NumberColumn("Return", help="Portfolio return: entry at lowest point during breach, exit at recovery", format="%.1f%%"),
                        "SPYSIM Max-Depth Return (%)": st.column_config.NumberColumn("SPY Return", help="SPYSIM return for same max-depth to recovery window", format="%.1f%%"),
                        "Max-Depth Entry Alpha (%)": st.column_config.NumberColumn("Alpha", help="Outperformance vs SPY (positive = beat SPY)", format="%+.1f%%"),
                    }

                    st.dataframe(
                        styled_df,
                        column_config=comp_column_config,
                        use_container_width=True,
                        hide_index=True,
                        key=f"comparison_table_{key_suffix}"
                    )
            else:
                st.info("No breach events to display.")


# -----------------------------------------------------------------------------
# Munger 200 Week Moving Average Analysis
# -----------------------------------------------------------------------------
def render_munger_wma_tab(port_series, portfolio_name, unique_id, window=200):
    """
    Renders the Munger 200 Week Moving Average Analysis tab.
    Charlie Munger advocated for long-term thinking - the 200WMA (~4 years)
    filters out short-term noise and shows secular trends.
    """
    st.subheader(f"{portfolio_name} Munger {window}-Week Moving Average Analysis")

    st.info("""
    💡 **Munger's Wisdom:** *"The big money is not in the buying and selling, but in the waiting."*

    The 200-Week Moving Average (~4 years) filters out short-term noise and reveals secular trends.
    This indicator helps identify generational buying opportunities during major market dislocations.
    """)

    # Controls
    c_ctrl1, c_ctrl2 = st.columns(2)
    key_suffix = f"{unique_id}_wma_{window}" if unique_id else f"wma_{window}"

    merge_tol = c_ctrl1.slider(
        "Merge Events Tolerance (Weeks)",
        min_value=0, max_value=12, value=2, step=1,
        key=f"wma_merge_{key_suffix}",
        help=f"**Merge Tolerance**: Ignores short recoveries. If the price recovers above {window}WMA for fewer than X weeks before dropping again, it is considered a single continuous 'Under' event."
    )
    min_weeks = c_ctrl2.slider(
        "Signal Filter (Min Weeks)",
        min_value=0, max_value=52, value=4, step=1,
        key=f"wma_min_{key_suffix}",
        help=f"**Signal Filter**: Excludes short-lived drops below the {window}WMA (noise). Events shorter than X weeks will be hidden from the analysis."
    )

    # Calculate Weekly MA
    weekly_series, wma_series, events_df = calculations.analyze_wma(port_series, window=window, tolerance_weeks=merge_tol)

    if weekly_series is None or wma_series is None or wma_series.dropna().empty:
        st.warning(f"Insufficient data to calculate {window}WMA. Need at least {window} weeks (~{window/52:.1f} years) of data.")
        return

    filtered_events = pd.DataFrame()
    if events_df.empty:
        st.success(f"🎉 Price has never been below the {window}WMA in this period - Strong secular uptrend!")
    else:
        # Apply Min Weeks Filter
        filtered_events = events_df[events_df["Duration (Weeks)"] >= min_weeks]

    # Chart - Weekly data with WMA overlay
    fig = go.Figure()

    # Weekly Price (Blue)
    fig.add_trace(go.Scatter(
        x=weekly_series.index, y=weekly_series,
        name="Weekly Close",
        line=dict(color='#2E86C1', width=2),
        hovertemplate="Price: $%{y:,.0f}<extra></extra>"
    ))

    # Price Below WMA (Gold Overlay)
    price_below = weekly_series.copy()
    price_below[weekly_series >= wma_series] = None

    fig.add_trace(go.Scatter(
        x=price_below.index, y=price_below,
        name=f"Below {window}WMA",
        line=dict(color='#FFD700', width=2),
        hovertemplate="Price: $%{y:,.0f}<extra></extra>",
        showlegend=False
    ))

    # WMA Line (Red)
    fig.add_trace(go.Scatter(
        x=wma_series.index, y=wma_series,
        name=f"{window}WMA",
        line=dict(color='#E74C3C', width=1.5),
        hovertemplate=f"{window}WMA: $%{{y:,.0f}}<extra></extra>"
    ))

    # Add Peak markers
    if not events_df.empty and "Peak Date" in events_df.columns:
        filtered_peak_dates = set(filtered_events["Peak Date"].dropna()) if not filtered_events.empty else set()
        peak_data = events_df[["Peak Date", "Bottom to Peak (%)"]].dropna(subset=["Peak Date"])
        if not peak_data.empty:
            for _, row in peak_data.iterrows():
                d = row["Peak Date"]
                rally = row["Bottom to Peak (%)"]
                if d in weekly_series.index:
                    is_filtered = d in filtered_peak_dates
                    rally_val = rally if pd.notna(rally) else 0
                    fig.add_trace(go.Scatter(
                        x=[d], y=[weekly_series.loc[d]],
                        mode='markers',
                        name="Peak" if is_filtered else "Peak (Filtered)",
                        legendgroup="peak" if is_filtered else "peak_filtered",
                        showlegend=False,
                        visible=True if is_filtered else 'legendonly',
                        marker=dict(
                            symbol='diamond', size=10 if is_filtered else 7,
                            color='#00CC96' if is_filtered else 'rgba(100, 100, 100, 0.5)',
                            line=dict(width=1, color='white' if is_filtered else 'grey')
                        ),
                        hovertemplate=f"Peak: $%{{y:,.0f}} (+{rally_val:.1f}%)<br>%{{x|%b %d, %Y}}<extra></extra>"
                    ))

    # Add Bottom markers
    if not events_df.empty and "Bottom Date" in events_df.columns:
        filtered_bottom_dates = set(filtered_events["Bottom Date"].dropna()) if not filtered_events.empty else set()
        bottom_data = events_df[["Bottom Date", "Max Depth (%)"]].dropna(subset=["Bottom Date"])
        if not bottom_data.empty:
            for _, row in bottom_data.iterrows():
                d = row["Bottom Date"]
                depth = row["Max Depth (%)"]
                if d in weekly_series.index:
                    is_filtered = d in filtered_bottom_dates
                    depth_val = depth if pd.notna(depth) else 0
                    fig.add_trace(go.Scatter(
                        x=[d], y=[weekly_series.loc[d]],
                        mode='markers',
                        name="Bottom" if is_filtered else "Bottom (Filtered)",
                        legendgroup="bottom" if is_filtered else "bottom_filtered",
                        showlegend=False,
                        visible=True if is_filtered else 'legendonly',
                        marker=dict(
                            symbol='triangle-down', size=10 if is_filtered else 7,
                            color='#EF553B' if is_filtered else 'rgba(100, 100, 100, 0.5)',
                            line=dict(width=1, color='white' if is_filtered else 'grey')
                        ),
                        hovertemplate=f"Bottom: $%{{y:,.0f}} ({depth_val:.1f}%)<br>%{{x|%b %d, %Y}}<extra></extra>"
                    ))

    fig.update_layout(
        title=f"Weekly Price vs {window}WMA (Munger Indicator)",
        template="plotly_dark",
        height=500,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        xaxis_title="Date",
        xaxis=dict(range=[weekly_series.index[0], weekly_series.index[-1]]),
        yaxis_title="Price ($)",
        yaxis_type="log",
        hovermode="x unified"
    )
    st.plotly_chart(fig, use_container_width=True, key=f"wma_chart_{key_suffix}")

    if not events_df.empty:
        # Summary Metrics
        total_weeks = len(weekly_series)
        if not filtered_events.empty:
            weeks_under = filtered_events["Duration (Weeks)"].sum()
            pct_under = (weeks_under / total_weeks) * 100 if total_weeks > 0 else 0

            longest_event_idx = filtered_events["Duration (Weeks)"].idxmax()
            longest_event = filtered_events.loc[longest_event_idx]
            l_dur = longest_event['Duration (Weeks)']
            l_depth = longest_event['Max Depth (%)']

            # Median stats and range
            median_depth = filtered_events["Max Depth (%)"].median()
            min_depth = filtered_events["Max Depth (%)"].min()
            max_depth = filtered_events["Max Depth (%)"].max()
            total_breaches = len(filtered_events)

            if "Recovery Weeks" in filtered_events.columns:
                recovered_events = filtered_events[filtered_events["Recovery Weeks"].notna()]
                median_recovery = recovered_events["Recovery Weeks"].median() if not recovered_events.empty else None
            else:
                median_recovery = None

            if "Weeks Bottom to Peak" in filtered_events.columns:
                rally_events = filtered_events[filtered_events["Weeks Bottom to Peak"].notna()]
                median_rally_weeks = rally_events["Weeks Bottom to Peak"].median() if not rally_events.empty else None
            else:
                median_rally_weeks = None

            if "Bottom to Peak (%)" in filtered_events.columns:
                rally_pct_events = filtered_events[filtered_events["Bottom to Peak (%)"].notna()]
                median_rally_pct = rally_pct_events["Bottom to Peak (%)"].median() if not rally_pct_events.empty else None
            else:
                median_rally_pct = None

            if "Weeks to ATH" in filtered_events.columns:
                ath_events = filtered_events[filtered_events["Weeks to ATH"].notna()]
                median_weeks_to_ath = ath_events["Weeks to ATH"].median() if not ath_events.empty else None
            else:
                median_weeks_to_ath = None
        else:
            weeks_under = 0
            pct_under = 0
            l_dur = 0
            l_depth = 0
            median_recovery = None
            median_rally_weeks = None
            median_rally_pct = None
            median_depth = None
            min_depth = None
            max_depth = None
            total_breaches = 0
            median_weeks_to_ath = None

        # Check current status and calculate depth metrics
        last_price = weekly_series.iloc[-1]
        last_wma = wma_series.iloc[-1]
        current_depth = None
        current_depth_rank = None
        is_below = False
        status_text = ""
        status_delta = ""

        if events_df.empty:
            if last_price >= last_wma:
                status_text = "🟢 Above"
                status_delta = f"Never below {window}WMA"
            else:
                status_text = "🔴 Below"
                status_delta = f"First breach of {window}WMA"
                is_below = True
        elif events_df.iloc[-1]["Status"] == "Ongoing":
            last_event = events_df.iloc[-1]
            status_text = "🔴 Below"
            status_delta = f"{last_event['Duration (Weeks)']}w under {window}WMA"
            is_below = True
            # Calculate current depth
            start_date = last_event["Start Date"]
            event_prices = weekly_series[start_date:]
            if not event_prices.empty:
                start_price_val = event_prices.iloc[0]
                min_price = event_prices.min()
                current_depth = ((min_price - start_price_val) / start_price_val) * 100
                # Calculate depth rank
                if total_breaches > 0 and min_depth is not None:
                    all_depths = filtered_events["Max Depth (%)"].dropna().tolist()
                    sorted_depths = sorted(all_depths)
                    current_depth_rank = 1
                    for d in sorted_depths:
                        if current_depth <= d:
                            break
                        current_depth_rank += 1
        elif last_price < last_wma:
            status_text = "🔴 Below"
            status_delta = f"Just crossed {window}WMA"
            is_below = True
        else:
            last_event = events_df.iloc[-1]
            last_end = last_event["End Date"]
            if pd.notna(last_end):
                weeks_above = len(weekly_series[last_end:]) - 1
                status_text = "🟢 Above"
                status_delta = f"{weeks_above}w over {window}WMA"
            else:
                status_text = "🟢 Above"
                status_delta = f"{window}WMA"

        # Calculate recovery stats from similar depths (for below WMA state)
        recovery_rate_similar = None
        num_similar = 0
        num_recovered = 0
        med_recovery_similar = None
        max_recovery_similar = None

        if is_below and current_depth is not None and total_breaches >= 1:
            similar_or_deeper = filtered_events[filtered_events["Max Depth (%)"] <= current_depth]
            num_similar = len(similar_or_deeper)
            if num_similar > 0:
                recovered = similar_or_deeper[similar_or_deeper["Status"] == "Recovered"]
                num_recovered = len(recovered)
                recovery_rate_similar = (num_recovered / num_similar) * 100
                if num_recovered > 0 and "Recovery Weeks" in recovered.columns:
                    recovery_weeks_data = recovered["Recovery Weeks"].dropna()
                    if not recovery_weeks_data.empty:
                        med_recovery_similar = recovery_weeks_data.median()
                        max_recovery_similar = recovery_weeks_data.max()

        # Row 1: Current State (4 cols)
        r1c1, r1c2, r1c3, r1c4 = st.columns(4)
        r1c1.metric("Status", status_text, status_delta)

        if current_depth is not None:
            rank_text = f"Rank #{current_depth_rank}/{total_breaches}" if current_depth_rank else ""
            r1c2.metric("Current Depth", f"{current_depth:.1f}%", rank_text)
        elif median_depth is not None:
            depth_range = f"Range: {min_depth:.0f}% to {max_depth:.0f}%" if min_depth is not None else ""
            r1c2.metric("Med. Depth", f"{median_depth:.1f}%", depth_range)

        r1c3.metric("Time Under", f"{pct_under:.1f}%", f"{weeks_under}w total")
        r1c4.metric("Longest", f"{l_dur:.0f}w", f"Depth: {l_depth:.1f}%")

        # Row 2: Recovery Outlook (4 cols)
        r2c1, r2c2, r2c3, r2c4 = st.columns(4)

        if is_below and recovery_rate_similar is not None:
            # Show recovery outlook for current situation
            r2c1.metric("Recovery Rate", f"{recovery_rate_similar:.0f}%", f"{num_recovered}/{num_similar} similar")
            if med_recovery_similar is not None:
                r2c2.metric("Similar Recovery", f"{med_recovery_similar:.0f}w", f"Max: {max_recovery_similar:.0f}w")
            elif median_recovery is not None:
                r2c2.metric("Med. Recovery", f"{median_recovery:.0f}w", "Start→Even")
            if median_rally_pct is not None and median_rally_weeks is not None:
                r2c3.metric("Med. Rally", f"{median_rally_pct:.1f}%", f"{median_rally_weeks:.0f}w Bottom→Peak")
            elif median_rally_pct is not None:
                r2c3.metric("Rally Gain", f"{median_rally_pct:.1f}%", "Bottom→Peak")
            if median_weeks_to_ath is not None:
                r2c4.metric("To ATH", f"{median_weeks_to_ath:.0f}w", "Cross→New High")
        else:
            # Show general historical stats
            if median_recovery is not None:
                r2c1.metric("Med. Recovery", f"{median_recovery:.0f}w", "Start→Even")
            if median_rally_pct is not None and median_rally_weeks is not None:
                r2c2.metric("Med. Rally", f"{median_rally_pct:.1f}%", f"{median_rally_weeks:.0f}w Bottom→Peak")
            elif median_rally_weeks is not None:
                r2c2.metric("Med. Rally", f"{median_rally_weeks:.0f}w", "Bottom→Peak")
            if median_weeks_to_ath is not None:
                r2c3.metric("To ATH", f"{median_weeks_to_ath:.0f}w", "Cross→New High")
            r2c4.metric("# Breaches", f"{total_breaches}", "Historical events")

        # Events Table
        st.subheader(f"Periods Under {window}WMA")

        with st.expander("ℹ️ Understanding Munger 200WMA Metrics"):
            st.markdown(f"""
**Why 200 Weeks (~4 Years)?**

Charlie Munger and Warren Buffett emphasize patience and long-term thinking. The 200-Week Moving Average:
- Filters out noise from business cycles and market corrections
- Reveals true secular trends
- Historically, drops below the 200WMA represent major bear markets or generational buying opportunities

**Historical Context:**
- S&P 500 has dropped below its 200WMA only a handful of times in the last 50 years
- Major instances: 1974, 2002-2003, 2008-2009, 2020 (briefly), 2022
- These often marked exceptional long-term entry points

**Row 1 - Current State:**
| Metric | Meaning |
|--------|---------|
| **Status** | Current position: 🟢 Above or 🔴 Below the {window}WMA, with duration |
| **Current Depth** | *(When below)* Current drawdown with rank (1 = deepest in history) |
| **Med. Depth** | *(When above)* Typical drawdown with historical range |
| **Time Under** | % of period spent below WMA, with total weeks |
| **Longest** | Longest breach duration with its max depth |

**Row 2 - Recovery Outlook** *(when below WMA)*:
| Metric | Meaning |
|--------|---------|
| **Recovery Rate** | % of similar-depth breaches that recovered |
| **Similar Recovery** | Median recovery time from similar depths, with max |
| **Med. Rally** | Typical rally gain (bottom to peak) |
| **To ATH** | Typical weeks to new all-time high |

**Table Columns:**
| Column | Meaning |
|--------|---------|
| **Start / End** | When price dropped below / recovered above the {window}WMA |
| **Weeks Under WMA** | Duration in weeks |
| **Max Depth** | Maximum drawdown from event start |
| **Breakeven Gain** | Rally needed from bottom to recover start price |
| **Recovery Weeks** | Weeks from start to breakeven |
| **Full Rally %** | Gain from bottom to subsequent peak |
| **Rally Weeks** | Weeks from bottom to peak |
| **Weeks to ATH** | Weeks from WMA crossover to new all-time high |
            """)

        display_df = filtered_events.copy()
        if not display_df.empty:
            display_df["Start"] = display_df["Start Date"].dt.date
            display_df["End"] = display_df["End Date"].apply(lambda x: x.date() if pd.notna(x) else "Ongoing")
            if "Peak Date" in display_df.columns:
                display_df["Peak Date"] = display_df["Peak Date"].apply(lambda x: x.date() if pd.notna(x) else "-")

            # Recovery Pattern Classification
            def classify_recovery(row):
                status = row.get("Status", "")
                if "Recovered" not in str(status):
                    return status

                weeks = row.get("Weeks Bottom to Peak")
                pct = row.get("Bottom to Peak (%)")

                if pd.isna(weeks) or pd.isna(pct):
                    return status

                median_weeks = filtered_events["Weeks Bottom to Peak"].median()
                median_pct = filtered_events["Bottom to Peak (%)"].median()

                short_weeks = weeks <= median_weeks
                high_pct = pct >= median_pct

                if short_weeks and high_pct:
                    return "⚡ V-Shape"
                elif not short_weeks and high_pct:
                    return "📈 Grind"
                elif not short_weeks and not high_pct:
                    return "🐌 Choppy"
                else:
                    return "📉 Weak"

            display_df["Pattern"] = display_df.apply(classify_recovery, axis=1)

            cols_map = {
                "Start": "Start",
                "End": "End",
                "Duration (Weeks)": "Weeks Under WMA",
                "Duration (Years)": "Years",
                "Max Depth (%)": "Max Depth",
                "Bottom to Recovery (%)": "Breakeven Gain",
                "Subsequent Peak (%)": "Post-WMA Rally",
                "Post-WMA Rally Weeks": "Post-WMA Rally Wks",
                "Bottom to Peak (%)": "Full Rally %",
                "Weeks Bottom to Peak": "Rally Weeks",
                "Recovery Weeks": "Recovery Weeks",
                "Weeks to ATH": "Weeks to ATH",
                "Status": "Status",
                "Pattern": "Pattern",
                "Entry WMA": "Entry WMA",
                "Exit WMA": "Exit WMA",
                "WMA Change (%)": "WMA Δ%",
                "True Recovery Weeks": "True Recovery"
            }

            final_cols = [c for c in cols_map.keys() if c in display_df.columns or c in ["Start", "End"]]
            display_df = display_df[final_cols].rename(columns=cols_map)
            display_df = display_df.sort_values("Start", ascending=False)

            display_df["Status"] = display_df["Status"].apply(
                lambda x: f"🟡 {x}" if "Ongoing" in str(x) or "Current" in str(x) else x
            )

            # Toggle WMA context
            show_wma_context = st.checkbox(
                "Show WMA Context",
                value=False,
                help="Show Entry WMA, Exit WMA, WMA Δ%, and True Recovery columns.",
                key=f"wma_context_{unique_id}_{window}"
            )

            if show_wma_context:
                final_display_cols = [
                    "Start", "End", "Weeks Under WMA", "Years", "Max Depth", "Breakeven Gain",
                    "Recovery Weeks", "True Recovery",
                    "Entry WMA", "Exit WMA", "WMA Δ%",
                    "Full Rally %", "Rally Weeks", "Post-WMA Rally", "Post-WMA Rally Wks",
                    "Weeks to ATH", "Status", "Pattern"
                ]
            else:
                final_display_cols = [
                    "Start", "End", "Weeks Under WMA", "Years", "Max Depth", "Breakeven Gain",
                    "Recovery Weeks",
                    "Full Rally %", "Rally Weeks", "Post-WMA Rally", "Post-WMA Rally Wks",
                    "Weeks to ATH", "Status", "Pattern"
                ]
            final_display_cols = [c for c in final_display_cols if c in display_df.columns]
            display_df = display_df[final_display_cols]

            column_config = {
                "Start": st.column_config.DateColumn("Start", help="Date when price dropped below the WMA", format="YYYY-MM-DD"),
                "End": st.column_config.DateColumn("End", help="Date when price recovered above the WMA", format="YYYY-MM-DD"),
                "Weeks Under WMA": st.column_config.NumberColumn("Weeks Under WMA", help="Total weeks spent below the WMA", format="%.0f"),
                "Years": st.column_config.NumberColumn("Years", help="Duration in years", format="%.1f"),
                "Max Depth": st.column_config.NumberColumn("Max Depth", help="Price drawdown from event start to lowest point", format="%.1f%%"),
                "Breakeven Gain": st.column_config.NumberColumn("Breakeven Gain", help="% gain needed from bottom to recover start price", format="%.1f%%"),
                "Recovery Weeks": st.column_config.NumberColumn("Recovery Weeks", help="Weeks from event start to breakeven", format="%.0f"),
                "True Recovery": st.column_config.NumberColumn("True Recovery", help="Weeks until price ≥ start AND > WMA", format="%.0f"),
                "Entry WMA": st.column_config.NumberColumn("Entry WMA", help="WMA value when price dropped below", format="$%.2f"),
                "Exit WMA": st.column_config.NumberColumn("Exit WMA", help="WMA value when price recovered", format="$%.2f"),
                "WMA Δ%": st.column_config.NumberColumn("WMA Δ%", help="WMA change during event", format="%+.1f%%"),
                "Full Rally %": st.column_config.NumberColumn("Full Rally %", help="Gain from bottom to peak", format="%.1f%%"),
                "Rally Weeks": st.column_config.NumberColumn("Rally Weeks", help="Weeks from bottom to peak", format="%.0f"),
                "Post-WMA Rally": st.column_config.NumberColumn("Post-WMA Rally", help="Gain from WMA crossover to peak", format="%.1f%%"),
                "Post-WMA Rally Wks": st.column_config.NumberColumn("Post-WMA Rally Wks", help="Weeks from WMA crossover to peak", format="%.0f"),
                "Weeks to ATH": st.column_config.NumberColumn("Weeks to ATH", help="Weeks from WMA crossover to new ATH", format="%.0f"),
                "Status": st.column_config.TextColumn("Status", help="Recovery status"),
                "Pattern": st.column_config.TextColumn("Pattern", help="Recovery shape classification"),
            }

            st.dataframe(
                display_df,
                column_config=column_config,
                use_container_width=True,
                hide_index=True
            )

            # Entry Strategy Comparison Table
            st.subheader(f"{portfolio_name} Entry Strategy Comparison (vs SPYSIM)")

            with st.expander("ℹ️ Understanding the Metrics"):
                st.markdown("""
**Strategy:** Buy at the **maximum depth** (lowest point) during each WMA breach, sell when price recovers above the WMA. Compare returns to buying SPY at the same time.

**Summary Metrics:**
| Metric | Meaning |
|--------|---------|
| **Total Events** | Number of completed (recovered) breach events analyzed |
| **Win Rate** | % of events where buying at max-depth beat buying SPY |
| **Avg Alpha** | Average outperformance vs SPY across all events |
| **Median Alpha** | Typical (median) outperformance vs SPY |

**Table Columns:**
| Column | Meaning |
|--------|---------|
| **Start / End** | When price dropped below / recovered above the WMA |
| **Weeks** | Duration of the breach event in weeks |
| **Max Depth** | Maximum drawdown from breach start price during this event |
| **Depth Rank** | Depth rank among all breaches (1 = deepest in history) |
| **Return** | Portfolio return from max-depth entry to recovery |
| **SPY Return** | SPYSIM return for the same window (max-depth to recovery) |
| **Alpha** | Outperformance vs SPY (Return - SPY Return). Green = beat SPY, Red = underperformed |
                """)

            comparison_df = calculations.compare_wma_breach_events(
                port_series,
                window=window,
                tolerance_weeks=merge_tol
            )

            # Also get ongoing event if exists (from the full events_df)
            _, _, all_events_df = calculations.analyze_wma(port_series, window=window, tolerance_weeks=merge_tol)
            ongoing_event = None
            if not all_events_df.empty:
                last_event = all_events_df.iloc[-1]
                if last_event["Status"] == "Ongoing":
                    ongoing_event = last_event

            # Summary Statistics Row (only from recovered events)
            if not comparison_df.empty or ongoing_event is not None:
                total_recovered = len(comparison_df) if not comparison_df.empty else 0

                # Extract alpha column with NA handling
                maxdepth_alpha = comparison_df["Max-Depth Entry Alpha (%)"].dropna() if not comparison_df.empty else pd.Series(dtype=float)

                # Calculate statistics
                maxdepth_wins = (maxdepth_alpha > 0).sum() if len(maxdepth_alpha) > 0 else 0
                maxdepth_win_rate = (maxdepth_wins / len(maxdepth_alpha) * 100) if len(maxdepth_alpha) > 0 else 0
                maxdepth_avg = maxdepth_alpha.mean() if len(maxdepth_alpha) > 0 else 0
                maxdepth_median = maxdepth_alpha.median() if len(maxdepth_alpha) > 0 else 0

                # Display metrics row
                m1, m2, m3, m4, m5 = st.columns(5)
                m1.metric("Total Events", total_recovered + (1 if ongoing_event is not None else 0))
                m2.metric("Win Rate", f"{maxdepth_win_rate:.1f}%", help="% of recovered events where max-depth entry beat SPY")
                m3.metric("Avg Alpha", f"{maxdepth_avg:+.1f}%", help="Average outperformance vs SPY (recovered events)")
                m4.metric("Median Alpha", f"{maxdepth_median:+.1f}%", help="Median outperformance vs SPY (recovered events)")
                m5.metric("Events Analyzed", f"{len(maxdepth_alpha)}/{total_recovered}" + (" +1 ongoing" if ongoing_event is not None else ""))

                st.markdown("---")

                # Prepare display DataFrame
                comp_display = comparison_df.copy() if not comparison_df.empty else pd.DataFrame()

                # Add ongoing event to display if exists
                if ongoing_event is not None:
                    ongoing_row = {
                        "Start Date": ongoing_event["Start Date"],
                        "End Date": pd.NaT,
                        "Duration (Weeks)": ongoing_event["Duration (Weeks)"],
                        "Max Depth (%)": ongoing_event["Max Depth (%)"],
                        "Max-Depth Entry Return (%)": None,
                        "SPYSIM Max-Depth Return (%)": None,
                        "Max-Depth Entry Alpha (%)": None,
                        "Status": "🟠 Ongoing"
                    }
                    ongoing_df = pd.DataFrame([ongoing_row])
                    comp_display = pd.concat([ongoing_df, comp_display], ignore_index=True)

                if not comp_display.empty:
                    # Convert date columns to .date for cleaner display
                    comp_display["Start"] = comp_display["Start Date"].dt.date
                    comp_display["End"] = comp_display["End Date"].apply(lambda x: x.date() if pd.notna(x) else "Ongoing")
                    comp_display["Duration"] = comp_display["Duration (Weeks)"]

                    # Calculate Depth Rank across ALL events including ongoing (1 = deepest)
                    if "Max Depth (%)" in comp_display.columns:
                        comp_display["Depth Rank"] = comp_display["Max Depth (%)"].rank(method='min').astype(int)

                    display_cols = [
                        "Start", "End", "Duration", "Max Depth (%)", "Depth Rank",
                        "Max-Depth Entry Return (%)", "SPYSIM Max-Depth Return (%)",
                        "Max-Depth Entry Alpha (%)"
                    ]
                    comp_display = comp_display[[c for c in display_cols if c in comp_display.columns]]
                    comp_display = comp_display.sort_values("Start", ascending=False)

                    def color_alpha(val):
                        if pd.isna(val):
                            return ''
                        color = '#00CC96' if val >= 0 else '#EF553B'
                        return f'color: {color}'

                    alpha_cols_present = [c for c in ["Max-Depth Entry Alpha (%)"] if c in comp_display.columns]
                    styled_df = comp_display.style.map(color_alpha, subset=alpha_cols_present)

                    comp_column_config = {
                        "Start": st.column_config.DateColumn("Start", help="Date when price dropped below the WMA", format="YYYY-MM-DD"),
                        "End": st.column_config.TextColumn("End", help="Date when price recovered above the WMA (or 'Ongoing')"),
                        "Duration": st.column_config.NumberColumn("Weeks", help="Total weeks of the breach event", format="%.0f"),
                        "Max Depth (%)": st.column_config.NumberColumn("Max Depth", help="Maximum drawdown from breach start price during this event", format="%.1f%%"),
                        "Depth Rank": st.column_config.NumberColumn("Depth Rank", help="Depth rank among all breaches (1 = deepest)", format="%d"),
                        "Max-Depth Entry Return (%)": st.column_config.NumberColumn("Return", help="Portfolio return: entry at lowest point during breach, exit at recovery", format="%.1f%%"),
                        "SPYSIM Max-Depth Return (%)": st.column_config.NumberColumn("SPY Return", help="SPYSIM return for same max-depth to recovery window", format="%.1f%%"),
                        "Max-Depth Entry Alpha (%)": st.column_config.NumberColumn("Alpha", help="Outperformance vs SPY (positive = beat SPY)", format="%+.1f%%"),
                    }

                    st.dataframe(
                        styled_df,
                        column_config=comp_column_config,
                        use_container_width=True,
                        hide_index=True,
                        key=f"wma_comparison_table_{key_suffix}"
                    )
            else:
                st.info("No breach events to display.")


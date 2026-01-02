def render_ma_analysis_tab(port_series, portfolio_name, unique_id, window=200):
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
        min_value=0, max_value=30, value=3, step=1,
        key=f"ma_merge_{key_suffix}",
        help=f"**Merge Tolerance**: Ignores short recoveries. If the price recovers above {window}MA for fewer than X days before dropping again, it is considered a single continuous 'Under' event. Useful for filtering out fake breakouts."
    )
    min_days = c_ctrl2.slider(
        "Signal Filter (Min Days)", 
        min_value=0, max_value=90, value=0, step=1,
        key=f"ma_min_{key_suffix}",
        help=f"**Signal Filter**: Excludes short-lived drops below the {window}MA (noise). Events shorter than X days will be hidden from the analysis table and statistics."
    )

    # Calculate Stats (Reactive)
    # Use the generalized analyze_ma function
    dma_series, events_df = calculations.analyze_ma(port_series, window=window, tolerance_days=merge_tol)
    
    if dma_series is None or dma_series.dropna().empty: 
            st.info(f"Insufficient data to calculate {window}MA (need >{window} days).")
            return

    # Just in case events_df is empty but we have DMA
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
    fig.update_layout(
        title=f"Price vs {window}MA",
        template="plotly_dark",
        height=400,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        xaxis_title="Date",
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
        else:
            days_under = 0
            pct_under = 0
            l_dur = 0
            l_depth = 0
        
        c1, c2, c3 = st.columns(3)
        c1.metric(f"Time Under {window}MA", f"{pct_under:.1f}%", f"{days_under} days total")
        c2.metric("Longest Period Under", f"{l_dur} days", f"Max Depth: {l_depth:.2f}%")
        
        # Check current status
        last_event = events_df.iloc[-1]
        last_price = port_series.iloc[-1]
        last_dma = dma_series.iloc[-1]
        
        if last_event["Status"] == "Ongoing":
                c3.metric("Current Status", f"🔴 Below/Risk {window}MA", f"Duration: {last_event['Duration (Days)']} days")
        elif last_price < last_dma:
                c3.metric("Current Status", f"🔴 Below {window}MA", "Just started")
        else:
            # We are above and outside tolerance
            # Find last end
            last_end = last_event["End Date"]
            if pd.notna(last_end):
                days_above = (port_series.index[-1] - last_end).days
                c3.metric("Current Status", f"🟢 Above {window}MA", f"For {days_above} days")
            else:
                c3.metric("Current Status", f"🟢 Above {window}MA")
        
        # Events Table
        st.subheader(f"Periods Under {window}MA")
        
        display_df = filtered_events.copy() # Use filtered_events here
        if not display_df.empty:
            # Formatting Dates
            display_df["Start"] = display_df["Start Date"].dt.date
            display_df["End"] = display_df["End Date"].apply(lambda x: x.date() if pd.notna(x) else "Ongoing")
            if "Peak Date" in display_df.columns:
                display_df["Peak Date"] = display_df["Peak Date"].apply(lambda x: x.date() if pd.notna(x) else "-")
            
            # Selection & Renaming for cleaner UI
            cols_map = {
                "Start": "Start",
                "End": "End",
                "Duration (Days)": "Days Under", 
                "Max Depth (%)": "Max Depth",
                "Subsequent Peak (%)": "Next Peak",
                "Days Bottom to Peak": "Bottom->Peak Days",
                "Peak Date": "Peak Date",
                "Status": "Status"
            }
            
            # Ensure columns exist before selecting
            final_cols = [c for c in cols_map.keys() if c in display_df.columns or c in ["Start", "End"]]
            
            display_df = display_df[final_cols].rename(columns=cols_map)
            display_df = display_df.sort_values("Start", ascending=False)
            
            st.dataframe(
                display_df.style.format({
                    "Max Depth": "{:.2f}%",
                    "Next Peak": "{:.2f}%",
                    "Days Under": "{:.0f}",
                    "Bottom->Peak Days": "{:.0f}",
                }, na_rep="-"), 
                use_container_width=True,
                hide_index=True
            )

            # Probability Histogram
            st.subheader(f"Distribution of Time Under {window}MA")
            
            hist_fig = go.Figure()
            
            # Histogram
            hist_fig.add_trace(go.Histogram(
                x=filtered_events["Duration (Days)"], # Use filtered_events here
                marker_color='#636EFA',
                marker_line_color='black',
                marker_line_width=1,
                opacity=0.8,
                nbinsx=50
            ))
            
            hist_fig.update_layout(
                template="plotly_dark",
                height=400,
                title=dict(text=f"Distribution of Time Under {window}MA", font=dict(size=14)),
                xaxis_title="Duration (Days)",
                yaxis_title="Frequency",
                showlegend=False,
                bargap=0.05,
                hovermode="x unified"
            )
            
            st.plotly_chart(hist_fig, use_container_width=True, key=f"ma_hist_{key_suffix}")

        else:
            st.info(f"No periods under {window}MA found longer than {min_days} days.")

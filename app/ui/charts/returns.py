import plotly.graph_objects as go
import streamlit as st
import pandas as pd
import numpy as np
from plotly.subplots import make_subplots

from app.common.utils import color_return
from app.core import calculations

def render_cheat_sheet(port_series, portfolio_name, unique_id, component_data=None):
    # Determine target series and name
    target_series = port_series
    target_name = portfolio_name
    
    if component_data is not None and not component_data.empty:
        if isinstance(component_data, pd.DataFrame):
            cols = list(component_data.columns)
            if len(cols) == 1:
                    target_name = cols[0]
                    target_series = component_data[target_name]
            elif len(cols) > 1:
                    target_name = st.selectbox("Select Asset to Analyze", cols, key=f"cs_sel_{unique_id}")
                    target_series = component_data[target_name]
                    
    # Fetch OHLC for Pivot Points
    ohlc_data = None
    try:
            from app.core.shadow_backtest import parse_ticker
            import yfinance as yf
            
            mapped_ticker, _ = parse_ticker(target_name)
            if not target_series.empty:
                last_dt = target_series.index[-1]
                start_f = (last_dt - pd.Timedelta(days=7)).strftime('%Y-%m-%d')
                end_f = (last_dt + pd.Timedelta(days=3)).strftime('%Y-%m-%d')
                
                base_yf = yf.Ticker(mapped_ticker)
                hist_ohlc = base_yf.history(start=start_f, end=end_f, auto_adjust=True, timeout=10)
                
                if hist_ohlc.index.tz is not None:
                    hist_ohlc.index = hist_ohlc.index.tz_localize(None)

                if not hist_ohlc.empty:
                    if last_dt in hist_ohlc.index:
                        ref_bar = hist_ohlc.loc[last_dt]
                    else:
                        ref_bar = hist_ohlc.iloc[-1]
                    
                    ohlc_data = {
                        'High': ref_bar['High'], 
                        'Low': ref_bar['Low'], 
                        'Close': ref_bar['Close']
                    }
    except Exception:
            pass
            
    st.subheader(f"{target_name} Trader's Cheat Sheet")
    cs_df = calculations.calculate_cheat_sheet(target_series, ohlc_data=ohlc_data)
    
    if cs_df is None or cs_df.empty:
            st.info("Insufficient data for Technical Analysis.")
    else:
            # Transform to 3-Column Layout (Barchart Style)
            # Left Column: High/Low, StdDev, Pivot S/R, Session Levels
            # Right Column: Moving Averages, Fibonacci, Pivot Point (P)
            
            left_types = ["High/Low", "StdDev", "Pivot Support", "Pivot Resistance", "Session Level"]
            
            display_rows = []
            for _, row in cs_df.iterrows():
                lbl = row['Label']
                typ = row['Type']
                px = row['Price']
                
                # Format Price as String immediately to allow centering (Numeric defaults to Right)
                px_str = "{:,.2f}".format(px)
                
                if typ == "Current":
                    display_rows.append({"Support/Resistance Levels": "Latest", "Price": px_str, "Key Turning Points": "Latest", "Type": "Current"})
                elif typ in left_types:
                    display_rows.append({"Support/Resistance Levels": lbl, "Price": px_str, "Key Turning Points": "", "Type": "Left"})
                else:
                    display_rows.append({"Support/Resistance Levels": "", "Price": px_str, "Key Turning Points": lbl, "Type": "Right"})
        
            disp_df = pd.DataFrame(display_rows)
            
            # Identify Current Price Row Index for coloring
            try:
                curr_idx = disp_df[disp_df["Type"] == "Current"].index[0]
            except IndexError:
                curr_idx = -1

            # Dark Mode Toggle
            is_dark = st.toggle("Dark Mode Colors", value=True, key=f"cs_dark_{unique_id}")
            
            if is_dark:
                # Dark Mode Palette
                c_res_bg = "#4a1c1c" # Dark Red BG
                c_res_txt = "#ffcdd2" # Light Red Text
                c_sup_bg = "#1b3e20" # Dark Green BG
                c_sup_txt = "#c8e6c9" # Light Green Text
                c_neutral_bg = "#262730" # Streamlit Secondary Dark BG
                c_neutral_txt = "#fafafa" # White Text
                c_current_bg = "#FFD700"
                c_current_txt = "black"
            else:
                # Light Mode Palette (Barchart)
                c_res_bg = "#FFEBEE"
                c_res_txt = "#B71C1C"
                c_sup_bg = "#E8F5E9"
                c_sup_txt = "#1B5E20"
                c_neutral_bg = "white"
                c_neutral_txt = "#333333"
                c_current_bg = "#FFD700"
                c_current_txt = "black"

            def style_barchart(row):
                idx = row.name
                styles = []
                
                # Determine Active Colors based on Row Position
                if idx < curr_idx: # Resistance
                    bg_active = c_res_bg
                    txt_active = c_res_txt
                    weight = "600"
                elif idx > curr_idx: # Support
                    bg_active = c_sup_bg
                    txt_active = c_sup_txt
                    weight = "600"
                else: # Current
                    # Current Row is special: Full Gold
                    return [f'background-color: {c_current_bg}; color: {c_current_txt}; font-weight: bold; text-align: center !important; vertical-align: middle;'] * len(row)

                # Default (Empty/Price) Style
                bg_neutral = c_neutral_bg
                txt_neutral = c_neutral_txt
                
                # Helper to check if string is non-empty
                def is_populated(val):
                    return bool(val and str(val).strip())

                # Col 0: Support/Resistance Levels (Left) => Right Align
                # Using iloc to access by position is safer if col names change slightly
                val_left = row.iloc[0] 
                if is_populated(val_left):
                    s_left = f'background-color: {bg_active}; color: {txt_active}; font-weight: {weight};'
                else:
                    s_left = f'background-color: {bg_neutral}; color: {txt_neutral};'
                styles.append(f'{s_left} text-align: right !important; padding-right: 15px; vertical-align: middle;')
                
                # Col 1: Price (Center) => Center Align
                styles.append(f'background-color: {bg_neutral}; color: {txt_neutral}; font-weight: normal; text-align: center !important; vertical-align: middle;')
                
                # Col 2: Key Turning Points (Right) => Left Align
                val_right = row.iloc[2]
                if is_populated(val_right):
                    s_right = f'background-color: {bg_active}; color: {txt_active}; font-weight: {weight};'
                else:
                    s_right = f'background-color: {bg_neutral}; color: {txt_neutral};'
                styles.append(f'{s_right} text-align: left !important; padding-left: 15px; vertical-align: middle;')
                
                return styles

            # Display
            final_view = disp_df.drop(columns=["Type"])
            
            st.dataframe(
                final_view.style.apply(style_barchart, axis=1), 
                use_container_width=True, 
                height=(len(final_view) + 1) * 35,
                hide_index=True
            )
            
            # Legend and Explanation
            st.markdown("""
            <div style="margin-top: 20px; font-size: 0.9em; color: #888;">
            <p><strong>Standard deviation</strong> is calculated using the closing price over the past 20-periods. To calculate standard deviation:</p>
            <ul style="list-style-type: disc; margin-left: 20px;">
                <li><strong>Step 1:</strong> Average = Calculate the average closing price over the past 20-days.</li>
                <li><strong>Step 2:</strong> Difference = Calculate the variance from the Average for each Price.</li>
                <li><strong>Step 3:</strong> Square the variance of each data point.</li>
                <li><strong>Step 4:</strong> Sum of the squared variance value.</li>
                <li><strong>Step 5:</strong> For Standard Deviation 2 multiple the result by 2. For Standard Deviation 3 multiple the result by 3.</li>
                <li><strong>Step 6:</strong> Divide the result by the number of data points in the series less 1.</li>
                <li><strong>Step 7:</strong> The final result is the Square root of the result of Step 6.</li>
            </ul>
            
            <p><strong>Legend:</strong></p>
            <ul style="list-style-type: none; padding-left: 10px;">
                <li><span style="color: #E8F5E9; background-color: #E8F5E9; border: 1px solid #ccc;">&nbsp;&nbsp;&nbsp;&nbsp;</span> <strong>Green areas below the Last Price</strong> will tend to provide support to limit the downward move.</li>
                <li><span style="color: #FFEBEE; background-color: #FFEBEE; border: 1px solid #ccc;">&nbsp;&nbsp;&nbsp;&nbsp;</span> <strong>Red areas above the Last Price</strong> will tend to provide resistance to limit the upward move.</li>
            </ul>
            </div>
            """, unsafe_allow_html=True)

def render_returns_analysis(port_series, bench_series=None, comparison_series=None, unique_id="", portfolio_name="Strategy", component_data=None, raw_port_series=None):
    daily_ret = port_series.pct_change().dropna()
    monthly_ret = port_series.resample("ME").last().pct_change().dropna()
    quarterly_ret = port_series.resample("QE").last().pct_change().dropna()
    annual_ret = port_series.resample("YE").last().pct_change().dropna()

    # --- HEATMAP HELPERS ---
    def render_seasonal_summary(series, suffix=""):
        full_suffix = f"{unique_id}_{suffix}" if unique_id else suffix
        if series.empty: return
        
        # 1. Prepare Data (Same as Monthly View)
        m_ret = series.resample("ME").last().pct_change().dropna()
        df_monthly = m_ret.to_frame(name="Return")
        df_monthly["Year"] = df_monthly.index.year
        df_monthly["Month"] = df_monthly.index.month
        
        pivot = df_monthly.pivot(index="Year", columns="Month", values="Return")
        for i in range(1, 13):
            if i not in pivot.columns: pivot[i] = float("nan")
        pivot = pivot.sort_index(ascending=True)
        month_names = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
        
        # 2. Add Yearly Return Column
        yearly_col = []
        years = pivot.index
        for y in years:
            row = pivot.loc[y]
            ret = (1 + row.fillna(0)).prod() - 1
            yearly_col.append(ret)
        
        # Add Yearly column to pivot for display/calculation
        display_pivot = pivot.copy()
        display_pivot.columns = month_names
        # display_pivot["Yearly Return"] = yearly_col # Removed per user request
        
        # 3. Calculate Summary Statistics Table
        st.subheader("Summary")
        
        # Create a DF for stats calc: Join pivot (months) and yearly_col
        stats_source = display_pivot.copy() # Columns: Jan..Dec, Yearly Return
        
        # Define rows
        stats_rows = ["Average", "% Positive", "% Negative", "Median", "Best", "Worst", "Abs Average", "Abs Best", "Abs Worst"]
        stats_df = pd.DataFrame(index=stats_rows, columns=stats_source.columns)
        
        for col in stats_source.columns:
            s_data = stats_source[col].dropna()
            if s_data.empty:
                continue
                
            stats_df.loc["Average", col] = s_data.mean()
            stats_df.loc["% Positive", col] = (s_data > 0).mean()
            stats_df.loc["% Negative", col] = (s_data < 0).mean()
            stats_df.loc["Median", col] = s_data.median()
            stats_df.loc["Best", col] = s_data.max()
            stats_df.loc["Worst", col] = s_data.min()
            stats_df.loc["Abs Average", col] = s_data.abs().mean()
            stats_df.loc["Abs Best", col] = s_data.abs().max()
            stats_df.loc["Abs Worst", col] = s_data.abs().min()
            
        stats_df = stats_df.astype(float)
        
        return_rows = ["Average", "Median", "Best", "Worst"]
        pct_rows = ["% Positive", "% Negative"]
        abs_rows = ["Abs Average", "Abs Best", "Abs Worst"]
        
        def style_summary(df):
            return df.style.format("{:.2%}") \
                .map(color_return, subset=pd.IndexSlice[return_rows, :]) \
                .background_gradient(cmap="Greens", subset=pd.IndexSlice["% Positive", :], vmin=0, vmax=1) \
                .background_gradient(cmap="Reds", subset=pd.IndexSlice["% Negative", :], vmin=0, vmax=1) \
                .background_gradient(cmap="Oranges", subset=pd.IndexSlice[abs_rows, :])

        st.dataframe(style_summary(stats_df), use_container_width=True)

    # --- HEATMAP HELPERS ---
    def render_quarterly_returns_view(series, suffix=""):
        # Combine unique_id with suffix for truly unique keys
        full_suffix = f"{unique_id}_{suffix}" if unique_id else suffix
        if series.empty: return
        quarterly_ret = series.resample("QE").last().pct_change().dropna()

        
        q_ret = quarterly_ret.to_frame(name="Return")
        q_ret["Year"] = q_ret.index.year
        q_ret["Quarter"] = q_ret.index.quarter
        q_ret["Quarter Name"] = "Q" + q_ret["Quarter"].astype(str)
        
        pivot = q_ret.pivot(index="Year", columns="Quarter", values="Return")
        for i in range(1, 5):
            if i not in pivot.columns: pivot[i] = float("nan")
        pivot = pivot.sort_index(ascending=True)
        


        quarter_names = ["Q1", "Q2", "Q3", "Q4"]
        quarterly_avgs = pivot.mean()
        z_data = pivot.values
        z_avgs = quarterly_avgs.values.reshape(1, -1)
        z_combined_main = np.concatenate([z_data, z_avgs], axis=0)
        
        years = pivot.index
        yearly_col = []
        for y in years:
            # Calculate geometric sum of the quarters for consistency
            row = pivot.loc[y]
            ret = (1 + row.fillna(0)).prod() - 1
            yearly_col.append(ret)
        yearly_avg = np.nanmean(yearly_col) if len(yearly_col) > 0 else float("nan")
        z_combined_yearly = np.array(yearly_col + [yearly_avg]).reshape(-1, 1)

        y_labels = [str(y) for y in pivot.index] + ["Average"]
        
        # Scaling
        std_dev_main = np.nanstd(z_combined_main * 100)
        scale_main = 2 * std_dev_main if not np.isnan(std_dev_main) else 10
        std_dev_yearly = np.nanstd(z_combined_yearly * 100)
        scale_yearly = 2 * std_dev_yearly if not np.isnan(std_dev_yearly) else 10
        colorscale_heatmap = [[0, '#E53935'], [0.5, '#FFFFFF'], [1, '#43A047']]

        # Hover Text
        date_map = {}
        try:
            periods = series.index.to_period("Q")
            for p, dates in series.index.groupby(periods).items():
                if not dates.empty:
                    date_map[(p.year, p.quarter)] = f"{dates.min().strftime('%b %d')} - {dates.max().strftime('%b %d')}"
        except (AttributeError, TypeError, ValueError):
            pass

        hover_main = []
        z_rounded_main = (z_combined_main * 100).round(2)
        for i, row_label in enumerate(y_labels):
            row_txt = []
            for j, col_label in enumerate(quarter_names):
                val = z_rounded_main[i][j]
                if np.isnan(val): row_txt.append("")
                elif row_label == "Average": row_txt.append(f"Average<br>{col_label}: {val:+.2f}%")
                else:
                    dr = date_map.get((int(row_label), j+1), "") if row_label.isdigit() else ""
                    dr_str = f"<br>{dr}" if dr else ""
                    row_txt.append(f"Year: {row_label}<br>{col_label}: {val:+.2f}%{dr_str}")
            hover_main.append(row_txt)

        hover_yearly = []
        z_rounded_yearly = (z_combined_yearly * 100).round(2)
        for i, row_label in enumerate(y_labels):
            val = z_rounded_yearly[i][0]
            val_str = "" if np.isnan(val) else f"{val:+.2f}%"
            if row_label == "Average": hover_yearly.append([f"Average Annual<br>{val_str}"])
            else: hover_yearly.append([f"Year: {row_label}<br>Annual: {val_str}"])

        # Plot
        n_years = len(y_labels) - 1
        fig = make_subplots(rows=2, cols=2, shared_xaxes=True, shared_yaxes=True,
            horizontal_spacing=0.03, vertical_spacing=0.02, column_widths=[0.85, 0.15],
            row_heights=[n_years/(n_years+1), 1/(n_years+1)]
        )

        fig.add_trace(go.Heatmap(z=(z_combined_main[:-1] * 100).round(2), x=quarter_names, y=y_labels[:-1], colorscale=colorscale_heatmap, zmid=0, zmin=-scale_main, zmax=scale_main, texttemplate="%{z:+.2f}%", hoverinfo="text", hovertext=hover_main[:-1], showscale=False), row=1, col=1)
        fig.add_trace(go.Heatmap(z=(z_combined_yearly[:-1] * 100).round(2), x=["Yearly"], y=y_labels[:-1], colorscale=colorscale_heatmap, zmid=0, zmin=-scale_yearly, zmax=scale_yearly, texttemplate="%{z:+.2f}%", hoverinfo="text", hovertext=hover_yearly[:-1], showscale=False), row=1, col=2)
        fig.add_trace(go.Heatmap(z=(z_combined_main[-1:] * 100).round(2), x=quarter_names, y=y_labels[-1:], colorscale=colorscale_heatmap, zmid=0, zmin=-scale_main, zmax=scale_main, texttemplate="%{z:+.2f}%", hoverinfo="text", hovertext=hover_main[-1:], showscale=False), row=2, col=1)
        fig.add_trace(go.Heatmap(z=(z_combined_yearly[-1:] * 100).round(2), x=["Yearly"], y=y_labels[-1:], colorscale=colorscale_heatmap, zmid=0, zmin=-scale_yearly, zmax=scale_yearly, texttemplate="%{z:+.2f}%", hoverinfo="text", hovertext=hover_yearly[-1:], showscale=False), row=2, col=2)

        fig.update_layout(title="Quarterly Returns Heatmap (%)", template="plotly_white", height=max(400, (len(y_labels)+1)*30), yaxis=dict(autorange="reversed", type="category"), yaxis3=dict(autorange="reversed", type="category"))
        fig.update_yaxes(showticklabels=False, col=2)
        st.plotly_chart(fig, use_container_width=True, key=f"q_hm_{full_suffix}")
        
        st.subheader("Quarterly Returns List")
        df_quarterly_list = q_ret.copy()
        df_quarterly_list["Period"] = df_quarterly_list.index.to_period("Q").astype(str)
        df_quarterly_list = df_quarterly_list[["Period", "Return"]].sort_index(ascending=False)
        
        st.dataframe(
            df_quarterly_list.style.format({"Return": "{:+.1%}"}).map(color_return, subset=["Return"]),
            use_container_width=True,
            hide_index=True
        )

    def render_monthly_returns_view(series, suffix=""):
        # Combine unique_id with suffix for truly unique keys
        full_suffix = f"{unique_id}_{suffix}" if unique_id else suffix
        if series.empty: return
        m_ret = series.resample("ME").last().pct_change().dropna()

        
        df_monthly = m_ret.to_frame(name="Return")
        df_monthly["Year"] = df_monthly.index.year
        df_monthly["Month"] = df_monthly.index.month
        
        pivot = df_monthly.pivot(index="Year", columns="Month", values="Return")
        for i in range(1, 13):
            if i not in pivot.columns: pivot[i] = float("nan")
        pivot = pivot.sort_index(ascending=True)
        month_names = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
        monthly_avgs = pivot.mean()

        z_data = pivot.values
        z_combined_main = np.concatenate([z_data, monthly_avgs.values.reshape(1, -1)], axis=0)
        
        years = pivot.index
        yearly_col = []
        for y in years:
            # Calculate geometric sum of the months for consistency
            # This ensures the Yearly column matches the compounded value of the displayed months
            row = pivot.loc[y]
            # compound: product(1+r) - 1. Treat NaNs as 0 (no return for that period)
            ret = (1 + row.fillna(0)).prod() - 1
            yearly_col.append(ret)
        yearly_avg = np.nanmean(yearly_col) if len(yearly_col) > 0 else float("nan")
        z_combined_yearly = np.array(yearly_col + [yearly_avg]).reshape(-1, 1)

        y_labels = [str(y) for y in pivot.index] + ["Average"]
        
        std_dev_main = np.nanstd(z_combined_main * 100)
        scale_main = 2 * std_dev_main if not np.isnan(std_dev_main) else 10
        std_dev_yearly = np.nanstd(z_combined_yearly * 100)
        scale_yearly = 1.0 * std_dev_yearly if not np.isnan(std_dev_yearly) else 10
        colorscale_heatmap = [[0, '#E53935'], [0.5, '#FFFFFF'], [1, '#43A047']]
        
        hover_main = []
        z_rounded_main = (z_combined_main * 100).round(2)
        for i, row_label in enumerate(y_labels):
            row_txt = []
            for j, col_label in enumerate(month_names):
                val = z_rounded_main[i][j]
                val_str = "" if np.isnan(val) else f"{val:+.2f}%"
                row_txt.append(f"{row_label} {col_label}<br>{val_str}")
            hover_main.append(row_txt)

        hover_yearly = []
        z_rounded_yearly = (z_combined_yearly * 100).round(2)
        for i, row_label in enumerate(y_labels):
            val = z_rounded_yearly[i][0]
            val_str = "" if np.isnan(val) else f"{val:+.2f}%"
            hover_yearly.append([f"{row_label} Total<br>{val_str}"])

        n_years = len(y_labels) - 1
        fig = make_subplots(rows=2, cols=2, shared_xaxes=True, shared_yaxes=True,
            horizontal_spacing=0.03, vertical_spacing=0.02, column_widths=[0.85, 0.15],
            row_heights=[n_years/(n_years+1), 1/(n_years+1)]
        )

        fig.add_trace(go.Heatmap(z=(z_combined_main[:-1] * 100).round(2), x=month_names, y=y_labels[:-1], colorscale=colorscale_heatmap, zmid=0, zmin=-scale_main, zmax=scale_main, texttemplate="%{z:+.2f}%", hoverinfo="text", hovertext=hover_main[:-1], showscale=False), row=1, col=1)
        fig.add_trace(go.Heatmap(z=(z_combined_yearly[:-1] * 100).round(2), x=["Yearly"], y=y_labels[:-1], colorscale=colorscale_heatmap, zmid=0, zmin=-scale_yearly, zmax=scale_yearly, texttemplate="%{z:+.2f}%", hoverinfo="text", hovertext=hover_yearly[:-1], showscale=False), row=1, col=2)
        fig.add_trace(go.Heatmap(z=(z_combined_main[-1:] * 100).round(2), x=month_names, y=y_labels[-1:], colorscale=colorscale_heatmap, zmid=0, zmin=-scale_main, zmax=scale_main, texttemplate="%{z:+.2f}%", hoverinfo="text", hovertext=hover_main[-1:], showscale=False), row=2, col=1)
        fig.add_trace(go.Heatmap(z=(z_combined_yearly[-1:] * 100).round(2), x=["Yearly"], y=y_labels[-1:], colorscale=colorscale_heatmap, zmid=0, zmin=-scale_yearly, zmax=scale_yearly, texttemplate="%{z:+.2f}%", hoverinfo="text", hovertext=hover_yearly[-1:], showscale=False), row=2, col=2)

        fig.update_layout(title="Monthly Returns Heatmap (%)", template="plotly_white", height=max(400, (len(y_labels)+1)*30), yaxis=dict(autorange="reversed", type="category"), yaxis3=dict(autorange="reversed", type="category"))
        fig.update_yaxes(showticklabels=False, col=2)
        st.plotly_chart(fig, use_container_width=True, key=f"m_hm_{full_suffix}")

    
    tab_summary, tab_annual, tab_quarterly, tab_monthly, tab_daily = st.tabs(["📋 Summary", "📅 Annual", "📆 Quarterly", "🗓️ Monthly", "📊 Daily"])

    with tab_summary:
        st.subheader(f"{portfolio_name} Seasonal Summary")
        render_seasonal_summary(port_series)
    
    with tab_annual:
        st.subheader(f"{portfolio_name} Annual Returns")
        
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
            df_annual.style.format({"Return": "{:+.1%}"}).map(color_return, subset=["Return"]),
            use_container_width=True,
            hide_index=True
        )

    with tab_quarterly:
        qt_tabs = [portfolio_name]
        if comparison_series is not None and not comparison_series.empty: qt_tabs.append("Benchmark (Comparison)")
        if bench_series is not None and not bench_series.empty: qt_tabs.append("Benchmark (Primary)")
            
        q_view_tabs = st.tabs(qt_tabs)
        with q_view_tabs[0]:
            st.subheader(f"{portfolio_name} Quarterly Returns")
            render_quarterly_returns_view(port_series)
        
        if len(qt_tabs) > 1 and "Benchmark (Comparison)" in qt_tabs:
            with q_view_tabs[qt_tabs.index("Benchmark (Comparison)")]:
                st.subheader("Standard Rebalance (Comparison) Quarterly Returns")
                render_quarterly_returns_view(comparison_series, suffix="_comp")
        
        if len(qt_tabs) > 1 and "Benchmark (Primary)" in qt_tabs:
             with q_view_tabs[qt_tabs.index("Benchmark (Primary)")]:
                st.subheader("Primary Benchmark Quarterly Returns")
                render_quarterly_returns_view(bench_series, suffix="_bench")

    with tab_monthly:
        hm_tabs = [portfolio_name]
        if comparison_series is not None and not comparison_series.empty: hm_tabs.append("Benchmark (Comparison)")
        if bench_series is not None and not bench_series.empty: hm_tabs.append("Benchmark (Primary)")
            
        m_view_tabs = st.tabs(hm_tabs)
        with m_view_tabs[0]:
            st.subheader(f"{portfolio_name} Monthly Returns")
            render_monthly_returns_view(port_series)
            
            st.subheader("Monthly Returns List")
            df_monthly_list = monthly_ret.to_frame(name="Return")
            df_monthly_list["Date"] = df_monthly_list.index.strftime("%Y-%m")
            df_monthly_list = df_monthly_list[["Date", "Return"]].sort_index(ascending=False)
            st.dataframe(df_monthly_list.style.format({"Return": "{:+.1%}"}).map(color_return, subset=["Return"]), use_container_width=True, hide_index=True)

        if len(hm_tabs) > 1 and "Benchmark (Comparison)" in hm_tabs:
            with m_view_tabs[hm_tabs.index("Benchmark (Comparison)")]:
                st.subheader("Standard Rebalance (Comparison) Monthly Returns")
                render_monthly_returns_view(comparison_series, suffix="_comp")
                
        if len(hm_tabs) > 1 and "Benchmark (Primary)" in hm_tabs:
             with m_view_tabs[hm_tabs.index("Benchmark (Primary)")]:
                st.subheader("Primary Benchmark Monthly Returns")
                render_monthly_returns_view(bench_series, suffix="_bench")

    with tab_daily:
        st.subheader(f"{portfolio_name} Daily Returns")
        
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
            df_daily_list.style.format({"Return": "{:+.1%}"}).map(color_return, subset=["Return"]),
            use_container_width=True,
            hide_index=True
        )



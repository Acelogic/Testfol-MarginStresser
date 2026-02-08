import plotly.graph_objects as go
import plotly.express as px
import streamlit as st
import pandas as pd
import numpy as np
import datetime as dt

from app.common.utils import color_return
from app.core import tax_library, calculations

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
    
    fig = px.bar(
        df, 
        y="Date Label", 
        x="Value", 
        color="Ticker", 
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


def render_rebalancing_analysis(trades_df, pl_by_year, composition_df, tax_method, other_income, filing_status, state_tax_rate, rebalance_freq="Yearly", use_standard_deduction=True, unrealized_pl_df=None, custom_freq="Yearly", unique_id=None):
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
            
            # Add Unrealized P&L Trace
            if "Unrealized P&L" in agg_df.columns and agg_df["Unrealized P&L"].abs().sum() > 0:
                 fig.add_trace(go.Bar(
                    x=x_axis, 
                    y=agg_df["Unrealized P&L"], 
                    name="Unrealized P&L (Deferred)",
                    marker_color="#636EFA", # Blue/Purple
                    opacity=0.6, # Slightly transparent to distinguish from realized
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
            xaxis_title=view_freq[:-2], # Year/Quarter/Month
            template="plotly_dark",
            showlegend=True,
            height=400,
            hovermode="x unified"
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Estimated Tax Owed Chart
        if not pl_by_year.empty:
            st.subheader(f"Estimated Total Tax Owed ({view_freq} - {tax_method})")
            
            # 1. Calculate Annual Federal Tax (Base)
            federal_tax_annual = tax_library.calculate_tax_series_with_carryforward(
                pl_by_year, 
                other_income,
                filing_status,
                method=tax_method,
                use_standard_deduction=use_standard_deduction
            )
            
            # 2. Calculate Annual State Tax (Base)
            state_tax_annual = pd.Series(0.0, index=federal_tax_annual.index)
            loss_cf = 0.0
            total_pl_series = pl_by_year["Realized P&L"] if isinstance(pl_by_year, pd.DataFrame) else pl_by_year
            
            for y, pl in total_pl_series.sort_index().items():
                net = pl - loss_cf
                if net > 0:
                    state_tax_annual[y] = net * state_tax_rate
                    loss_cf = 0.0
                else:
                    loss_cf = abs(net)
            
            total_tax_annual = federal_tax_annual + state_tax_annual
            
            # 3. Allocate to Periods if needed
            if view_freq == "Yearly":
                tax_to_plot = total_tax_annual
                x_axis_tax = tax_to_plot.index
            else:
                # Allocate annual tax to periods based on realized gains
                # We need the period data (agg_df) which we calculated above
                
                # Create a Series to hold allocated tax
                tax_to_plot = pd.Series(0.0, index=agg_df.index)
                
                # Iterate through each year
                for year in total_tax_annual.index:
                    annual_tax = total_tax_annual.get(year, 0.0)
                    if annual_tax <= 0:
                        continue
                        
                    # Get periods for this year
                    if view_freq == "Quarterly":
                        # Filter agg_df for this year (Quarter index)
                        # Period index is like "2021Q1"
                        periods_in_year = [p for p in agg_df.index if p.year == year]
                    elif view_freq == "Monthly":
                        periods_in_year = [p for p in agg_df.index if p.year == year]
                    elif view_freq == "Per Event":
                        periods_in_year = [p for p in agg_df.index if p.year == year]
                    
                    # Calculate total POSITIVE realized P&L for this year from the periods
                    # We only allocate tax to periods that had gains
                    year_gains = 0.0
                    period_gains = {}
                    
                    for p in periods_in_year:
                        gain = agg_df.loc[p, "Realized P&L"]
                        if gain > 0:
                            year_gains += gain
                            period_gains[p] = gain
                        else:
                            period_gains[p] = 0.0
                            
                    # Allocate
                    if year_gains > 0:
                        for p in periods_in_year:
                            if period_gains[p] > 0:
                                allocation_ratio = period_gains[p] / year_gains
                                tax_to_plot[p] = annual_tax * allocation_ratio
                
                x_axis_tax = tax_to_plot.index.astype(str)

            if tax_to_plot.sum() > 0:
                fig_tax = go.Figure(go.Bar(
                    x=x_axis_tax,
                    y=tax_to_plot,
                    marker_color="#EF553B", # Red for taxes
                    texttemplate="%{y:$.2s}",
                    textposition="auto",
                    hovertemplate="%{y:$,.0f}<extra></extra>",
                    name="Estimated Tax"
                ))
                fig_tax.update_layout(
                    yaxis_title="Tax Owed ($)",
                    xaxis_title=view_freq[:-2], # Year/Quarter/Month
                    template="plotly_dark",
                    showlegend=False,
                    height=400,
                    hovermode="x unified"
                )
                st.plotly_chart(fig_tax, use_container_width=True)
                
                total_tax = tax_to_plot.sum()
                st.metric("Total Estimated Tax Owed", f"${total_tax:,.2f}")
            else:
                st.info("No taxable gains realized.")
        
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

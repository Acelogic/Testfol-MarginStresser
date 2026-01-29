"""
NDX-100 Moving Average Scanner

Scans all current Nasdaq 100 components and identifies which are trading
below their 200-Day Moving Average or 200-Week Moving Average (Munger).
"""

import streamlit as st
import pandas as pd
import yfinance as yf
import os
import json
import time
from datetime import datetime, timedelta
from app.core import calculations


# Paths to data files
DATA_DIR = os.path.join(os.path.dirname(__file__), "../../data/ndx_simulation/data")
NDX_COMPONENTS_FILE = os.path.join(DATA_DIR, "assets/nasdaq_components.csv")
NAME_MAPPING_FILE = os.path.join(DATA_DIR, "assets/name_mapping.json")


def get_current_ndx_components():
    """
    Get current NDX-100 components from nasdaq_components.csv.
    Uses name_mapping.json to translate company names to tickers.
    Returns DataFrame with Ticker and Name columns.
    """
    try:
        # Load components
        df = pd.read_csv(NDX_COMPONENTS_FILE)
        df['Date'] = pd.to_datetime(df['Date'])
        
        # Filter out corrupt rows (Company column should not look like dates)
        # Valid company names don't start with digits
        df = df[~df['Company'].str.match(r'^\d{4}-\d{2}-\d{2}', na=False)]
        
        # Also ensure Value is numeric and reasonable
        df['Value'] = pd.to_numeric(df['Value'], errors='coerce')
        df = df[df['Value'] > 0]
        
        # Filter out future dates
        today = pd.Timestamp.now().normalize()
        df = df[df['Date'] <= today]
        
        # Get latest date
        latest_date = df['Date'].max()
        
        # Filter to latest date
        current = df[df['Date'] == latest_date].copy()
        
        # Load name mapping
        with open(NAME_MAPPING_FILE, 'r') as f:
            name_mapping = json.load(f)
        
        # Map company names to tickers
        current['Ticker'] = current['Company'].map(name_mapping)
        
        # Filter out unmapped companies
        current = current[current['Ticker'].notna()].copy()
        
        # Calculate weight from Value
        total_value = current['Value'].sum()
        current['Weight'] = current['Value'] / total_value
        
        # Rename for consistency
        current = current.rename(columns={'Company': 'Name'})
        
        # Sort by weight descending
        current = current.sort_values('Weight', ascending=False)
        
        return current[['Ticker', 'Name', 'Weight']], latest_date
        
    except Exception as e:
        st.error(f"Failed to load NDX components: {e}")
        return pd.DataFrame(), None


@st.cache_data(ttl=900, show_spinner=False)  # 15-minute cache
def fetch_ma_data(tickers: list, tolerance_days: int = 0):
    """
    Fetch price data and calculate 200-day SMA for all tickers.
    Returns DataFrame with: Ticker, Price, SMA_200, Distance_Pct, Status, Max Depth, Duration
    """
    if not tickers:
        return pd.DataFrame()
    
    results = []
    
    # Batch download for efficiency
    try:
        # Get 2 years of data (need 200+ trading days for SMA + buffer for finding crossover)
        data = yf.download(
            tickers, 
            period="2y", 
            auto_adjust=True, 
            threads=True, 
            progress=False
        )['Close']
        
        # Handle single ticker case (returns Series instead of DataFrame)
        if isinstance(data, pd.Series):
            data = data.to_frame(tickers[0])
        
        for ticker in tickers:
            try:
                if ticker not in data.columns:
                    continue
                    
                series = data[ticker].dropna()
                
                if len(series) < 200:
                    continue
                
                # Use shared analysis logic (handles Merge Tolerance)
                dma_series, events_df = calculations.analyze_ma(series, window=200, tolerance_days=tolerance_days)
                
                if dma_series is None or dma_series.empty:
                    continue

                current_price = series.iloc[-1]
                sma_200 = dma_series.iloc[-1]
                
                distance_pct = ((current_price - sma_200) / sma_200) * 100
                
                status = "🟢 Above"
                duration = 0
                max_deviation = 0.0
                
                # Determine Status from Events
                if not events_df.empty:
                    last_event = events_df.iloc[-1]
                    
                    if last_event["Status"] == "Ongoing":
                        status = "🔴 Below"
                        duration = last_event["Duration (Days)"] # Days Under
                        
                        # Calculate Max Depth (as Drawdown % from Start Price) for this event
                        # This matches calculations.py logic logic
                        start_date = last_event["Start Date"]
                        
                        # Get data for the event duration
                        # We need Price series to calculate drawdown
                        event_prices = series[start_date:]
                        
                        if not event_prices.empty:
                            start_price = event_prices.iloc[0] # Price at crossover
                            min_price = event_prices.min()     # Lowest price during event
                            
                            # Drawdown % = (Min - Start) / Start
                            # Should be negative
                            max_deviation = ((min_price - start_price) / start_price) * 100
                    else:
                        # Last event recovered. We are currently Above.
                        # Duration = days since recovery?
                        last_end = last_event["End Date"]
                        if pd.notna(last_end):
                            duration = (series.index[-1] - last_end).days
                            
                            # Calculate Max Peak (as Deviation %) for this 'Above' period
                            # Start checking from the day AFTER the recovery
                            # (or strictly > last_end)
                            # current_above_prices = series[last_end:].iloc[1:] # Skip the crossover day if it was the recovery day?
                            # Using direct indexing is safer: data after last_end
                            
                            valid_range = series.index > last_end
                            above_prices = series[valid_range]
                            above_sma = dma_series[valid_range]
                            
                            if not above_prices.empty:
                                deviations = ((above_prices - above_sma) / above_sma) * 100
                                max_deviation = deviations.max() # Most positive deviation (Peak)
                        else:
                            duration = 0 
                else:
                    # Never below SMA in this period (Always Above)
                     duration = len(series) # Acts as "Days Above"
                     
                     # Calculate Peak for the entire loaded period
                     deviations = ((series - dma_series) / dma_series) * 100
                     max_deviation = deviations.max()

                results.append({
                    'Ticker': ticker,
                    'Price': current_price,
                    'SMA_200': sma_200,
                    'Distance %': distance_pct,
                    'Max Depth / Peak': max_deviation,
                    'Status': status,
                    'Duration': duration
                })
                
            except Exception as e:
                # Debug info only if failed
                # st.error(f"Error {ticker}: {e}") 
                continue
                
    except Exception as e:
        st.error(f"Failed to fetch price data: {e}")
        return pd.DataFrame()
    
    return pd.DataFrame(results)


@st.cache_data(ttl=900, show_spinner=False)  # 15-minute cache
def fetch_wma_data(tickers: list, tolerance_weeks: int = 0):
    """
    Fetch price data and calculate 200-week WMA for all tickers.
    Returns DataFrame with: Ticker, Price, WMA_200, Distance_Pct, Status, Max Depth, Duration (weeks)
    """
    if not tickers:
        return pd.DataFrame()

    results = []

    try:
        # Get 5 years of data (need 200+ weeks for WMA + buffer)
        data = yf.download(
            tickers,
            period="5y",
            auto_adjust=True,
            threads=True,
            progress=False
        )['Close']

        # Handle single ticker case
        if isinstance(data, pd.Series):
            data = data.to_frame(tickers[0])

        for ticker in tickers:
            try:
                if ticker not in data.columns:
                    continue

                series = data[ticker].dropna()

                if len(series) < 200 * 5:  # Need ~200 weeks of daily data
                    continue

                # Use the weekly MA analysis
                weekly_series, wma_series, events_df = calculations.analyze_wma(
                    series, window=200, tolerance_weeks=tolerance_weeks
                )

                if wma_series is None or wma_series.empty:
                    continue

                current_price = weekly_series.iloc[-1]
                wma_200 = wma_series.iloc[-1]

                distance_pct = ((current_price - wma_200) / wma_200) * 100

                status = "🟢 Above"
                duration = 0
                max_deviation = 0.0

                # Determine Status from Events
                if not events_df.empty:
                    last_event = events_df.iloc[-1]

                    if last_event["Status"] == "Ongoing":
                        status = "🔴 Below"
                        duration = last_event["Duration (Weeks)"]

                        # Calculate Max Depth for this event
                        start_date = last_event["Start Date"]
                        event_prices = weekly_series[start_date:]

                        if not event_prices.empty:
                            start_price = event_prices.iloc[0]
                            min_price = event_prices.min()
                            max_deviation = ((min_price - start_price) / start_price) * 100
                    else:
                        # Last event recovered - currently Above
                        last_end = last_event["End Date"]
                        if pd.notna(last_end):
                            # Count weeks since recovery
                            duration = len(weekly_series[weekly_series.index > last_end])

                            # Calculate Max Peak for this 'Above' period
                            valid_range = weekly_series.index > last_end
                            above_prices = weekly_series[valid_range]
                            above_wma = wma_series[valid_range]

                            if not above_prices.empty and not above_wma.empty:
                                deviations = ((above_prices - above_wma) / above_wma) * 100
                                max_deviation = deviations.max()
                        else:
                            duration = 0
                else:
                    # Never below WMA in this period
                    duration = len(weekly_series)
                    deviations = ((weekly_series - wma_series) / wma_series) * 100
                    max_deviation = deviations.dropna().max() if not deviations.dropna().empty else 0.0

                results.append({
                    'Ticker': ticker,
                    'Price': current_price,
                    'WMA_200': wma_200,
                    'Distance %': distance_pct,
                    'Max Depth / Peak': max_deviation,
                    'Status': status,
                    'Duration': duration
                })

            except Exception:
                continue

    except Exception as e:
        st.error(f"Failed to fetch price data: {e}")
        return pd.DataFrame()

    return pd.DataFrame(results)


def render_ndx_scanner():
    """
    Main renderer for the NDX-100 Moving Average Scanner.
    """
    st.header("📊 NDX-100 Moving Average Scanner")

    # MA Type selector
    ma_type = st.radio(
        "Moving Average Type",
        ["200 SMA (Daily)", "200 WMA (Weekly - Munger)"],
        horizontal=True,
        help="**200 SMA**: 200-day moving average (short-term trends)\n\n**200 WMA**: 200-week moving average (~4 years, secular trends)"
    )

    is_weekly = ma_type == "200 WMA (Weekly - Munger)"
    ma_label = "200 WMA" if is_weekly else "200 SMA"
    duration_label = "weeks" if is_weekly else "days"

    if is_weekly:
        st.caption("Identify Nasdaq 100 components trading below their 200-Week Moving Average (Munger Indicator)")
    else:
        st.caption("Identify Nasdaq 100 components trading below their 200-Day Moving Average")

    # Load components
    components_df, latest_date = get_current_ndx_components()

    # Assign Rank by Weight (it's already sorted by Weight)
    if not components_df.empty:
        components_df['Rank'] = range(1, len(components_df) + 1)

    if components_df.empty:
        st.warning("No NDX component data available. Run the NDX rebuild script first.")
        return

    st.info(f"📅 **Data as of:** {latest_date.strftime('%Y-%m-%d')} | **Components:** {len(components_df)}")

    # Filter controls - different defaults for weekly vs daily
    c1, c2 = st.columns([3, 1])
    with c1:
        st.write("")  # Spacer
    with c2:
        if is_weekly:
            use_filters = st.checkbox(
                "Apply Noise Filter (2w)",
                value=True,
                help="Merges events with <2 week recovery gaps and ignores drops <4 weeks duration"
            )
            tol_param = 2 if use_filters else 0
            min_filter = 4 if use_filters else 0
        else:
            use_filters = st.checkbox(
                "Apply Noise Filter (14d)",
                value=True,
                help="Merges events with <14d recovery gaps and ignores drops <14d duration"
            )
            tol_param = 14 if use_filters else 0
            min_filter = 14 if use_filters else 0

    # Get tickers list
    tickers = components_df['Ticker'].tolist()

    # Fetch MA data with progress
    if is_weekly:
        with st.spinner(f"Fetching weekly price data for {len(tickers)} tickers (this may take longer)..."):
            ma_data = fetch_wma_data(tickers, tolerance_weeks=tol_param)
        ma_col = 'WMA_200'
    else:
        with st.spinner(f"Fetching price data for {len(tickers)} tickers..."):
            ma_data = fetch_ma_data(tickers, tolerance_days=tol_param)
        ma_col = 'SMA_200'

    if ma_data.empty:
        st.error("Failed to fetch moving average data.")
        return

    # Merge with component names
    result_df = ma_data.merge(
        components_df[['Ticker', 'Name', 'Weight', 'Rank']],
        on='Ticker',
        how='left'
    )

    # Reorder columns
    result_df = result_df[['Rank', 'Ticker', 'Name', 'Weight', 'Price', ma_col, 'Distance %', 'Max Depth / Peak', 'Duration', 'Status']]

    # Rename MA column for display
    result_df = result_df.rename(columns={ma_col: ma_label})

    # Filter controls
    col1, col2, col3 = st.columns([2, 2, 2])

    with col1:
        filter_option = st.selectbox(
            "Filter",
            ["All", f"🔴 Below {ma_label}", f"🟢 Above {ma_label}"],
            index=0
        )

    with col2:
        sort_option = st.selectbox(
            "Sort By",
            ["Distance % (Ascending)", "Distance % (Descending)", "Max Depth / Peak (Ascending)", "Max Depth / Peak (Descending)", "Weight (Descending)", "Ticker (A-Z)"],
            index=0
        )

    # Apply filter
    filtered_df = result_df.copy()

    if "Below" in filter_option:
        filtered_df = filtered_df[filtered_df['Distance %'] < 0]
        if use_filters:
            filtered_df = filtered_df[filtered_df['Duration'] >= min_filter]

    elif "Above" in filter_option:
        filtered_df = filtered_df[filtered_df['Distance %'] >= 0]

    # Apply sort
    if sort_option == "Distance % (Ascending)":
        filtered_df = filtered_df.sort_values('Distance %', ascending=True)
    elif sort_option == "Distance % (Descending)":
        filtered_df = filtered_df.sort_values('Distance %', ascending=False)
    elif sort_option == "Max Depth / Peak (Ascending)":
        filtered_df = filtered_df.sort_values('Max Depth / Peak', ascending=True)
    elif sort_option == "Max Depth / Peak (Descending)":
        filtered_df = filtered_df.sort_values('Max Depth / Peak', ascending=False)
    elif sort_option == "Weight (Descending)":
        filtered_df = filtered_df.sort_values('Weight', ascending=False)
    elif sort_option == "Ticker (A-Z)":
        filtered_df = filtered_df.sort_values('Ticker', ascending=True)

    # Summary stats
    below_count = len(result_df[result_df['Distance %'] < 0])
    above_count = len(result_df[result_df['Distance %'] >= 0])

    col_s1, col_s2, col_s3 = st.columns(3)
    col_s1.metric(f"🔴 Below {ma_label}", below_count)
    col_s2.metric(f"🟢 Above {ma_label}", above_count)
    col_s3.metric("Total Scanned", len(result_df))
    
    # --- Quick Analyze Action ---
    st.markdown("##### 🚀 Quick Analyze")
    qa_col1, qa_col2 = st.columns([3, 1])
    
    with qa_col1:
        # Get list of tickers from the current filtered view
        available_tickers = filtered_df['Ticker'].tolist()
        selected_ticker_analyze = st.selectbox(
            "Select Ticker to Analyze", 
            available_tickers if available_tickers else ["No tickers available"],
            key="scanner_ticker_select"
        )
        
    with qa_col2:
        # Align button with the selectbox input (compensate for label height)
        st.markdown('<div style="margin-top: 28px;"></div>', unsafe_allow_html=True)
        if st.button("Load as New Portfolio", type="primary", use_container_width=True, disabled=not available_tickers):
            if "portfolios" in st.session_state and st.session_state.portfolios:
                if len(st.session_state.portfolios) < 5:
                    import uuid
                    new_id = f"p_scan_{uuid.uuid4().hex[:8]}"
                    new_name = f"Analysis: {selected_ticker_analyze}"
                    
                    new_port = {
                        "id": new_id,
                        "name": new_name,
                        "alloc_df": pd.DataFrame([
                            {"Ticker": selected_ticker_analyze, "Weight %": 100.0, "Maint %": 25.0} # Default Maint
                        ]),
                        "rebalance": {
                            "mode": "Standard",
                            "freq": "Yearly",
                            "month": 1,
                            "day": 1,
                            "compare_std": False
                        },
                         "cashflow": {
                            "start_val": 10000.0,
                            "amount": 0.0, 
                            "freq": "Monthly", 
                            "invest_div": True,
                            "pay_down_margin": False
                        }
                    }
                    
                    st.session_state.portfolios.append(new_port)
                    
                    # Auto-select this new portfolio
                    # We cannot modify 'portfolio_selector' directly after it's been rendered.
                    # Instead, we update the index and clear the widget key to force a reset to default.
                    st.session_state.active_tab_idx = len(st.session_state.portfolios) - 1
                    
                    if "portfolio_selector" in st.session_state:
                        del st.session_state.portfolio_selector
                    
                    st.toast(f"✅ Added '{new_name}'! Switch to the Portfolio tab to view.", icon="🚀")
                    
                    # Rerun to refresh the app state
                    time.sleep(0.5) # smooth transition
                    st.rerun()
                else:
                    st.error("Maximum of 5 portfolios reached. Please delete one first.")
            else:
                st.error("Portfolio state not initialized.")
    
    # Display table with column configs for tooltips
    duration_unit = "weeks" if is_weekly else "days"
    ma_full_name = "200-Week Moving Average" if is_weekly else "200-Day Moving Average"

    column_config = {
        "Rank": st.column_config.NumberColumn(
            "Rank",
            help="NDX-100 weight ranking (1 = highest weight)",
            format="%d"
        ),
        "Ticker": st.column_config.TextColumn(
            "Ticker",
            help="Stock ticker symbol"
        ),
        "Name": st.column_config.TextColumn(
            "Name",
            help="Company name"
        ),
        "Weight": st.column_config.NumberColumn(
            "Weight",
            help="Weight in the NDX-100 index",
            format="%.2f%%"
        ),
        "Price": st.column_config.NumberColumn(
            "Price",
            help="Current stock price",
            format="$%.2f"
        ),
        ma_label: st.column_config.NumberColumn(
            ma_label,
            help=f"{ma_full_name} value",
            format="$%.2f"
        ),
        "Distance %": st.column_config.NumberColumn(
            "Distance %",
            help=f"Percentage distance from {ma_label}. Negative = below MA, Positive = above MA",
            format="%+.2f%%"
        ),
        "Max Depth / Peak": st.column_config.NumberColumn(
            "Max Depth / Peak",
            help=f"For stocks BELOW {ma_label}: Maximum drawdown from when price crossed below MA. For stocks ABOVE: Maximum peak above MA since recovery.",
            format="%+.2f%%"
        ),
        "Duration": st.column_config.NumberColumn(
            "Duration",
            help=f"For stocks BELOW {ma_label}: {duration_unit.capitalize()} since price dropped below MA. For stocks ABOVE: {duration_unit.capitalize()} since price recovered above MA.",
            format="%d"
        ),
        "Status": st.column_config.TextColumn(
            "Status",
            help=f"Current position relative to {ma_label}"
        )
    }

    # Apply styling for status colors
    styled_df = filtered_df.style.applymap(
        lambda x: 'color: #ff4b4b' if '🔴' in str(x) else 'color: #21c354',
        subset=['Status']
    )

    st.dataframe(
        styled_df,
        column_config=column_config,
        use_container_width=True,
        hide_index=True,
        height=600
    )

    st.caption(f"💡 Data cached for 15 minutes. Duration is in {duration_unit}. Refresh the page to update.")

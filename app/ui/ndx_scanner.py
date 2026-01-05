"""
NDX-100 Moving Average Scanner

Scans all current Nasdaq 100 components and identifies which are trading 
below their 200-Day Moving Average.
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
                        
                        # Calculate Max Depth (as Deviation %) for this event
                        start_date = last_event["Start Date"]
                        # Get data for the event duration
                        # We need both Price and SMA series to calculate deviation
                        # Filter to event period: Start -> Now
                        event_prices = series[start_date:]
                        event_sma = dma_series[start_date:]
                        
                        if not event_prices.empty:
                            # Deviation = (Price - SMA) / SMA
                            deviations = ((event_prices - event_sma) / event_sma) * 100
                            max_deviation = deviations.min() # Most negative deviation (Depth)
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


def render_ndx_scanner():
    """
    Main renderer for the NDX-100 Moving Average Scanner.
    """
    st.header("📊 NDX-100 Moving Average Scanner")
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
    
    # Filter controls
    c1, c2 = st.columns([3, 1])
    with c1:
        st.write("") # Spacer
    with c2:
        use_filters = st.checkbox("Apply Noise Filter (14d)", value=True, help="Merges events with <14d recovery gaps and ignores drops <14d duration")
    
    # Params based on checkbox
    tol_days = 14 if use_filters else 0
    min_days_filter = 14 if use_filters else 0

    # Get tickers list
    tickers = components_df['Ticker'].tolist()
    
    # Fetch MA data with progress
    with st.spinner(f"Fetching price data for {len(tickers)} tickers..."):
        ma_data = fetch_ma_data(tickers, tolerance_days=tol_days)
    
    if ma_data.empty:
        st.error("Failed to fetch moving average data.")
        return
    
    # Merge with component names
    result_df = ma_data.merge(
        components_df[['Ticker', 'Name', 'Weight', 'Rank']], 
        on='Ticker', 
        how='left'
    )
    
    # Apply Min Days Filter (Signal Filter)
    # If Status is Below, but Duration < 14, we filter it out (or reclassify?)
    # Usually "Signal Filter" means "Don't confirm the signal until X days".
    # So if Below and Duration < 14, treat as Above/Noise?
    # Simply filtering out "Below" rows that are too short is safer for a "Scanner".
    # If users look for "Below", they want CONFIRMED below.
    # Let's flag them? Or effectively treat them as "Above" for the "Below" filter?
    # Simple approach: If "Below" filter is selected, enforcing Duration >= 14.
    pass

    # Reorder columns
    result_df = result_df[['Rank', 'Ticker', 'Name', 'Weight', 'Price', 'SMA_200', 'Distance %', 'Max Depth / Peak', 'Duration', 'Status']]
    
    # Filter controls
    col1, col2, col3 = st.columns([2, 2, 2])
    
    with col1:
        filter_option = st.selectbox(
            "Filter",
            ["All", "🔴 Below 200 SMA", "🟢 Above 200 SMA"],
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
    
    if filter_option == "🔴 Below 200 SMA":
        filtered_df = filtered_df[filtered_df['Distance %'] < 0]
        if use_filters:
            filtered_df = filtered_df[filtered_df['Duration'] >= min_days_filter]
            
    elif filter_option == "🟢 Above 200 SMA":
        # If noise filter is on, we might want to Include "Below" stocks that are < 14 days?
        # No, that's confusing.
        filtered_df = filtered_df[filtered_df['Distance %'] >= 0]
    
    # Apply sort
    if sort_option == "Distance % (Ascending)":
        filtered_df = filtered_df.sort_values('Distance %', ascending=True)
    elif sort_option == "Distance % (Descending)":
        filtered_df = filtered_df.sort_values('Distance %', ascending=False)
    elif sort_option == "Max Depth / Peak (Ascending)":
        filtered_df = filtered_df.sort_values('Max Depth / Peak', ascending=True) # Ascending: Most Negative (Deepest) first
    elif sort_option == "Max Depth / Peak (Descending)":
         filtered_df = filtered_df.sort_values('Max Depth / Peak', ascending=False) # Descending: Most Positive (Highest Peak) first
    elif sort_option == "Weight (Descending)":
        filtered_df = filtered_df.sort_values('Weight', ascending=False)
    elif sort_option == "Ticker (A-Z)":
        filtered_df = filtered_df.sort_values('Ticker', ascending=True)
    
    # Summary stats
    below_count = len(result_df[result_df['Distance %'] < 0])
    above_count = len(result_df[result_df['Distance %'] >= 0])
    
    col_s1, col_s2, col_s3 = st.columns(3)
    col_s1.metric("🔴 Below 200 SMA", below_count)
    col_s2.metric("🟢 Above 200 SMA", above_count)
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
    
    # Display table
    st.dataframe(
        filtered_df.style.format({
            'Weight': '{:.2%}',
            'Price': '${:,.2f}',
            'SMA_200': '${:,.2f}',
            'Distance %': '{:+.2f}%',
            'Max Depth / Peak': '{:+.2f}%'
        }).applymap(
            lambda x: 'color: #ff4b4b' if '🔴' in str(x) else 'color: #21c354',
            subset=['Status']
        ),
        use_container_width=True,
        hide_index=True,
        height=600
    )
    
    st.caption("💡 Data cached for 15 minutes. Refresh the page to update.")

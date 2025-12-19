import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from app.services import testfol_api as api
from app.common.utils import color_return

# Define Asset Class Universe (Proxies)
# Using Testfol Simulated Tickers for maximum history
ASSET_CLASSES = {
    "US Large Cap": "SPYSIM",
    "US Small Cap": "VBSIM",     # 1926+
    "Int'l Stocks": "VXUSSIM",   # 1970+
    "Real Estate": "REITSIM",    # 1993+
    "Gold": "GLDSIM",            # 1968+
    "Long Treas": "TLTSIM",      # 1962+
    "Interm Treas": "IEFSIM",    # 1962+ (Replaces Agg Bond)
    "Commodities": "GSGSIM",     # 1979+
    "Cash": "CASHX"              # 1885+ (Bills)
}

# Assign consistent colors for the periodic table
ASSET_COLORS = {
    "SPYSIM": "#4A90E2",  # Blue (Large Cap)
    "VBSIM": "#F5A623",   # Orange (Small Cap)
    "VXUSSIM": "#7ED321", # Green (Int'l)
    "REITSIM": "#BD10E0", # Purple (Real Estate)
    "GLDSIM": "#FFD700",  # Gold
    "TLTSIM": "#D0021B",  # Red (Long Bond)
    "IEFSIM": "#9B9B9B",  # Grey (Interm Bond)
    "GSGSIM": "#8B572A",  # Brown (Commodities)
    "CASHX": "#50E3C2"    # Teal (Cash)
}

@st.cache_data(show_spinner="Fetching Asset History...", ttl=3600)
def cached_fetch_asset_history(ticker):
    """
    Fetches long-term history for a single asset using Testfol API.
    """
    try:
        # Request full history
        end_date = pd.Timestamp.now().strftime("%Y-%m-%d")
        
        # We rely on Testfol API's behavior: asking for 100% of a ticker
        # returns its price history in the 'charts' response.
        series, _, _ = api.fetch_backtest(
            start_date="1900-01-01",
            end_date=end_date,
            start_val=10000,
            cashflow=0,
            cashfreq="Monthly",
            rolling=1,
            invest_div=True, # Total Return
            rebalance="Yearly",
            allocation={ticker: 100.0},
            return_raw=False
        )
        return series
    except Exception as e:
        print(f"Error fetching {ticker}: {e}")
        return pd.Series(dtype=float)

def render_asset_explorer():
    """
    Renders the Periodic Table of Asset Classes view.
    """
    st.subheader("🧩 Asset Class Explorer (Periodic Table)")
    st.markdown("Annual performance ranking of major asset classes.")
    
    # 1. Fetch Data
    price_data = {}
    
    # 1. Fetch Data
    price_data = {}
    
    # Custom Asset Selection (Interactive Legend)
    # Use session state to track exclusions
    if "ae_excluded_assets" not in st.session_state:
        st.session_state.ae_excluded_assets = []
        
    all_assets = list(ASSET_CLASSES.keys())
    
    # Helper for Color-coded Emoji
    def get_color_emoji(ticker):
        # Approximate mapping or just use a generic circle
        # We can try to use a specific circle color if we had a map, 
        # but standardized circles are limited (🔴 🟠 🟡 🟢 🔵 🟣 ⚫ ⚪ 🟤)
        c = ASSET_COLORS.get(ticker, "").upper()
        if "4A90E2" in c: return "🔵" # Blue
        if "F5A623" in c: return "🟠" # Orange
        if "7ED321" in c: return "🟢" # Green
        if "BD10E0" in c: return "🟣" # Purple (Estate)
        if "FFD700" in c: return "🟡" # Gold
        if "D0021B" in c: return "🔴" # Red
        if "9B9B9B" in c: return "🔘" # Grey
        if "8B572A" in c: return "🟤" # Brown
        if "50E3C2" in c: return "🟢" # Teal (use Green or Blue?)
        return "⚪"

    st.markdown("##### Filter Assets (Click to Toggle)")
    
    # Create rows of buttons (Interactive Key)
    # Split into 2 rows to avoid squashing when sidebar is open
    # 5 items first row, 4 items second row
    
    row1_assets = all_assets[:5]
    row2_assets = all_assets[5:]
    
    # Row 1
    cols1 = st.columns(len(row1_assets))
    for i, name in enumerate(row1_assets):
        ticker = ASSET_CLASSES[name]
        is_excluded = name in st.session_state.ae_excluded_assets
        emoji = get_color_emoji(ticker)
        label = f"{emoji} {name}"
        btn_type = "secondary" if is_excluded else "primary"
        with cols1[i]:
            if st.button(label, key=f"btn_{ticker}", type=btn_type, use_container_width=True):
                if is_excluded:
                    st.session_state.ae_excluded_assets.remove(name)
                else:
                    st.session_state.ae_excluded_assets.append(name)
                st.rerun()

    # Row 2
    cols2 = st.columns(len(row2_assets))
    for i, name in enumerate(row2_assets):
        ticker = ASSET_CLASSES[name]
        is_excluded = name in st.session_state.ae_excluded_assets
        emoji = get_color_emoji(ticker)
        label = f"{emoji} {name}"
        btn_type = "secondary" if is_excluded else "primary"
        with cols2[i]:
            if st.button(label, key=f"btn_{ticker}", type=btn_type, use_container_width=True):
                if is_excluded:
                    st.session_state.ae_excluded_assets.remove(name)
                else:
                    st.session_state.ae_excluded_assets.append(name)
                st.rerun()

    # Determine Selected
    selected_assets = [a for a in all_assets if a not in st.session_state.ae_excluded_assets]
            
    if not selected_assets:
        st.warning("Please select at least one asset class.")
        return
    
    with st.spinner("Fetching Asset Class History..."):
        # Parallel fetch could be faster, but cached functions handle it okay
        for name in selected_assets:
            ticker = ASSET_CLASSES[name]
            series = cached_fetch_asset_history(ticker)
            if not series.empty:
                price_data[ticker] = series
    
    if not price_data:
        st.error("Failed to fetch asset data. Please check API connection.")
        return

    # 2. Combine & Clean
    # Use Names for columns instead of Tickers
    df_prices = pd.DataFrame()
    name_to_color = {}
    
    for name in selected_assets:
        ticker = ASSET_CLASSES[name]
        if ticker in price_data:
            df_prices[name] = price_data[ticker]
            name_to_color[name] = ASSET_COLORS.get(ticker, "#333")
            
    df_prices = df_prices.ffill().dropna() # intersection
    
    if df_prices.empty:
        st.warning("No overlapping data found for selected assets.")
        return
        
    # Show Data Range
    common_start_year = df_prices.index[0].year
    
    # Metric and Year Filter
    c1, c2 = st.columns([1, 3])
    with c1:
        st.metric("Earliest Common Year", f"{common_start_year}")
    
    # 3. Calculate Annual Returns
    df_annual = df_prices.resample("YE").last().pct_change().dropna()
    years = df_annual.index.year.tolist()
    
    # Filter Years (Optional UI)
    min_year, max_year = min(years), max(years)
    
    with c2:
        selected_years = st.slider("Year Range", min_value=min_year, max_value=max_year, value=(max(min_year, max_year-14), max_year))
    
    # Filter DataFrame
    mask = (df_annual.index.year >= selected_years[0]) & (df_annual.index.year <= selected_years[1])
    df_annual_filtered = df_annual[mask]
    
    # 4. Build periodic table structure
    # Rows: Rank 1 to N
    # Columns: Years + "Period"
    
    names = list(df_annual_filtered.columns)
    n_assets = len(names)
    final_years = df_annual_filtered.index.year.tolist()
    
    # Dictionary to hold the data: {Year: [Ordered List of (Name, Ret)]}
    ranked_data = {}
    
    for y in final_years:
        # Get returns for this year
        year_rets = df_annual_filtered.loc[df_annual_filtered.index.year == y].iloc[0]
        # Sort descending
        sorted_rets = year_rets.sort_values(ascending=False)
        ranked_data[y] = [(n, v) for n, v in sorted_rets.items()]
        
    # Calculate Period Stats (CAGR) for the Final Column
    # Slice prices to the start of the first selected year vs end of last selected year
    # Start Date: Dec 31 of (SelectedYear - 1). 
    
    # Re-slice original prices to cover the range
    start_y = selected_years[0]
    end_y = selected_years[1]
    
    # Dynamic Column Name
    period_label = "Period CAGR"
    
    # Get price at end of (start_y - 1). If start_y is first year, we need inception?
    # df_prices is "YE" resampled? No it's daily.
    
    # Resample prices to YE to match logic
    df_prices_ye = df_prices.resample("YE").last()
    
    # We need the price at the END of the year BEFORE the start year.
    # e.g. for 2011 return, we need Dec 2010 price.
    
    ranked_data[period_label] = []
    
    try:
        # Check if we have data for year before start
        prev_year = start_y - 1
        # Indices of YE prices
        ye_years = df_prices_ye.index.year
        
        if prev_year in ye_years and end_y in ye_years:
            start_prices = df_prices_ye.loc[df_prices_ye.index.year == prev_year].iloc[0]
            end_prices = df_prices_ye.loc[df_prices_ye.index.year == end_y].iloc[0]
            
            n_years = end_y - prev_year
            total_ret = (end_prices / start_prices) - 1
            cagrs = (1 + total_ret) ** (1/n_years) - 1
            
            # Sort
            sorted_cagr = cagrs.sort_values(ascending=False)
            ranked_data[period_label] = [(n, v) for n, v in sorted_cagr.items()]
        else:
            # Fallback: Just average of annual returns (Arithmetic, not Geo, but ok for display backup)
            # Better: Cumulative of the annuals we have
            if not df_annual_filtered.empty:
               cum_ret = (1 + df_annual_filtered).prod() - 1
               n_years = len(final_years)
               if n_years > 0:
                   geo_mean = (1 + cum_ret) ** (1/n_years) - 1
                   sorted_geo = geo_mean.sort_values(ascending=False)
                   ranked_data[period_label] = [(n, v) for n, v in sorted_geo.items()]
            
    except Exception as e:
        print(f"Error calc period: {e}")
        ranked_data[period_label] = []

        
    # Create Display DataFrame
    display_index = range(1, n_assets + 1)
    
    # Columns: Years ... "Period"
    cols = final_years + [period_label]
    
    df_display = pd.DataFrame(index=display_index, columns=cols)
    # df_colors logic... use name_to_color directly in styler
    
    for col in cols:
        if col not in ranked_data: continue
        
        rank_list = ranked_data[col]
        col_data = [] # Text
        
        for i, (name, ret) in enumerate(rank_list):
            # Format: 'Gold +25.4%'
            col_data.append(f"{name} {ret:+.1%}")
            
        while len(col_data) < n_assets:
            col_data.append("")
            
        df_display[col] = col_data

    # 5. Render with Styler
    def color_cells(val):
        # We need to find the Name to parse color.
        if not isinstance(val, str) or not val:
            return ""
            
        # Find matching name
        found_name = None
        for name in name_to_color.keys():
            if val.startswith(name):
                found_name = name
                break
                
        bg_color = name_to_color.get(found_name, "#333333")
        
        # Text Color: Black or White depending on lightness?
        text_color = "white"
        if found_name == "Gold" or found_name == "Cash" or found_name == "US Small Cap":
            text_color = "black"
        
        return f'background-color: {bg_color}; color: {text_color}; border: 1px solid #222; font-weight: bold;'

    st.markdown("###  Periodic Table of Investment Returns")
    
    st.dataframe(
        df_display.style.applymap(color_cells),
        use_container_width=True,
        height=(n_assets + 1) * 35 + 50 
    )

    
    # Legend (Static Bottom Key)
    st.markdown("#### Key (Ticker Mapping)")
    
    # User prefers single row for the key
    l_cols = st.columns(len(all_assets))
    
    for i, name in enumerate(all_assets):
        ticker = ASSET_CLASSES[name]
        color = ASSET_COLORS.get(ticker, "#333")
        text_color = "white"
        if name == "Gold" or name == "Cash" or name == "US Small Cap": # Use name for simpler check
             text_color = "black"
             
        with l_cols[i]:
             st.markdown(
                f"<div style='background-color: {color}; color: {text_color}; padding: 8px; border-radius: 4px; text-align: center; font-size: 0.8em; font-weight: bold;'>{ticker}<br><span style='font-size: 0.9em; font-weight: normal;'>{name}</span></div>", 
                unsafe_allow_html=True
            )

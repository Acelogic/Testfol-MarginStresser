"""
X-Ray Engine
Expands ETF holdings recursively and aggregates them with direct stock holdings.
"""

import pandas as pd
import logging
import sys
import os
import re

# Ensure we can import from data/etf_xray/src
sys.path.append(os.path.join(os.getcwd(), "data/etf_xray/src"))
try:
    import etf_holdings_fetcher
except ImportError:
    # Handle if run from different CWD
    sys.path.append(os.path.join(os.path.dirname(__file__), "../../data/etf_xray/src"))
    import etf_holdings_fetcher

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

# Load Ticker to Name mapping from SEC data for better aggregation
TICKER_TO_NAME = {}
NAME_TO_TICKER = {}
try:
    # Updated path after separate module refactor
    # Assets are now in data/ndx_simulation/data/assets
    assets_dir = os.path.join(os.getcwd(), "data/ndx_simulation/data/assets")
    
    # Try alternate location if cwd is different (e.g. running from test suite)
    if not os.path.exists(assets_dir):
         assets_dir = os.path.join(os.path.dirname(__file__), "../../data/ndx_simulation/data/assets")
         
    json_path = os.path.join(assets_dir, "company_tickers.json")
    if os.path.exists(json_path):
        import json
        with open(json_path, 'r') as f:
            data = json.load(f)
            # Format: {"0": {"cik_str": 320193, "ticker": "AAPL", "title": "Apple Inc."}, ...}
            for v in data.values():
                title = v['title']
                ticker = v['ticker']
                TICKER_TO_NAME[ticker] = title
                
                # Build Name -> Ticker
                NAME_TO_TICKER[title.upper()] = ticker
                
                # Also add a version without common suffixes for better matching
                # Strip commas first (e.g. "Meta Platforms, Inc." -> "Meta Platforms Inc")
                clean_title_raw = title.upper().replace(',', '')
                clean_title = re.sub(r'\s+(INC\.?|CORP\.?|LTD\.?|CO\.?|PLC|S\.A\.?|N\.V\.?|AG)\s*$', '', clean_title_raw, flags=re.IGNORECASE).strip()
                if clean_title and clean_title not in NAME_TO_TICKER:
                    NAME_TO_TICKER[clean_title] = ticker
                    
        logging.info(f"Loaded {len(TICKER_TO_NAME)} tickers and {len(NAME_TO_TICKER)} name mappings for X-Ray.")
except Exception as e:
    logging.warning(f"Could not load ticker mappings: {e}")

# Map for merging share classes or equivalent tickers
TICKER_ALIASES = {
    'GOOGL': 'GOOG',  # Alphabet Class A -> Class C (canonical)
    # Add others here if needed, e.g. BRK.B -> BRK.A ?
}

# Static leverage map for known leveraged ETFs
STATIC_LEVERAGE = {
    # 3x Bull
    "UPRO": 3.0, "SPEU": 3.0, 
    "TQQQ": 3.0, "TECL": 3.0, "FNGU": 3.0,
    "SOXL": 3.0, "FAS": 3.0, "DPST": 3.0,
    "LABU": 3.0, "NAIL": 3.0, "WANT": 3.0,
    "BULZ": 3.0, "CURE": 3.0, "HIBL": 3.0,
    "MIDU": 3.0, "UDOW": 3.0, "URTY": 3.0,
    "TMF": 3.0, "TYD": 3.0,
    
    # 2x Bull
    "SSO": 2.0, "QLD": 2.0, "UWM": 2.0,
    "DDM": 2.0, "MVV": 2.0, "DIG": 2.0,
    "UYG": 2.0, "ROM": 2.0, "URE": 2.0,
    "USD": 2.0, "NUGT": 2.0, "JNUG": 2.0,
    "GUSH": 2.0, "ERX": 2.0, "UBT": 2.0,
    "UST": 2.0,
}

def parse_leverage(ticker):
    """Extract leverage factor from Testfol ticker string (e.g. ?L=2) or known static map."""
    clean_ticker = ticker.split('?')[0].split('@')[0].upper()
    if clean_ticker in STATIC_LEVERAGE:
        return STATIC_LEVERAGE[clean_ticker]
    if '?L=' in ticker:
        try:
            return float(ticker.split('?L=')[1].split('&')[0])
        except: pass
    if '-' in ticker:
        # Check for 3X, 2X, etc suffixes
        parts = ticker.split('-')
        last = parts[-1].upper()
        if last.endswith('X'):
            try:
                return float(last[:-1])
            except: pass
    return 1.0


def compute_xray(portfolio_dict, depth=0, max_depth=2):
    """
    portfolio_dict: {ticker: weight, ...} e.g., {"QQQ": 0.5, "AAPL": 0.5}
    depth: current recursion depth
    max_depth: limit recursion to avoid infinite loops or excessive API calls
    
    Returns: DataFrame with [Ticker, Name, Weight, Source]
    """
    if depth > max_depth:
        return pd.DataFrame()

    all_holdings = []
    
    # Helper to process a single holding (for threading)
    def process_holding(item):
        ticker, port_weight = item
        local_holdings = []
        
        if port_weight <= 0:
            return []
            
        logging.info(f"Processing {ticker} at depth {depth}...")
        
        # Parse Leverage
        leverage = parse_leverage(ticker)
        effective_port_weight = port_weight * leverage
        
        # Track margin debt if leverage > 1
        if leverage > 1.0:
            margin_debt = port_weight - effective_port_weight
            local_holdings.append({
                'Ticker': 'CASH',
                'Name': 'Margin Debt / Leverage Cost',
                'Weight': margin_debt,
                'Source': ticker
            })
            
        # Try to fetch ETF holdings
        etf_df = etf_holdings_fetcher.get_etf_holdings(ticker)
        
        if etf_df is not None and not etf_df.empty:
            # It's an ETF, process its holdings
            for _, row in etf_df.iterrows():
                holding_name = row['name']
                holding_ticker = row.get('ticker', '')
                holding_weight = row['weight']
                
                # Absolute weight in the total portfolio
                abs_weight = effective_port_weight * holding_weight
                
                local_holdings.append({
                    'Ticker': holding_ticker,
                    'Name': holding_name,
                    'Weight': abs_weight,
                    'Source': ticker
                })
        else:
            # It's a direct holding
            # Resolve name from ticker if possible
            # Strip modifiers for name lookup
            clean_ticker = ticker.split('?')[0].split('@')[0].upper()
            
            # Use clean ticker for name lookup
            name = TICKER_TO_NAME.get(clean_ticker, clean_ticker)
            
            # If we didn't find a name, try to fetch it if it's a valid ticker?
            # For now, just rely on clean ticker being better than "AVGO?L=2"
            
            local_holdings.append({
                'Ticker': clean_ticker,
                'Name': name,
                'Weight': effective_port_weight,
                'Source': 'Direct'
            })
        return local_holdings

    # Use ThreadPool to process holdings in parallel
    # Max workers = 10 to allow decent parallelism for network properties
    import concurrent.futures
    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        results = executor.map(process_holding, portfolio_dict.items())
        
    for res in results:
        all_holdings.extend(res)
            
    if not all_holdings:
        return pd.DataFrame()
        
    df = pd.DataFrame(all_holdings)
    
    # Normalize names for better aggregation (handle variations like 'Apple Inc.' vs 'APPLE INC')
    df['NormalizedName'] = df['Name'].str.upper().str.strip()
    df['NormalizedName'] = df['NormalizedName'].str.replace(r'[,\.\s]+', ' ', regex=True)
    # Remove common suffixes and specific patterns like "(b)" or "(c)" or "(a)" often seen in filings
    df['NormalizedName'] = df['NormalizedName'].str.replace(r'\s*\([A-Z]\)$', '', regex=True)
    df['NormalizedName'] = df['NormalizedName'].str.replace(r'\s+(INC|CORP|LTD|PLC|CO|CLASS [A-Z])$', '', regex=True)
    df['NormalizedName'] = df['NormalizedName'].str.strip()
    
    # First try to aggregate by CUSIP if available
    # Actually, Ticker is better if we have it. The get_ticker enrichment attempts to find Tickers for everything.
    # So we don't strictly need these pre-calculation steps for AggKey anymore since we define AggKey below.
    # But let's leave CUSIP logic as a fallback to NormalizedName if needed, but Ticker is supreme.
    pass

    # Enrich missing tickers using NAME_TO_TICKER
    def get_ticker(row):
        if row['Ticker']: return row['Ticker']
        
        # Try exact match on NormalizedName
        if row['NormalizedName'] in NAME_TO_TICKER:
            return NAME_TO_TICKER[row['NormalizedName']]
        
        # Try cleaning the name further locally if needed (though we cleaned keys in map)
        # Check against original name upper just in case
        name_upper = str(row['Name']).upper().strip()
        if name_upper in NAME_TO_TICKER:
            return NAME_TO_TICKER[name_upper]

        return ''

    df['Ticker'] = df.apply(get_ticker, axis=1)

    # Apply Ticker Aliases (e.g. GOOGL -> GOOG) to canonicalize
    df['Ticker'] = df['Ticker'].replace(TICKER_ALIASES)

    # Use Ticker as the primary aggregation key if available, otherwise NormalizedName
    # This ensures "AMZN" from source A matches "AMZN" from source B even if names differ slightly
    df['AggKey'] = df['Ticker'].where(df['Ticker'] != '', df['NormalizedName'])
    
    # Aggregate by the key
    summary = df.groupby('AggKey').agg({
        'Name': 'first',  # Keep original name for display
        'Weight': 'sum',
        'Ticker': 'first', # Keep the ticker
        'Source': lambda x: ', '.join(sorted(set(x)))
    }).reset_index(drop=True)

    
    # Sort by weight descending
    summary = summary.sort_values('Weight', ascending=False).reset_index(drop=True)
    
    # Log aggregation summary
    total_weight = summary['Weight'].sum()
    logging.info(f"X-Ray: {len(summary)} unique holdings, total weight: {total_weight:.2%}")
    
    return summary

if __name__ == "__main__":
    # Test
    test_portfolio = {"QQQ": 0.5, "AAPL": 0.5}
    print(f"X-Ray result for {test_portfolio}:")
    result = compute_xray(test_portfolio)
    print(result.head(20))
    print(f"Total Portfolio Weight: {result['Weight'].sum():.2%}")

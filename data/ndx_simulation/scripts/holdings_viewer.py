import pandas as pd
import argparse
import sys
import os
from datetime import datetime
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), "../src"))
import config

# Configuration imported from config.py
WEIGHTS_FILE = config.WEIGHTS_FILE
MEGA_TARGET_THRESHOLD = config.MEGA1_TARGET_THRESHOLD
MEGA_BUFFER_THRESHOLD = config.MEGA1_BUFFER_THRESHOLD
SINGLE_STOCK_CAP = config.MEGA1_SINGLE_STOCK_CAP

# Mega 2.0 Settings
MEGA2_TARGET_THRESHOLD = config.MEGA2_TARGET_THRESHOLD
SINGLE_STOCK_CAP2 = config.MEGA2_SINGLE_STOCK_CAP
MIN_CONSTITUENTS2 = config.MEGA2_MIN_CONSTITUENTS

def apply_caps(w_series, cap, total_target=1.0):
    """
    Apply capping rules iteratively.
    Re-normalizes weights to sum to total_target, then caps any single weight > cap.
    Surplus is redistributed proportionally to uncapped members.
    """
    w = w_series.copy()
    if w.sum() == 0: return w
    w = (w / w.sum()) * total_target
    
    for _ in range(10):
        excess = w[w > cap]
        if excess.empty: break
        
        surplus = (excess - cap).sum()
        w[w > cap] = cap
        
        others = w[w < cap]
        if others.empty: break
        
        # Redistribute surplus
        w[w < cap] = others + (surplus * others / others.sum())
        
    return w

def get_mega_holdings(df_date):
    """
    Derive NDX Mega holdings for a specific date (Strict Selection).
    """
    df_date = df_date.sort_values(by='Weight', ascending=False).copy()
    df_date['CumWeight'] = df_date['Weight'].cumsum()
    
    selected_tickers = []
    curr_sum = 0.0
    for tick, w, mapped in zip(df_date['Ticker'], df_date['Weight'], df_date['IsMapped']):
        if curr_sum + w <= MEGA_TARGET_THRESHOLD + 0.01:
            if mapped:
                selected_tickers.append(tick)
            curr_sum += w
        else:
            break
            
    mega_df = df_date[df_date['Ticker'].isin(selected_tickers)].copy()
    if mega_df.empty: return pd.DataFrame()
        
    mega_df['OriginalWeight'] = mega_df['Weight']
    w_series = mega_df.set_index('Ticker')['OriginalWeight']
    final_weights = apply_caps(w_series, SINGLE_STOCK_CAP)
    
    mega_df['FinalWeight'] = mega_df['Ticker'].map(final_weights)
    return mega_df.sort_values('FinalWeight', ascending=False)[['Ticker', 'Name', 'FinalWeight', 'OriginalWeight']]

def get_mega2_holdings(df_date):
    """
    Derive NDX Mega 2.0 holdings for a specific date (Strict Selection).
    """
    df_date = df_date.sort_values(by='Weight', ascending=False).copy()
    df_date['CumWeight'] = df_date['Weight'].cumsum()
    
    # 1. Standard Selection (40% Target)
    standard_tickers = []
    curr_sum = 0.0
    for tick, w, mapped in zip(df_date['Ticker'], df_date['Weight'], df_date['IsMapped']):
        if curr_sum + w <= MEGA2_TARGET_THRESHOLD + 0.01:
            if mapped:
                standard_tickers.append(tick)
            curr_sum += w
        else:
            break
            
    # 2. Minimum Security Rule (9 stocks)
    selected_tickers = standard_tickers.copy()
    is_min_security_triggered = False
    
    if len(selected_tickers) < MIN_CONSTITUENTS2:
        is_min_security_triggered = True
        valid_mapped_all = df_date[df_date['IsMapped'] == True]
        remaining = valid_mapped_all[~valid_mapped_all['Ticker'].isin(selected_tickers)]
        needed = MIN_CONSTITUENTS2 - len(selected_tickers)
        if not remaining.empty:
            fillers = remaining.head(needed)['Ticker'].tolist()
            selected_tickers.extend(fillers)

    mega_df = df_date[df_date['Ticker'].isin(selected_tickers)].copy()
    if mega_df.empty: return pd.DataFrame()

    mega_df['OriginalWeight'] = mega_df['Weight']

    # 3. Weighting Logic
    if not is_min_security_triggered:
        w_series = mega_df.set_index('Ticker')['OriginalWeight']
        final_weights = apply_caps(w_series, SINGLE_STOCK_CAP2)
    else:
        # Standard group gets 99%, Fillers get 1% total
        standard_subset = mega_df[mega_df['Ticker'].isin(standard_tickers)]
        filler_subset = mega_df[~mega_df['Ticker'].isin(standard_tickers)]
        
        if not standard_subset.empty:
            w_standard = apply_caps(standard_subset.set_index('Ticker')['OriginalWeight'], SINGLE_STOCK_CAP2, total_target=0.99)
        else:
            w_standard = pd.Series()
            
        if not filler_subset.empty:
            w_filler = pd.Series(0.01 / len(filler_subset), index=filler_subset['Ticker'])
        else:
            w_filler = pd.Series()
            
        final_weights = pd.concat([w_standard, w_filler])
    
    mega_df['FinalWeight'] = mega_df['Ticker'].map(final_weights)
    return mega_df.sort_values('FinalWeight', ascending=False)[['Ticker', 'Name', 'FinalWeight', 'OriginalWeight']]

CONST_FILE_M1 = os.path.join(config.RESULTS_DIR, "ndx_mega_constituents.csv")
CONST_FILE_M2 = os.path.join(config.RESULTS_DIR, "ndx_mega2_constituents.csv")

def get_holdings_from_history(history_df, target_date):
    """
    Retrieves holdings from backtest history if available.
    """
    target_str = target_date.strftime('%Y-%m-%d')
    match = history_df[history_df['Date'] == target_str]
    
    if match.empty:
        return None
        
    row = match.iloc[0]
    if 'Tickers' not in row or pd.isna(row['Tickers']):
        return None
        
    tickers = row['Tickers'].split('|')
    weights = [float(w) for w in row['Weights'].split('|')]
    
    df = pd.DataFrame({'Ticker': tickers, 'FinalWeight': weights})
    return df

def main():
    print("Loading data...")
    try:
        df = pd.read_csv(WEIGHTS_FILE)
        df['Date'] = pd.to_datetime(df['Date'])
        
        # Filter future dates (Defense in depth)
        today = pd.Timestamp.now().normalize()
        df = df[df['Date'] <= today]
    except FileNotFoundError:
        print(f"Error: {WEIGHTS_FILE} not found. Run reconstruct_weights.py first.")
        sys.exit(1)
        
    # Load Histories
    hist_m1 = pd.DataFrame()
    hist_m2 = pd.DataFrame()
    try:
        if os.path.exists(CONST_FILE_M1):
            hist_m1 = pd.read_csv(CONST_FILE_M1)
    except: pass
    
    try:
        if os.path.exists(CONST_FILE_M2):
            hist_m2 = pd.read_csv(CONST_FILE_M2)
    except: pass
        
    available_dates = sorted(df['Date'].unique())
    min_year = available_dates[0].year
    max_year = available_dates[-1].year
    
    print(f"\nData available from {min_year} to {max_year}.")
    if not hist_m1.empty: print(f"Loaded Mega 1.0 History ({len(hist_m1)} periods)")
    if not hist_m2.empty: print(f"Loaded Mega 2.0 History ({len(hist_m2)} periods)")
    
    while True:
        print("\n" + "="*50)
        user_input = input(f"Enter Year (e.g. 2005), or 'q' to quit: ").strip().lower()
        if user_input == 'q': break
            
        try:
            year = int(user_input)
        except ValueError:
            print("Invalid year."); continue
            
        year_dates = [d for d in available_dates if d.year == year]
        if not year_dates:
            print(f"No data found for {year}."); continue
            
        print(f"\nAvailable Quarters for {year}:")
        for i, d in enumerate(year_dates):
            print(f"{i+1}. {d.date()}")
            
        q_idx = input("\nSelect Quarter (1-4) or Enter for last available: ").strip()
        target_date = year_dates[int(q_idx)-1] if q_idx and q_idx.isdigit() and int(q_idx) <= len(year_dates) else year_dates[-1]
            
        print(f"\nAnalyzing funds for {target_date.date()}...")
        full_slice = df[df['Date'] == target_date].sort_values('Weight', ascending=False)
        
        # 1. Full NDX Top 25
        print(f"\n--- NDX Top 25 Holdings ({target_date.date()}) ---")
        print(f"{'Ticker':<10} {'Name':<40} {'Weight':<10}")
        print("-" * 65)
        for _, row in full_slice.head(25).iterrows():
            print(f"{row['Ticker']:<10} {row['Name'][:38]:<40} {row['Weight']:.2%}")
            
        # 2. NDX Mega 1.0
        # Try History First
        m1 = get_holdings_from_history(hist_m1, target_date)
        source_m1 = "Backtest History (Accurate)"
        
        if m1 is None:
            m1 = get_mega_holdings(full_slice)
            source_m1 = "Strict Calculation (Approx)"
        else:
            # Join Name
            m1 = m1.merge(full_slice[['Ticker', 'Name']], on='Ticker', how='left')
            m1['Name'] = m1['Name'].fillna('Unknown')
            
        print(f"\n--- NDX Mega 1.0 Holdings ({target_date.date()}) ---")
        print(f"Source: {source_m1}")
        print(f"Constituents: {len(m1)}")
        print(f"{'Ticker':<10} {'Name':<40} {'Weight':<10}")
        print("-" * 75)
        for _, row in m1.sort_values('FinalWeight', ascending=False).iterrows():
            print(f"{row['Ticker']:<10} {row['Name'][:38]:<40} {row['FinalWeight']:.2%}")
        print("-" * 75)
        
        # 3. NDX Mega 2.0
        m2 = get_holdings_from_history(hist_m2, target_date)
        source_m2 = "Backtest History (Accurate)"
        
        if m2 is None:
            m2 = get_mega2_holdings(full_slice)
            source_m2 = "Strict Calculation (Approx)"
        else:
             m2 = m2.merge(full_slice[['Ticker', 'Name']], on='Ticker', how='left')
             m2['Name'] = m2['Name'].fillna('Unknown')

        print(f"\n--- NDX Mega 2.0 Holdings ({target_date.date()}) ---")
        print(f"Source: {source_m2}")
        print(f"Constituents: {len(m2)}")
        print(f"{'Ticker':<10} {'Name':<40} {'Weight':<10}")
        print("-" * 75)
        for _, row in m2.sort_values('FinalWeight', ascending=False).iterrows():
            print(f"{row['Ticker']:<10} {row['Name'][:38]:<40} {row['FinalWeight']:.2%}")
        print("-" * 75)

if __name__ == "__main__":
    main()

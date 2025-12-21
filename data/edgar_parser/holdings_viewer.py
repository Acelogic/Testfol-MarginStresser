import pandas as pd
import argparse
import sys
from datetime import datetime

# Configuration matching backtest_ndx_mega.py
WEIGHTS_FILE = "nasdaq_quarterly_weights.csv"
MEGA_TARGET_THRESHOLD = 0.47
MEGA_BUFFER_THRESHOLD = 0.50
SINGLE_STOCK_CAP = 0.35

# Configuration matching backtest_ndx_mega2.py
MEGA2_TARGET_THRESHOLD = 0.40
SINGLE_STOCK_CAP2 = 0.30
MIN_CONSTITUENTS2 = 9

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

def main():
    print("Loading data...")
    try:
        df = pd.read_csv(WEIGHTS_FILE)
        df['Date'] = pd.to_datetime(df['Date'])
    except FileNotFoundError:
        print(f"Error: {WEIGHTS_FILE} not found. Run reconstruct_weights.py first.")
        sys.exit(1)
        
    available_dates = sorted(df['Date'].unique())
    min_year = available_dates[0].year
    max_year = available_dates[-1].year
    
    print(f"\nData available from {min_year} to {max_year}.")
    
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
        m1 = get_mega_holdings(full_slice)
        print(f"\n--- NDX Mega 1.0 Holdings ({target_date.date()}) ---")
        print(f"Constituents: {len(m1)}")
        print(f"{'Ticker':<10} {'Name':<40} {'Weight':<10} {'(Orig Wgt)'}")
        print("-" * 75)
        for _, row in m1.iterrows():
            print(f"{row['Ticker']:<10} {row['Name'][:38]:<40} {row['FinalWeight']:.2%}      ({row['OriginalWeight']:.2%})")
        print("-" * 75)
        
        # 3. NDX Mega 2.0
        m2 = get_mega2_holdings(full_slice)
        print(f"\n--- NDX Mega 2.0 Holdings ({target_date.date()}) ---")
        print(f"Constituents: {len(m2)}")
        print(f"{'Ticker':<10} {'Name':<40} {'Weight':<10} {'(Orig Wgt)'}")
        print("-" * 75)
        for _, row in m2.iterrows():
            print(f"{row['Ticker']:<10} {row['Name'][:38]:<40} {row['FinalWeight']:.2%}      ({row['OriginalWeight']:.2%})")
        print("-" * 75)

if __name__ == "__main__":
    main()


if __name__ == "__main__":
    main()

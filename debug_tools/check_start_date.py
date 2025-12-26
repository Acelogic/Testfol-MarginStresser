import pandas as pd
import os

def check():
    path = "data/NDXMEGASIM.csv"
    if not os.path.exists(path):
        print(f"File not found: {path}")
        return

    df = pd.read_csv(path)
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.set_index('Date')
    
    print(f"First 5 rows of {path}:")
    print(df.head())
    
    start_date = df.index.min()
    print(f"\nRaw Start Date: {start_date}")
    
    # Simulate return calculation
    returns = df.pct_change().dropna()
    print(f"\nStart Date after pct_change().dropna(): {returns.index.min()}")
    
    # Simulate forward fill then pct_change?
    # App logic might be: prices_df.ffill().pct_change()...
    # Let's create a date range for business days
    full_idx = pd.date_range(start='2000-06-30', end='2000-09-20', freq='B')
    df_reindexed = df.reindex(full_idx)
    print("\nReindexed (Business Days):")
    print(df_reindexed.head())
    
    df_ffilled = df_reindexed.ffill()
    print("\nAfter ffill:")
    print(df_ffilled.head())
    
    returns_ffilled = df_ffilled.pct_change().dropna()
    print(f"\nStart Date after ffill().pct_change().dropna(): {returns_ffilled.index.min()}")

if __name__ == "__main__":
    check()

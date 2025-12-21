import yfinance as yf
import pandas as pd
from datetime import datetime

def check_msft():
    # Filing Data from 2000-06-30 (derived from CSV)
    # Shares: 3,347,177
    # Value: $303,128,717
    # Implied Price: 90.5625
    
    ticker = "MSFT"
    date = "2000-06-30"
    
    print(f"Fetching {ticker} data around {date}...")
    
    try:
        # Fetch 5 days around the date to handle weekends
        df = yf.download(ticker, start="2000-06-25", end="2000-07-05")
        print("\nFull Data Frame head:")
        print(df.head(10))
        
        # Check specific date
        if date in df.index:
            row = df.loc[date]
            print(f"\nData for {date}:")
            print(row)
            
            close = row['Close']
            adj_close = row['Adj Close']
            
            print(f"\nComparisons:")
            print(f"Filing Implied Price: 90.56")
            print(f"YF Close: {close}")
            print(f"YF Adj Close: {adj_close}")
            
            # Check splits
            splits = yf.Ticker(ticker).splits
            print(f"\nSplits since 2000:")
            print(splits[splits.index > "2000-01-01"])
            
        else:
            print(f"Date {date} not found in index.")
            
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        tickers = sys.argv[1:]
    else:
        tickers = ["MSFT", "BIIB", "MXIM", "YHOO"]
        
    for ticker in tickers:
        print(f"\nChecking {ticker}...")
        try:
            df = yf.download(ticker, start="2005-01-01", end="2006-01-01", progress=False)
            if not df.empty:
                print(f"  Found data: {len(df)} rows. First: {df.index[0].date()}")
            else:
                print(f"  No data found.")
        except Exception as e:
            print(f"  Error: {e}")


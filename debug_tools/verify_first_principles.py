import yfinance as yf
import pandas as pd
import numpy as np

def calculate_cagr(start_val, end_val, days):
    years = days / 365.25
    if years <= 0: return 0.0
    return ((end_val / start_val) ** (1 / years) - 1) * 100

def run_verification():
    print("="*60)
    print("🧪 INDEPENDENT FIRST-PRINCIPLES VERIFICATION")
    print("="*60)
    print("Goal: Calculate 2x QQQ CAGR with DCA from scratch.")
    print("Parameters:")
    print("  - Ticker: QQQ")
    print("  - Start: 2000-09-15")
    print("  - End:   2025-12-23")
    print("  - Start Val: $10,000")
    print("  - DCA: $1,000 / month")
    print("  - Leverage: 2.0x")
    print("  - Cost of Debt: 4.0% (Annual Est)")
    print("  - Exp Ratio: 0.95% (ProShares QID/QLD avg) or 0.0%")
    print("="*60)

    # 1. Download Data
    print("\n1. Downloading QQQ Data...")
    df = yf.download("QQQ", start="2000-09-15", end="2025-12-23", progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.droplevel(1)
    
    # Calculate Daily Returns
    if 'Adj Close' in df.columns:
        col = 'Adj Close'
    elif 'Close' in df.columns:
        col = 'Close'
    else:
        print(f"❌ Error: Columns found: {df.columns}")
        return

    df['Ret'] = df[col].pct_change().fillna(0.0)
    
    # 2. Simulation Loop
    print("2. Running Daily Loop...")
    
    cash = 0.0
    equity = 10000.0 # Start with $10k invested
    
    # Leverage Parameters
    L = 2.0
    
    # Cost of borrowing (Daily)
    # R_lev = R_index * L - (L-1)*Cost
    # Let's use a fixed cost for simplicity of verification
    COST_ANNUAL = 0.05 # 5% total cost (FFR + Spread)
    COST_DAILY = COST_ANNUAL / 252.0
    
    current_month = df.index[0].month
    
    # Correct Way: Daily Return is applied to the PORTFOLIO VALUE
    # R_port = (R_index * L) - Cost
    
    port_vals = []
    
    for date, row in df.iterrows():
        # 1. Apply Daily Return
        r_asset = row['Ret']
        
        # Leveraged Return Formula
        # R_lev = (R_asset * L) - ((L-1) * COST_DAILY)
        r_lev = (r_asset * L) - ((L-1) * COST_DAILY)
        
        # Apply to Equity
        equity = equity * (1 + r_lev)
        
        # 2. Handle DCA (First of Month)
        if date.month != current_month:
            equity += 1000.0
            current_month = date.month
            
        port_vals.append(equity)
        
    final_val = equity
    total_invested = 10000.0 + (len(df.resample('ME')) * 1000.0)
    
    # 3. Calculate CAGR
    days = (df.index[-1] - df.index[0]).days
    cagr = calculate_cagr(10000.0, final_val, days) # Note: CAGR formula for DCA is complex (MWRR), but TWR is what we want?
    # Wait, Testfol reports TWR CAGR or MWRR?
    # Testfol reports BOTH. But "CAGR" usually implies TWR for the STRATEGY, not the Money.
    # HOWEVER, the User's chart error was "5.6%". 
    # If we are verifying the chart, we need to know what the chart depicts.
    # The chart depicts "Portfolio Value". The Table reports "CAGR".
    # Standard CAGR formula on the FINAL VALUE of a DCA strategy is misleading (MWRR).
    
    # Let's calculate both roughly.
    print(f"\nFinal Value: ${final_val:,.2f}")
    print(f"Total Years: {days/365.25:.2f}")
    print(f"Computed End-Point CAGR (Blind): {cagr:.2f}% (Misleading for DCA)")
    
    # To get TWR CAGR (Strategy Performance), we must track a "Unit NAV".
    print("\n3. Calculating TWR (Unit NAV) in parallel...")
    nav = 100.0
    nav_vals = []
    for date, row in df.iterrows():
        r_asset = row['Ret']
        r_lev = (r_asset * L) - ((L-1) * COST_DAILY)
        nav = nav * (1 + r_lev)
        nav_vals.append(nav)
        
    cagr_twr = calculate_cagr(100.0, nav, days)
    print(f"Computed TWR CAGR (Strategy Only): {cagr_twr:.2f}%")
    
    print("="*60)
    print("CONCLUSION:")
    print(f"If the Chart/Table shows ~{cagr_twr:.2f}%, then the Strategy Logic is correct.")
    print(f"If the Chart/Table shows ~40%, likely using MWRR or Double-Lev bug.")
    print("="*60)

if __name__ == "__main__":
    run_verification()

import pandas as pd
import numpy as np

def calculate_cagr(series):
    if series.empty: return 0.0
    start_val = series.iloc[0]
    end_val = series.iloc[-1]
    years = (series.index[-1] - series.index[0]).days / 365.25
    if years <= 0: return 0.0
    return ((end_val / start_val) ** (1 / years) - 1) * 100

def calculate_tax_adjusted_equity(base_equity_series, tax_payment_series, port_series, loan_series, rate_annual):
    if base_equity_series.empty: return base_equity_series, pd.Series(dtype=float)
    
    asset_returns = port_series.pct_change().fillna(0)
    daily_rate = (1 + rate_annual)**(1/365.25) - 1
    
    adj_equity = [base_equity_series.iloc[0]]
    current_equity = base_equity_series.iloc[0]
    
    adj_tax_payments = {}
    
    for i in range(1, len(base_equity_series)):
        date = base_equity_series.index[i]
        r_asset = asset_returns.iloc[i]
        current_loan = loan_series.iloc[i]
        current_assets = current_equity + current_loan
        
        dollar_gain = current_assets * r_asset
        dollar_interest = current_loan * daily_rate
        
        current_equity = current_equity + dollar_gain - dollar_interest
        
        tax = tax_payment_series.iloc[i] if i < len(tax_payment_series) else 0
        if tax > 0:
            full_val = port_series.at[date]
            if full_val > 0:
                scaling_factor = current_assets / full_val
                scaling_factor = min(1.0, max(0.0, scaling_factor))
                tax *= scaling_factor
            current_equity -= tax
            adj_tax_payments[date] = tax
            
        adj_equity.append(current_equity)
        
    adj_equity_series = pd.Series(adj_equity, index=base_equity_series.index)
    adj_tax_series = pd.Series(adj_tax_payments)
    adj_tax_series = adj_tax_series.reindex(base_equity_series.index, fill_value=0.0)
    
    return adj_equity_series, adj_tax_series

# Test Data
dates = pd.date_range(start="2020-01-01", periods=365*2, freq="D")
# Portfolio grows 10% per year (1.1^2 = 1.21)
port_series = pd.Series(np.linspace(10000, 12100, len(dates)), index=dates) 
# Loan is constant 5000 (0.5x leverage initially)
loan_series = pd.Series(5000, index=dates)
# Equity starts at 5000
base_equity_series = port_series - loan_series # This is wrong for simulation but okay for inputs

# Tax payment of $1000 at end of year 1
tax_payment_series = pd.Series(0.0, index=dates)
tax_payment_series.iloc[365] = 1000.0

rate_annual = 0.08

# Run Calculation
adj_equity, adj_tax = calculate_tax_adjusted_equity(base_equity_series, tax_payment_series, port_series, loan_series, rate_annual)

# Calculate CAGR
cagr = calculate_cagr(adj_equity)
print(f"Final Equity: {adj_equity.iloc[-1]:.2f}")
print(f"CAGR: {cagr:.2f}%")

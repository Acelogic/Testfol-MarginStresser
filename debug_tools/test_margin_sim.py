
import pandas as pd
import numpy as np

def simulate_margin(port, starting_loan, rate_annual, draw_monthly, maint_pct, tax_series=None):
    rate_daily = (rate_annual / 100) / 252
    loan_vals, loan = [], starting_loan
    prev_m = port.index[0].month
    
    for d in port.index:
        loan *= 1 + rate_daily
        if draw_monthly and d.month != prev_m:
            loan += draw_monthly
            prev_m = d.month
            
        if tax_series is not None:
            payment = tax_series.get(d, 0.0)
            if payment > 0:
                loan += payment
                
        loan_vals.append(loan)
    loan_series = pd.Series(loan_vals, index=port.index, name="Loan")
    equity = port - loan_series
    equity_pct = (equity / port).rename("Equity %")
    usage_pct = (loan_series / (port * (1 - maint_pct))).rename("Margin usage %")
    return loan_series, equity, equity_pct, usage_pct

# Test Case
dates = pd.date_range("2024-01-01", periods=100)
port = pd.Series(100000.0, index=dates) # Flat portfolio
starting_loan = 50000.0
rate = 0.0
draw = 0.0
maint = 0.25 # 25%

loan, eq, eq_pct, usage = simulate_margin(port, starting_loan, rate, draw, maint)

print(f"Loan: {loan.iloc[-1]}")
print(f"Port: {port.iloc[-1]}")
print(f"Maint: {maint}")
print(f"Expected Usage: {50000 / (100000 * 0.75)}")
print(f"Actual Usage: {usage.iloc[-1]}")

# Test Case 2: High Maint
maint2 = 0.50
loan2, eq2, eq_pct2, usage2 = simulate_margin(port, starting_loan, rate, draw, maint2)
print(f"Expected Usage 2: {50000 / (100000 * 0.50)}")
print(f"Actual Usage 2: {usage2.iloc[-1]}")

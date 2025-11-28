import sys
import os
import pandas as pd
from datetime import date, timedelta

# Add parent directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from shadow_backtest import TaxLot
import tax_library

def test_tax_lot_fifo():
    print("--- Testing Tax Lot FIFO Logic ---")
    
    # Buy 10 shares at $100 on Jan 1, 2020
    lot1 = TaxLot(
        ticker="TEST",
        date_acquired=pd.Timestamp("2020-01-01"),
        quantity=10.0,
        cost_basis_per_share=100.0,
        total_cost_basis=1000.0
    )
    
    # Buy 10 shares at $150 on Jan 1, 2021
    lot2 = TaxLot(
        ticker="TEST",
        date_acquired=pd.Timestamp("2021-01-01"),
        quantity=10.0,
        cost_basis_per_share=150.0,
        total_cost_basis=1500.0
    )
    
    lots = [lot1, lot2]
    
    # Sell 15 shares at $200 on Jan 2, 2022
    # Should consume all of lot1 (10 shares) and 5 shares of lot2
    sell_qty = 15.0
    sell_price = 200.0
    sell_date = pd.Timestamp("2022-01-02")
    
    st_gain = 0.0
    lt_gain = 0.0
    
    shares_remaining = sell_qty
    
    while shares_remaining > 0 and lots:
        current_lot = lots[0]
        sold_qty, cost_basis_sold, remaining_lot = current_lot.sell_shares(shares_remaining)
        
        proceeds = sold_qty * sell_price
        gain = proceeds - cost_basis_sold
        
        holding_period = sell_date - current_lot.date_acquired
        is_long_term = holding_period.days > 365
        
        print(f"Sold {sold_qty} shares acquired {current_lot.date_acquired.date()} (Held {holding_period.days} days)")
        print(f"  Cost Basis: ${cost_basis_sold:.2f}")
        print(f"  Proceeds:   ${proceeds:.2f}")
        print(f"  Gain:       ${gain:.2f} ({'LT' if is_long_term else 'ST'})")
        
        if is_long_term:
            lt_gain += gain
        else:
            st_gain += gain
            
        if remaining_lot:
            lots[0] = remaining_lot
            shares_remaining = 0
        else:
            lots.pop(0)
            shares_remaining -= sold_qty
            
    print(f"Total ST Gain: ${st_gain:.2f}")
    print(f"Total LT Gain: ${lt_gain:.2f}")
    
    # Expected:
    # Lot 1 (10 shares): Held > 1 year (2020 to 2022) -> LT
    #   Gain = 10 * (200 - 100) = 1000
    # Lot 2 (5 shares): Held > 1 year (Jan 2021 to Jan 2022 = 366 days) -> LT
    #   Gain = 5 * (200 - 150) = 250
    # Total LT = 1250
    
    assert lt_gain == 1250.0
    assert st_gain == 0.0
    print("FIFO Test Passed!")
    print()

def test_tax_calculation_st_lt():
    print("--- Testing Tax Calculation (ST vs LT) ---")
    
    other_income = 100000.0 # 22% or 24% bracket
    st_gain = 10000.0
    lt_gain = 10000.0
    year = 2024
    
    tax = tax_library.calculate_tax_on_realized_gains(
        other_income=other_income,
        year=year,
        short_term_gain=st_gain,
        long_term_gain=lt_gain,
        filing_status="Single"
    )
    
    print(f"Other Income: ${other_income:,.2f}")
    print(f"ST Gain:      ${st_gain:,.2f}")
    print(f"LT Gain:      ${lt_gain:,.2f}")
    print(f"Total Tax:    ${tax:,.2f}")
    
    # Verification Logic:
    # ST Gain Tax (Ordinary):
    #   Income 100k -> 110k. 
    #   2024 Single Brackets: 22% up to 100,525, 24% above.
    #   100,000 to 100,525 (525) taxed at 22% = 115.5
    #   100,525 to 110,000 (9475) taxed at 24% = 2274
    #   Approx ST Tax ~ 2389.5
    
    # LT Gain Tax (Preferential):
    #   Stacked on top of 110k.
    #   110k is well within 15% bracket (47k to 518k).
    #   10k * 15% = 1500
    
    # Total Expected ~ 3889.5
    
    print(f"Calculated Tax seems reasonable? {tax > 3000 and tax < 4500}")
    assert tax > 3500
    print("Tax Calc Test Passed!")

if __name__ == "__main__":
    test_tax_lot_fifo()
    test_tax_calculation_st_lt()

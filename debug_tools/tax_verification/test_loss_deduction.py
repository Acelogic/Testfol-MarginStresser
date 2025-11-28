import sys
import os
import pandas as pd

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from tax_library import calculate_tax_series_with_carryforward, calculate_historical_tax

def test_loss_deduction():
    """
    Verifies the $3,000 capital loss deduction logic.
    """
    print("Testing Capital Loss Deduction...")

    # Scenario:
    # Year 2024: $5,000 Short-Term Loss. Other Income $100,000.
    # Expected (Current): Tax = $0 (on gains). Carryforward = $5,000.
    # Expected (Fixed): 
    #   Deduction = $3,000. 
    #   Tax Savings = Tax($100k) - Tax($97k).
    #   Carryforward = $2,000.

    other_income = 100000
    year = 2024
    filing_status = "Single"

    # Create P&L Series
    pl_data = {
        year: {"Realized ST P&L": -5000.0, "Realized LT P&L": 0.0}
    }
    pl_df = pd.DataFrame.from_dict(pl_data, orient='index')

    # Calculate Tax
    tax_series = calculate_tax_series_with_carryforward(
        pl_df, 
        other_income=other_income, 
        filing_status=filing_status
    )
    
    tax_res = tax_series[year]
    print(f"Tax Result for {year}: ${tax_res:.2f}")

    # Calculate expected tax savings manually
    base_tax = calculate_historical_tax(year, other_income, filing_status)
    reduced_tax = calculate_historical_tax(year, other_income - 3000, filing_status)
    expected_savings = base_tax - reduced_tax
    
    print(f"Base Tax on ${other_income:,.0f}: ${base_tax:,.2f}")
    print(f"Tax on ${other_income - 3000:,.0f}: ${reduced_tax:,.2f}")
    print(f"Expected Savings (Negative Tax): -${expected_savings:,.2f}")

    if tax_res < 0:
        print("PASS: Deduction logic is active (Negative tax returned).")
        if abs(abs(tax_res) - expected_savings) < 1.0:
             print("PASS: Savings amount is correct.")
        else:
             print(f"FAIL: Savings amount mismatch. Got {tax_res}, expected -{expected_savings}")
    else:
        print("FAIL: No deduction applied (Tax is 0 or positive).")

if __name__ == "__main__":
    test_loss_deduction()

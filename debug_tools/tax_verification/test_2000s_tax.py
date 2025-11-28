import pandas as pd
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import tax_library

# Test years in the 2000s
test_data = {
    2000: 100000.0,  # 20% max rate era
    2005: 100000.0,  # 15% max rate era (post-2003 tax cuts)
    2010: 100000.0,  # Still 15% max rate
}

other_income = 200000.0  # High earner
filing_status = "Single"

# Update file paths to parent directory
csv_path = os.path.join(os.path.dirname(__file__), "..", "Historical Income Tax Rates and Brackets, 1862-2025.csv")
excel_path = os.path.join(os.path.dirname(__file__), "..", "Federal-Capital-Gains-Tax-Rates-Collections-1913-2025_fv.xlsx")

print("=== Testing 2000-2010 Tax Calculations ===\n")

for year, gain in test_data.items():
    print(f"Year {year}: ${gain:,.0f} capital gain")
    print(f"Other Income: ${other_income:,.0f}")
    
    # Calculate using Historical Smart
    tax = tax_library.calculate_tax_on_realized_gains(
        gain, other_income, year, filing_status, method="historical_smart", excel_path=excel_path
    )
    
    effective_rate = (tax / gain) * 100 if gain > 0 else 0
    
    print(f"  Tax: ${tax:,.2f}")
    print(f"  Effective Rate: {effective_rate:.1f}%")
    print()

print("\nHow it works for 2000-2010:")
print("1. Inclusion Rate: 100% (no exclusion after 1987)")
print("2. Ordinary Tax: Calculated using historical brackets (could be 35-39.6%)")
print("3. Alternative Cap: 20% (2000-2003) or 15% (2003-2010)")
print("4. Result: Pays the LOWER amount → typically the 15-20% preferential rate")

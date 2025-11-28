import pandas as pd
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import tax_library

# Mock Data: Year 1 Loss, Year 2 Gain
# Without carryforward, Year 2 pays full tax.
# With carryforward, Year 2 should pay 0 (or reduced) tax.

pl_data = {
    2020: -50000.0,
    2021: 40000.0
}

other_income = 100000.0
filing_status = "Single"

print("--- Current Logic (No Carryforward) ---")
# ... (Old logic omitted for brevity in output)

print("\n--- New Logic (With Carryforward) ---")
pl_series = pd.Series(pl_data)
tax_series = tax_library.calculate_tax_series_with_carryforward(
    pl_series, other_income, filing_status, method="2024_fixed"
)

for year, tax in tax_series.items():
    print(f"Year {year}: P&L ${pl_series[year]:,.0f} -> Tax ${tax:,.2f}")

print(f"Total Tax Paid: ${tax_series.sum():,.2f}")

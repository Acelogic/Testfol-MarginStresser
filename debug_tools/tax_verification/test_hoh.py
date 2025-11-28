import sys
import os

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from tax_library import get_standard_deduction, calculate_federal_tax

def test_hoh_support():
    """
    Verifies Head of Household support for standard deduction and tax brackets.
    """
    print("Testing Head of Household Support...")

    # 1. Standard Deduction Check
    print("\n--- Standard Deduction Check ---")
    checks = [
        (2024, 21900),
        (2025, 23625),
        (1970, 1100),
        (1988, 4400)
    ]
    
    for year, expected in checks:
        val = get_standard_deduction(year, "Head of Household")
        print(f"{year} HoH Deduction: ${val:,.2f} (Expected: ${expected:,.2f})")
        assert val == expected, f"Mismatch for {year}"
        
    print("PASS: Standard Deductions correct.")

    # 2. Tax Bracket Check (2024)
    # HoH 0% up to $63,000
    print("\n--- 2024 HoH Tax Bracket Check ---")
    
    # Case A: Gain $60,000 (Within 0% bracket)
    tax_0 = calculate_federal_tax(60000, 0, "Head of Household", year=2024)
    print(f"2024 Tax on $60k Gain: ${tax_0:.2f} (Expected: $0.00)")
    assert tax_0 == 0.0
    
    # Case B: Gain $64,000 (Just into 15% bracket)
    # Excess = 1000. Tax = 150.
    tax_15 = calculate_federal_tax(64000, 0, "Head of Household", year=2024)
    print(f"2024 Tax on $64k Gain: ${tax_15:.2f} (Expected: ~$150.00)")
    assert abs(tax_15 - 150.0) < 1.0
    
    # 3. Tax Bracket Check (2025)
    # HoH 0% up to $64,750
    print("\n--- 2025 HoH Tax Bracket Check ---")
    
    # Case C: Gain $64,000 (Now within 0% bracket for 2025!)
    tax_0_2025 = calculate_federal_tax(64000, 0, "Head of Household", year=2025)
    print(f"2025 Tax on $64k Gain: ${tax_0_2025:.2f} (Expected: $0.00)")
    assert tax_0_2025 == 0.0

    print("\nSUCCESS: Head of Household support verified!")

if __name__ == "__main__":
    test_hoh_support()

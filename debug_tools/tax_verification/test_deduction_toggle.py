import sys
import os

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from tax_library import calculate_tax_on_realized_gains

def test_deduction_toggle():
    """
    Verifies that the standard deduction toggle works.
    """
    print("Testing Standard Deduction Toggle...")

    # Scenario: 2024 Single, $100k Income, $10k ST Gain
    # With Deduction: Taxable Ordinary = $100k - $14.6k = $85.4k.
    # Without Deduction: Taxable Ordinary = $100k.
    
    # The ST gain sits on top.
    # Tax(Base + Gain) - Tax(Base)
    # The marginal rate might be the same (22% or 24%), but let's check.
    # 2024 Single Brackets:
    # 22% up to $100,525
    # 24% up to $191,950
    
    # Without Deduction: Base $100k. Gain $10k -> $110k.
    # $525 taxed at 22%, $9475 taxed at 24%.
    
    # With Deduction: Base $85.4k. Gain $10k -> $95.4k.
    # All taxed at 22%.
    
    # So tax should be LOWER with deduction.
    
    tax_with = calculate_tax_on_realized_gains(
        realized_gain=0,
        other_income=100000,
        year=2024,
        filing_status="Single",
        short_term_gain=10000,
        use_standard_deduction=True
    )
    
    tax_without = calculate_tax_on_realized_gains(
        realized_gain=0,
        other_income=100000,
        year=2024,
        filing_status="Single",
        short_term_gain=10000,
        use_standard_deduction=False
    )
    
    print(f"Tax WITH Deduction: ${tax_with:.2f}")
    print(f"Tax WITHOUT Deduction: ${tax_without:.2f}")
    
    if tax_with < tax_without:
        print("PASS: Tax is lower with deduction enabled.")
        diff = tax_without - tax_with
        print(f"Difference: ${diff:.2f}")
    else:
        print("FAIL: Tax did not decrease with deduction.")
        assert False

if __name__ == "__main__":
    test_deduction_toggle()

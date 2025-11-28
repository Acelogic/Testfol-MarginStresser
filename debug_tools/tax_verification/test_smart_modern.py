import sys
import os

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from tax_library import calculate_tax_on_realized_gains

def test_smart_modern_brackets():
    """
    Verifies that 'historical_smart' uses the correct brackets for 2024 and 2025.
    """
    print("Testing Smart Calculation for Modern Years...")

    # Case 1: 2024 Tax
    # Single Filer
    # 0% bracket up to $47,025
    # Gain: $47,000 (Should be $0 tax)
    tax_2024_zero = calculate_tax_on_realized_gains(
        realized_gain=47000, 
        other_income=0, 
        year=2024, 
        filing_status="Single", 
        method="historical_smart"
    )
    print(f"2024 Tax on $47k (Limit $47,025): ${tax_2024_zero:.2f} (Expected: $0.00)")
    assert tax_2024_zero == 0.0

    # Gain: $48,000 (Should be taxed on excess)
    # Excess = 48000 - 47025 = 975
    # Tax = 975 * 0.15 = 146.25
    tax_2024_taxed = calculate_tax_on_realized_gains(
        realized_gain=48000, 
        other_income=0, 
        year=2024, 
        filing_status="Single", 
        method="historical_smart"
    )
    print(f"2024 Tax on $48k: ${tax_2024_taxed:.2f} (Expected: ~$146.25)")
    assert abs(tax_2024_taxed - 146.25) < 1.0

    # Case 2: 2025 Tax
    # Single Filer
    # 0% bracket up to $49,450
    # Gain: $48,000 (Should be $0 tax because 2025 limit is higher)
    tax_2025_zero = calculate_tax_on_realized_gains(
        realized_gain=48000, 
        other_income=0, 
        year=2025, 
        filing_status="Single", 
        method="historical_smart"
    )
    print(f"2025 Tax on $48k (Limit $49,450): ${tax_2025_zero:.2f} (Expected: $0.00)")
    assert tax_2025_zero == 0.0

    print("\nSUCCESS: Smart mode correctly distinguishes 2024 and 2025 brackets!")

if __name__ == "__main__":
    test_smart_modern_brackets()

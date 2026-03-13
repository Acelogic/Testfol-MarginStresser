"""Tests for app/core/tax_library.py — pure math with data files."""

import pytest
from app.core.tax_library import (
    get_standard_deduction,
    calculate_tax_on_realized_gains,
    calculate_federal_tax,
)


def test_standard_deduction_2024_single():
    """Known $14,600 for Single filer in 2024."""
    deduction = get_standard_deduction(2024, "Single")
    assert deduction == 14600


def test_capital_gains_zero():
    """Zero gains -> zero tax."""
    tax = calculate_tax_on_realized_gains(
        realized_gain=0.0,
        other_income=100000.0,
        year=2024,
        filing_status="Single",
    )
    assert tax == 0.0


def test_capital_gains_basic():
    """$50k LT gain, $100k income -> plausible tax (should be > 0)."""
    tax = calculate_tax_on_realized_gains(
        long_term_gain=50000.0,
        other_income=100000.0,
        year=2024,
        filing_status="Single",
        method="2024_fixed",
    )
    assert tax > 0
    # 15% bracket for most of this: $50k * 15% = $7500
    # Could be slightly different due to stacking, but should be in a plausible range
    assert 5000 < tax < 15000


def test_short_term_taxed_as_ordinary():
    """ST gains should be taxed higher than LT gains (same amount)."""
    tax_st = calculate_tax_on_realized_gains(
        short_term_gain=50000.0,
        other_income=100000.0,
        year=2024,
        filing_status="Single",
        method="2024_fixed",
    )
    tax_lt = calculate_tax_on_realized_gains(
        long_term_gain=50000.0,
        other_income=100000.0,
        year=2024,
        filing_status="Single",
        method="2024_fixed",
    )
    assert tax_st > tax_lt


# ---------------------------------------------------------------------------
# Bug-fix regression tests
# ---------------------------------------------------------------------------


def test_niit_not_double_counted():
    """NIIT must appear once (in calculate_tax_on_realized_gains), not in calculate_federal_tax."""
    # Single, 2024: $300k ordinary + $200k LT gain → MAGI $500k > $200k threshold
    bracket_only = calculate_federal_tax(200000, 300000, "Single", year=2024)
    # bracket_only must NOT contain any 3.8% NIIT component
    # At $300k stacking income, entire $200k is in 15% bracket (tops out at $518,900)
    expected_bracket = 200000 * 0.15
    assert bracket_only == pytest.approx(expected_bracket, rel=1e-6)

    full_tax = calculate_tax_on_realized_gains(
        long_term_gain=200000,
        other_income=300000,
        year=2024,
        filing_status="Single",
        method="2024_fixed",
        use_standard_deduction=False,
    )
    # full_tax = bracket tax + exactly one NIIT
    niit = min(200000, (300000 + 200000) - 200000) * 0.038  # min(gain, excess) * 3.8%
    assert full_tax == pytest.approx(bracket_only + niit, rel=1e-6)


def test_mfs_capital_gains_brackets():
    """MFS 20% rate kicks in at $291,875 (2024), not $518,900 (Single)."""
    # $300k LT gain, no other income, MFS 2024
    tax_mfs = calculate_federal_tax(300000, 0, "Married Filing Separately", year=2024)
    # 0% on first $47,025, 15% on $47,025→$291,875 ($244,850), 20% on $291,875→$300,000 ($8,125)
    expected = 0 + 244850 * 0.15 + 8125 * 0.20
    assert tax_mfs == pytest.approx(expected, rel=1e-6)

    # Same scenario as Single → all in 15% (threshold $518,900)
    tax_single = calculate_federal_tax(300000, 0, "Single", year=2024)
    assert tax_mfs > tax_single  # MFS should pay more


def test_mfs_niit_threshold():
    """MFS NIIT threshold is $125k, not $200k (Single)."""
    # $150k MAGI, $50k LT gain, MFS → triggers NIIT (>$125k)
    tax_mfs = calculate_tax_on_realized_gains(
        long_term_gain=50000,
        other_income=100000,
        year=2024,
        filing_status="Married Filing Separately",
        method="2024_fixed",
        use_standard_deduction=False,
    )
    # Same scenario as Single → does NOT trigger NIIT (<$200k)
    tax_single = calculate_tax_on_realized_gains(
        long_term_gain=50000,
        other_income=100000,
        year=2024,
        filing_status="Single",
        method="2024_fixed",
        use_standard_deduction=False,
    )
    # MFS should be higher due to NIIT
    assert tax_mfs > tax_single


def test_standard_deduction_mfs():
    """MFS deduction = MFJ // 2 for 2024 and 2025."""
    assert get_standard_deduction(2024, "Married Filing Separately") == 29200 // 2  # 14600
    assert get_standard_deduction(2025, "Married Filing Separately") == 30000 // 2  # 15000


def test_collectible_tax_included():
    """Collectible gains must produce non-zero tax (Bug 1: was computed but not returned)."""
    tax = calculate_tax_on_realized_gains(
        long_term_gain_collectible=100000.0,
        other_income=50000.0,
        year=2024,
        filing_status="Single",
        method="2024_fixed",
        use_standard_deduction=False,
    )
    # Collectible taxed at min(ordinary_rate, 28%); at $50k stacking ≈ 22% bracket → 22%
    # $100k * 22% = $22k, plus possible NIIT since MAGI=$150k < $200k → no NIIT
    assert tax > 0
    # Should be roughly $22k–$28k range
    assert 15000 < tax < 35000


def test_niit_uses_gross_income():
    """NIIT MAGI must use pre-deduction gain values (Bug 2: was using post-deduction)."""
    # $200k LT gain + $100k ordinary → MAGI = $300k; Single threshold = $200k
    # Excess = $100k; NIIT = min($200k gain, $100k excess) * 3.8% = $3,800
    tax_with_niit = calculate_tax_on_realized_gains(
        long_term_gain=200000.0,
        other_income=100000.0,
        year=2024,
        filing_status="Single",
        method="2024_fixed",
        use_standard_deduction=False,
    )
    # Without NIIT, bracket tax on $200k LT stacked on $100k:
    # 0% up to $47,025, 15% on $47,025-$200k (all in 15% bracket)
    bracket_tax = calculate_federal_tax(200000.0, 100000.0, "Single", year=2024)
    niit = min(200000.0, (100000.0 + 200000.0) - 200000.0) * 0.038  # $3,800
    assert tax_with_niit == pytest.approx(bracket_tax + niit, rel=1e-6)


def test_year_passthrough_default_method():
    """Default method path must use year=2024 brackets when year=2024, not 2025."""
    # 2024 Single 0% threshold = $47,025; 2025 = $49,450
    # $48k gain on zero income: if 2024 brackets used, $975 taxed at 15%; if 2025, $0 tax
    tax_2024 = calculate_tax_on_realized_gains(
        long_term_gain=48000,
        other_income=0,
        year=2024,
        filing_status="Single",
        method="2024_fixed",
        use_standard_deduction=False,
    )
    tax_2025 = calculate_tax_on_realized_gains(
        long_term_gain=48000,
        other_income=0,
        year=2025,
        filing_status="Single",
        method="2024_fixed",
        use_standard_deduction=False,
    )
    # 2024: $48k - $47,025 = $975 at 15% = $146.25
    assert tax_2024 == pytest.approx(975 * 0.15, rel=1e-6)
    # 2025: $48k < $49,450 → $0 bracket tax (only NIIT if applicable, but MAGI=$48k < $200k)
    assert tax_2025 == 0.0

"""Tests for app/core/tax_library.py — pure math with data files."""

import pytest
from app.core.tax_library import (
    get_standard_deduction,
    calculate_tax_on_realized_gains,
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

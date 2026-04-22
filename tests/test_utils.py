"""Tests for shared UI utility helpers."""

from app.common import utils


def test_optional_cents_format_hides_zero_cents():
    assert utils.optional_cents_format(100000.0) == "%.0f"
    assert utils.optional_cents_format("1000") == "%.0f"


def test_optional_cents_format_shows_nonzero_cents():
    assert utils.optional_cents_format(100000.25) == "%.2f"
    assert utils.optional_cents_format("1000.10") == "%.2f"

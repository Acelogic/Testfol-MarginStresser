"""Tests for app/core/calculations.py — pure functions, no mocking needed."""

import pandas as pd
import numpy as np
import pytest
from app.core.calculations import (
    calculate_cagr,
    calculate_max_drawdown,
    calculate_sharpe_ratio,
    generate_stats,
    calculate_pivot_points,
)


# ---------------------------------------------------------------------------
# CAGR
# ---------------------------------------------------------------------------

def test_cagr_known_growth():
    """10000 -> 20000 over 1 year = 100% CAGR."""
    dates = pd.bdate_range("2023-01-02", "2023-12-29")
    values = np.linspace(10000, 20000, len(dates))
    series = pd.Series(values, index=dates)
    cagr = calculate_cagr(series)
    # ~100% CAGR (not exactly 100 because bdate range ≠ exactly 365.25 days)
    assert 95.0 < cagr < 110.0


def test_cagr_flat():
    """Constant series -> 0% CAGR."""
    dates = pd.bdate_range("2023-01-02", "2023-12-29")
    series = pd.Series(10000.0, index=dates)
    cagr = calculate_cagr(series)
    assert cagr == pytest.approx(0.0, abs=0.01)


def test_cagr_empty():
    """Empty series -> 0.0."""
    series = pd.Series(dtype=float)
    assert calculate_cagr(series) == 0.0


# ---------------------------------------------------------------------------
# Max Drawdown
# ---------------------------------------------------------------------------

def test_max_drawdown():
    """Series with peak 100, trough 70 -> -30%."""
    dates = pd.date_range("2023-01-01", periods=5, freq="D")
    values = [100, 90, 70, 80, 95]  # peak=100, trough=70
    series = pd.Series(values, index=dates)
    mdd = calculate_max_drawdown(series)
    assert mdd == pytest.approx(-30.0, abs=0.01)


def test_max_drawdown_no_drawdown():
    """Monotonically increasing -> 0%."""
    dates = pd.date_range("2023-01-01", periods=5, freq="D")
    values = [100, 110, 120, 130, 140]
    series = pd.Series(values, index=dates)
    mdd = calculate_max_drawdown(series)
    assert mdd == pytest.approx(0.0, abs=0.01)


# ---------------------------------------------------------------------------
# Sharpe Ratio
# ---------------------------------------------------------------------------

def test_sharpe_ratio():
    """Positive excess returns -> positive Sharpe."""
    dates = pd.bdate_range("2023-01-02", "2023-12-29")
    # Steadily growing series → consistent positive returns
    values = np.linspace(10000, 15000, len(dates))
    series = pd.Series(values, index=dates)
    sharpe = calculate_sharpe_ratio(series, risk_free_rate=0.0)
    assert sharpe > 0


# ---------------------------------------------------------------------------
# generate_stats
# ---------------------------------------------------------------------------

def test_generate_stats_keys():
    """All expected keys present in output."""
    dates = pd.bdate_range("2023-01-02", "2023-12-29")
    series = pd.Series(np.linspace(10000, 12000, len(dates)), index=dates)
    stats = generate_stats(series)
    expected_keys = {"cagr", "std", "sharpe", "max_drawdown", "best_year", "worst_year", "ulcer", "sortino"}
    assert expected_keys.issubset(stats.keys())


def test_generate_stats_empty():
    """Empty series -> empty dict."""
    series = pd.Series(dtype=float)
    stats = generate_stats(series)
    assert stats == {}


# ---------------------------------------------------------------------------
# Pivot Points
# ---------------------------------------------------------------------------

def test_pivot_points():
    """Known H/L/C -> verify support/resistance formulas."""
    high, low, close = 110.0, 90.0, 100.0
    result = calculate_pivot_points(high, low, close)

    # Expected pivot point
    p = (high + low + close) / 3  # 100.0
    r1 = (2 * p) - low  # 110.0
    s1 = (2 * p) - high  # 90.0
    r2 = p + (high - low)  # 120.0
    s2 = p - (high - low)  # 80.0

    prices = {item["Label"]: item["Price"] for item in result}
    assert prices["Pivot Point"] == pytest.approx(p, abs=0.01)
    assert prices["Pivot Point 1st Level Resistance"] == pytest.approx(r1, abs=0.01)
    assert prices["Pivot Point 1st Level Support"] == pytest.approx(s1, abs=0.01)
    assert prices["Pivot Point 2nd Level Resistance"] == pytest.approx(r2, abs=0.01)
    assert prices["Pivot Point 2nd Level Support"] == pytest.approx(s2, abs=0.01)

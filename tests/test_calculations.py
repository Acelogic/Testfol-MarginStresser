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
    find_drawdown_episodes,
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
    expected_keys = {"cagr", "std", "sharpe", "max_drawdown", "best_year", "worst_year", "ulcer_index", "sortino", "calmar", "avg_drawdown"}
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


# ---------------------------------------------------------------------------
# find_drawdown_episodes
# ---------------------------------------------------------------------------

def test_find_drawdown_episodes_basic():
    """Detects a single drawdown episode with recovery."""
    dates = pd.bdate_range("2023-01-02", periods=60)
    values = np.concatenate([
        np.linspace(100, 120, 15),   # rise
        np.linspace(120, 80, 15),    # -33% drawdown
        np.linspace(80, 120, 15),    # recovery
        np.linspace(120, 130, 15),   # new high
    ])
    series = pd.Series(values, index=dates)
    episodes = find_drawdown_episodes(series, threshold=-0.05)
    assert len(episodes) == 1
    ep = episodes[0]
    assert ep["peak_val"] == pytest.approx(120.0, abs=1.0)
    assert ep["trough_val"] == pytest.approx(80.0, abs=1.0)
    assert ep["dd"] == pytest.approx(-1/3, abs=0.05)
    assert ep["recovery"] is not None


def test_find_drawdown_episodes_ongoing():
    """Detects an ongoing drawdown (no recovery)."""
    dates = pd.bdate_range("2023-01-02", periods=30)
    values = np.concatenate([
        np.linspace(100, 120, 15),
        np.linspace(120, 90, 15),   # -25% drop, no recovery
    ])
    series = pd.Series(values, index=dates)
    episodes = find_drawdown_episodes(series, threshold=-0.05)
    assert len(episodes) == 1
    assert episodes[0]["recovery"] is None


def test_find_drawdown_episodes_below_threshold():
    """Small drawdowns below threshold are ignored."""
    dates = pd.bdate_range("2023-01-02", periods=30)
    values = np.concatenate([
        np.linspace(100, 105, 15),
        np.linspace(105, 102, 15),  # -2.8% drop, below 5% threshold
    ])
    series = pd.Series(values, index=dates)
    episodes = find_drawdown_episodes(series, threshold=-0.05)
    assert len(episodes) == 0


def test_find_drawdown_episodes_empty():
    """Empty series returns empty list."""
    series = pd.Series(dtype=float)
    assert find_drawdown_episodes(series) == []

"""Tests for app.services.testfol_api."""
from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from app.services.testfol_api import table_to_dicts, fetch_backtest, simulate_margin


# ---------------------------------------------------------------------------
# table_to_dicts
# ---------------------------------------------------------------------------

def test_table_to_dicts():
    df = pd.DataFrame({
        "Ticker": ["AAPL", "GOOG"],
        "Weight %": [60.0, 40.0],
        "Maint %": [25.0, 30.0],
    })
    alloc, maint = table_to_dicts(df)
    assert alloc == {"AAPL": 60.0, "GOOG": 40.0}
    assert maint == {"AAPL": 25.0, "GOOG": 30.0}


def test_table_to_dicts_strips_ticker_modifiers():
    df = pd.DataFrame({
        "Ticker": ["AAPL?L=2"],
        "Weight %": [100.0],
        "Maint %": [25.0],
    })
    alloc, maint = table_to_dicts(df)
    assert "AAPL?L=2" in alloc
    # Maint strips the ?L=2 modifier
    assert "AAPL" in maint


def test_table_to_dicts_drops_nan_ticker():
    df = pd.DataFrame({
        "Ticker": ["SPY", None],
        "Weight %": [100.0, 0.0],
        "Maint %": [25.0, 25.0],
    })
    alloc, _ = table_to_dicts(df)
    assert len(alloc) == 1
    assert "SPY" in alloc


# ---------------------------------------------------------------------------
# fetch_backtest — cached path
# ---------------------------------------------------------------------------

def test_fetch_backtest_cached(monkeypatch):
    """When cache_get returns data, no HTTP call should be made."""
    fake_result = (pd.Series([10000, 10100], name="Portfolio"), {"cagr": 0.05}, {})
    monkeypatch.setattr("app.services.testfol_api.cache_get", lambda *a, **kw: fake_result)

    result = fetch_backtest(
        start_date="2020-01-01", end_date="2024-01-01",
        start_val=10000, cashflow=0, cashfreq="Monthly",
        rolling=1, invest_div=True, rebalance="Yearly",
        allocation={"SPY": 100.0},
    )
    assert result is fake_result


# ---------------------------------------------------------------------------
# fetch_backtest — API call
# ---------------------------------------------------------------------------

def _make_api_response(start_val=10000):
    """Build a realistic API JSON response."""
    ts = [1577836800, 1577923200]  # 2020-01-01, 2020-01-02
    vals = [start_val, start_val + 50]
    return {
        "charts": {"history": [ts, vals]},
        "stats": {"cagr": 0.1},
        "rebalancing_events": [],
        "rebalancing_stats": [],
        "daily_returns": [],
    }


def test_fetch_backtest_api_call(monkeypatch):
    """Simulate a cache miss and mock the HTTP POST."""
    monkeypatch.setattr("app.services.testfol_api.cache_get", lambda *a, **kw: None)
    monkeypatch.setattr("app.services.testfol_api.cache_set", lambda *a, **kw: None)

    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.content = b"x" * 100
    mock_response.json.return_value = _make_api_response()

    mock_session = MagicMock()
    mock_session.post.return_value = mock_response

    with patch("app.services.testfol_api.requests.Session", return_value=mock_session):
        series, stats, extra = fetch_backtest(
            start_date="2020-01-01", end_date="2020-01-02",
            start_val=10000, cashflow=0, cashfreq="Monthly",
            rolling=1, invest_div=True, rebalance="Yearly",
            allocation={"SPY": 100.0},
        )

    assert isinstance(series, pd.Series)
    assert len(series) == 2
    assert stats["cagr"] == 0.1


def test_fetch_backtest_return_raw(monkeypatch):
    """return_raw=True should return the raw dict."""
    monkeypatch.setattr("app.services.testfol_api.cache_get", lambda *a, **kw: None)
    monkeypatch.setattr("app.services.testfol_api.cache_set", lambda *a, **kw: None)

    raw = _make_api_response()
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.content = b"x"
    mock_response.json.return_value = raw

    mock_session = MagicMock()
    mock_session.post.return_value = mock_response

    with patch("app.services.testfol_api.requests.Session", return_value=mock_session):
        result = fetch_backtest(
            start_date="2020-01-01", end_date="2020-01-02",
            start_val=10000, cashflow=0, cashfreq="Monthly",
            rolling=1, invest_div=True, rebalance="Yearly",
            allocation={"SPY": 100.0},
            return_raw=True,
        )

    assert isinstance(result, dict)
    assert "charts" in result


def test_fetch_backtest_validation_missing_charts(monkeypatch):
    monkeypatch.setattr("app.services.testfol_api.cache_get", lambda *a, **kw: None)

    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.content = b"x"
    mock_response.json.return_value = {"stats": {}}

    mock_session = MagicMock()
    mock_session.post.return_value = mock_response

    with patch("app.services.testfol_api.requests.Session", return_value=mock_session):
        with pytest.raises(ValueError, match="missing chart history"):
            fetch_backtest(
                start_date="2020-01-01", end_date="2020-01-02",
                start_val=10000, cashflow=0, cashfreq="Monthly",
                rolling=1, invest_div=True, rebalance="Yearly",
                allocation={"SPY": 100.0},
            )


def test_fetch_backtest_validation_length_mismatch(monkeypatch):
    monkeypatch.setattr("app.services.testfol_api.cache_get", lambda *a, **kw: None)

    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.content = b"x"
    mock_response.json.return_value = {
        "charts": {"history": [[1, 2, 3], [100, 200]]},
        "stats": {},
    }

    mock_session = MagicMock()
    mock_session.post.return_value = mock_response

    with patch("app.services.testfol_api.requests.Session", return_value=mock_session):
        with pytest.raises(ValueError, match="length mismatch"):
            fetch_backtest(
                start_date="2020-01-01", end_date="2020-01-02",
                start_val=10000, cashflow=0, cashfreq="Monthly",
                rolling=1, invest_div=True, rebalance="Yearly",
                allocation={"SPY": 100.0},
            )


# ---------------------------------------------------------------------------
# simulate_margin
# ---------------------------------------------------------------------------

@pytest.fixture
def port_series():
    """Portfolio value series: $10k constant over 60 business days."""
    dates = pd.bdate_range("2023-01-02", periods=60)
    return pd.Series(10000.0, index=dates, name="Portfolio")


def test_simulate_margin_no_loan(port_series):
    loan, equity, eq_pct, usage, rate = simulate_margin(
        port=port_series, starting_loan=0, rate_annual=5.0,
        draw_monthly=0, maint_pct=0.25,
    )
    # With no starting loan and no draws, loan stays ~0 and equity ~= port
    assert loan.iloc[0] == pytest.approx(0, abs=1e-6)
    assert equity.iloc[0] == pytest.approx(10000.0, abs=1)


def test_simulate_margin_with_draws(port_series):
    loan, equity, eq_pct, usage, rate = simulate_margin(
        port=port_series, starting_loan=0, rate_annual=5.0,
        draw_monthly=500, maint_pct=0.25,
    )
    # After 60 days (~3 months) the loan should have grown from draws
    assert loan.iloc[-1] > 0
    assert equity.iloc[-1] < port_series.iloc[-1]


def test_simulate_margin_with_tax(port_series):
    # Inject a tax payment on day 10
    tax = pd.Series(0.0, index=port_series.index)
    tax.iloc[10] = 1000.0

    loan, equity, eq_pct, usage, rate = simulate_margin(
        port=port_series, starting_loan=0, rate_annual=5.0,
        draw_monthly=0, maint_pct=0.25, tax_series=tax,
    )
    # Tax payment should increase the loan
    assert loan.iloc[-1] > 500  # Tax + some interest


def test_simulate_margin_variable_rate(port_series):
    # Build a variable rate model
    base = pd.Series(5.0, index=port_series.index)
    model = {"type": "Variable", "base_series": base, "spread_pct": 1.0}

    loan, equity, eq_pct, usage, rate = simulate_margin(
        port=port_series, starting_loan=5000, rate_annual=model,
        draw_monthly=0, maint_pct=0.25,
    )
    # Rate should be ~6% (base 5 + spread 1)
    assert rate.iloc[0] == pytest.approx(6.0, abs=0.01)
    # Loan should grow slightly from interest
    assert loan.iloc[-1] > 5000

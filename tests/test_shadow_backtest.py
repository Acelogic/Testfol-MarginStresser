"""Tests for app/core/shadow_backtest.py — pass prices_df to bypass yfinance."""

import pandas as pd
import numpy as np
import pytest
from app.core.shadow_backtest import parse_ticker, run_shadow_backtest


# ---------------------------------------------------------------------------
# parse_ticker
# ---------------------------------------------------------------------------

def test_parse_ticker_simple():
    """Plain ticker -> (ticker, {})."""
    base, params = parse_ticker("SPY")
    # SPY maps to itself (or SPYSIM maps to SPY, but plain SPY stays SPY)
    assert base == "SPY"
    assert params == {}


def test_parse_ticker_with_params():
    """Ticker with query params -> parsed correctly."""
    base, params = parse_ticker("SPY?L=2&SW=30")
    assert base == "SPY"
    assert params == {"L": "2", "SW": "30"}


# ---------------------------------------------------------------------------
# run_shadow_backtest — basic scenarios
# ---------------------------------------------------------------------------

def test_basic_flat_backtest(flat_prices, sample_allocation):
    """Flat prices -> portfolio value stays at start_val."""
    result = run_shadow_backtest(
        allocation=sample_allocation,
        start_val=10000.0,
        start_date="2023-01-02",
        end_date="2023-12-29",
        prices_df=flat_prices,
        rebalance_freq="Yearly",
    )
    trades_df, pl_by_year, composition_df, unrealized_pl_df, logs, port_series, twr_series = result
    assert not port_series.empty
    # With flat prices, portfolio should stay near start_val
    assert port_series.iloc[-1] == pytest.approx(10000.0, rel=0.01)


def test_growth_backtest(growing_prices, sample_allocation):
    """Prices double -> portfolio roughly doubles, CAGR ~ 100% over the period."""
    result = run_shadow_backtest(
        allocation=sample_allocation,
        start_val=10000.0,
        start_date="2020-01-02",
        end_date="2024-12-31",
        prices_df=growing_prices,
        rebalance_freq="Yearly",
    )
    trades_df, pl_by_year, composition_df, unrealized_pl_df, logs, port_series, twr_series = result
    assert not port_series.empty
    # Prices go from 100 to 200 => portfolio should roughly double
    final_val = port_series.iloc[-1]
    assert 18000 < final_val < 22000


def test_dca_monthly(flat_prices, sample_allocation):
    """$1000/month cashflow -> correct number of buy trades."""
    result = run_shadow_backtest(
        allocation=sample_allocation,
        start_val=10000.0,
        start_date="2023-01-02",
        end_date="2023-12-29",
        prices_df=flat_prices,
        rebalance_freq="Yearly",
        cashflow=1000.0,
        cashflow_freq="Monthly",
    )
    trades_df, pl_by_year, composition_df, unrealized_pl_df, logs, port_series, twr_series = result
    # Should have roughly 11-12 monthly cashflow buy events
    if not trades_df.empty:
        buy_trades = trades_df[trades_df["Trade Amount"] > 0]
        assert len(buy_trades) >= 10  # At least 10 months of DCA


def test_return_structure(flat_prices, sample_allocation):
    """Verify 7-tuple, correct types."""
    result = run_shadow_backtest(
        allocation=sample_allocation,
        start_val=10000.0,
        start_date="2023-01-02",
        end_date="2023-12-29",
        prices_df=flat_prices,
        rebalance_freq="Yearly",
    )
    assert len(result) == 7
    trades_df, pl_by_year, composition_df, unrealized_pl_df, logs, port_series, twr_series = result
    assert isinstance(trades_df, pd.DataFrame)
    assert isinstance(logs, list)
    assert isinstance(port_series, pd.Series)

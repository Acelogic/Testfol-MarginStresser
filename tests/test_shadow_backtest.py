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
    trades_df, pl_by_year, composition_df, unrealized_pl_df, logs, port_series, twr_series, *_ = result
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
    trades_df, pl_by_year, composition_df, unrealized_pl_df, logs, port_series, twr_series, *_ = result
    assert not port_series.empty
    # Prices go from 100 to 200 => portfolio should roughly double
    final_val = port_series.iloc[-1]
    assert 18000 < final_val < 22000


def test_shadow_backtest_does_not_forward_fill_stale_component_prices():
    dates = pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04"])
    prices = pd.DataFrame(
        {
            "SIM": [100.0, 101.0, None],
            "LIVE": [100.0, 101.0, 102.0],
        },
        index=dates,
    )

    result = run_shadow_backtest(
        allocation={"SIM": 50.0, "LIVE": 50.0},
        start_val=10000.0,
        start_date="2024-01-02",
        end_date="2024-01-04",
        prices_df=prices,
        rebalance_freq="Yearly",
    )

    _, _, _, _, logs, port_series, _, *_ = result
    assert not port_series.empty, logs
    assert port_series.index.max() == pd.Timestamp("2024-01-03")


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
    trades_df, pl_by_year, composition_df, unrealized_pl_df, logs, port_series, twr_series, *_ = result
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
    assert len(result) == 8
    trades_df, pl_by_year, composition_df, unrealized_pl_df, logs, port_series, twr_series, *_ = result
    assert isinstance(trades_df, pd.DataFrame)
    assert isinstance(logs, list)
    assert isinstance(port_series, pd.Series)


def test_monthly_rebalance_keeps_running_lot_basis_consistent():
    """Selling at rebalance should update realized gains and unrealized basis correctly."""
    dates = pd.to_datetime(["2023-01-27", "2023-01-30", "2023-01-31", "2023-02-01", "2023-02-02"])
    prices = pd.DataFrame(
        {
            "A": [100.0, 100.0, 110.0, 110.0, 110.0],
            "B": [100.0, 100.0, 100.0, 100.0, 100.0],
        },
        index=dates,
    )

    trades_df, _, _, unrealized_pl_df, _, port_series, _, *_ = run_shadow_backtest(
        allocation={"A": 50.0, "B": 50.0},
        start_val=100.0,
        start_date="2023-01-27",
        end_date="2023-02-02",
        prices_df=prices,
        rebalance_freq="Monthly",
    )

    assert port_series.loc[pd.Timestamp("2023-01-31")] == pytest.approx(105.0)

    sell_trade = trades_df[(trades_df["Ticker"] == "A") & (trades_df["Trade Amount"] < 0)].iloc[0]
    buy_trade = trades_df[(trades_df["Ticker"] == "B") & (trades_df["Trade Amount"] > 0)].iloc[0]

    assert sell_trade["Trade Amount"] == pytest.approx(-2.5)
    assert sell_trade["Realized ST P&L"] == pytest.approx(0.2272727273, rel=1e-6)
    assert buy_trade["Trade Amount"] == pytest.approx(2.5)

    jan_unrealized = unrealized_pl_df.loc[pd.Timestamp("2023-01-31"), "Unrealized P&L"]
    assert jan_unrealized == pytest.approx(4.7727272727, rel=1e-6)

"""Tests for app.core.backtest_orchestrator module."""

import pandas as pd
import pytest

from app.common.constants import Freq, RebalMode
from app.core.backtest_orchestrator import calc_rebal_offset, run_single_backtest


# ---------------------------------------------------------------------------
# calc_rebal_offset
# ---------------------------------------------------------------------------

class TestCalcRebalOffset:
    def test_standard_mode_returns_zero(self):
        """Standard mode always returns 0 offset."""
        reb = {"mode": RebalMode.STANDARD, "freq": Freq.YEARLY}
        assert calc_rebal_offset(reb, Freq.YEARLY) == 0

    def test_custom_yearly_jan(self):
        """Custom yearly rebalancing in January yields high offset (far from Dec 31)."""
        reb = {"mode": RebalMode.CUSTOM, "month": 1, "day": 1}
        offset = calc_rebal_offset(reb, Freq.YEARLY)
        assert offset > 200  # ~252 trading days minus a few

    def test_custom_yearly_dec(self):
        """Custom yearly rebalancing in December yields low offset."""
        reb = {"mode": RebalMode.CUSTOM, "month": 12, "day": 31}
        offset = calc_rebal_offset(reb, Freq.YEARLY)
        assert offset == 0

    def test_custom_monthly(self):
        """Custom monthly returns a small offset."""
        reb = {"mode": RebalMode.CUSTOM, "month": 1, "day": 15}
        offset = calc_rebal_offset(reb, Freq.MONTHLY)
        assert 0 <= offset <= 21  # trading days in a month

    def test_missing_keys_defaults(self):
        """Missing month/day keys use defaults without error."""
        reb = {"mode": RebalMode.CUSTOM}
        offset = calc_rebal_offset(reb, Freq.YEARLY)
        assert isinstance(offset, int)


# ---------------------------------------------------------------------------
# run_single_backtest
# ---------------------------------------------------------------------------

def _mock_fetch(start_date, end_date, start_val, cashflow, cashfreq, rolling,
                invest_div, rebalance, allocation, return_raw=False,
                include_raw=False, rebalance_offset=0, cashflow_offset=0, **kwargs):
    """Minimal mock for fetch_backtest that returns deterministic data."""
    dates = pd.bdate_range(start_date, end_date)
    vals = [start_val * (1 + 0.0003 * i) for i in range(len(dates))]
    series = pd.Series(vals, index=dates, name="Portfolio")
    stats = {"cagr": 0.08, "max_drawdown": -5.0, "sharpe": 1.2, "volatility": 0.15}
    extra = {"rebalancing_events": [], "rebalancing_stats": [], "daily_returns": []}
    if return_raw:
        return {"charts": {"history": [[int(d.timestamp()) for d in dates], vals]}, "stats": stats}
    return (series, stats, extra)


def _mock_shadow(allocation, start_val, start_date, end_date,
                 api_port_series=None, rebalance_freq="Yearly",
                 cashflow=0.0, cashflow_freq="Monthly", prices_df=None,
                 rebalance_month=1, rebalance_day=1, custom_freq="Yearly",
                 invest_dividends=True, pay_down_margin=False,
                 tax_config=None, custom_rebal_config=None):
    """Minimal mock for run_shadow_backtest."""
    dates = pd.bdate_range(start_date, end_date)
    vals = [start_val * (1 + 0.0003 * i) for i in range(len(dates))]
    series = pd.Series(vals, index=dates, name="Portfolio")
    twr = pd.Series([1 + 0.0003 * i for i in range(len(dates))], index=dates, name="TWR")
    return (pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), [], series, twr)


class TestRunSingleBacktest:
    def test_returns_expected_keys(self):
        """Result dict has all required keys."""
        result = run_single_backtest(
            allocation={"SPY": 60.0, "BND": 40.0},
            maint_pcts={"SPY": 25.0, "BND": 25.0},
            rebalance={"mode": "Standard", "freq": "Yearly"},
            start_date="2020-01-02",
            end_date="2020-12-31",
            start_val=10000.0,
            cashflow_amount=0.0,
            cashflow_freq="Monthly",
            invest_div=True,
            pay_down_margin=False,
            tax_config={},
            bearer_token=None,
            name="Test",
            fetch_backtest_fn=_mock_fetch,
            run_shadow_fn=_mock_shadow,
        )
        expected_keys = {
            "name", "series", "port_series", "stats", "twr_series",
            "trades_df", "pl_by_year", "composition_df", "unrealized_pl_df",
            "component_prices", "allocation", "logs", "raw_response",
            "start_val", "sim_range", "shadow_range", "wmaint",
        }
        assert expected_keys.issubset(set(result.keys()))

    def test_name_propagated(self):
        """Portfolio name is propagated to result."""
        result = run_single_backtest(
            allocation={"SPY": 100.0},
            maint_pcts={},
            rebalance={"mode": "Standard", "freq": "Yearly"},
            start_date="2023-01-02",
            end_date="2023-06-30",
            start_val=5000.0,
            cashflow_amount=0.0,
            cashflow_freq="Monthly",
            invest_div=True,
            pay_down_margin=False,
            tax_config={},
            bearer_token=None,
            name="My Portfolio",
            fetch_backtest_fn=_mock_fetch,
            run_shadow_fn=_mock_shadow,
        )
        assert result["name"] == "My Portfolio"

    def test_wmaint_calculation(self):
        """Weighted maintenance is calculated from allocation."""
        result = run_single_backtest(
            allocation={"SPY": 50.0, "TQQQ": 50.0},
            maint_pcts={"SPY": 25.0, "TQQQ": 75.0},
            rebalance={"mode": "Standard", "freq": "Yearly"},
            start_date="2023-01-02",
            end_date="2023-03-31",
            start_val=10000.0,
            cashflow_amount=0.0,
            cashflow_freq="Monthly",
            invest_div=True,
            pay_down_margin=False,
            tax_config={},
            bearer_token=None,
            fetch_backtest_fn=_mock_fetch,
            run_shadow_fn=_mock_shadow,
        )
        # 50% * 25% + 50% * 75% = 50%
        assert abs(result["wmaint"] - 0.50) < 0.01

    def test_stats_populated(self):
        """Stats dict has standard fields from mock."""
        result = run_single_backtest(
            allocation={"SPY": 100.0},
            maint_pcts={},
            rebalance={"mode": "Standard", "freq": "Yearly"},
            start_date="2023-01-02",
            end_date="2023-06-30",
            start_val=10000.0,
            cashflow_amount=0.0,
            cashflow_freq="Monthly",
            invest_div=True,
            pay_down_margin=False,
            tax_config={},
            bearer_token=None,
            fetch_backtest_fn=_mock_fetch,
            run_shadow_fn=_mock_shadow,
        )
        assert "cagr" in result["stats"]
        assert "sharpe" in result["stats"]

"""Regression tests for serialized backtest result validation."""

import pandas as pd

from app.core.result_validation import has_stale_local_cashflow_series, has_stale_margin_results
from app.core.withdrawals import PERFORMANCE_CASHFLOW_POLICY


def test_detects_local_cashflow_series_cached_as_twr():
    dates = pd.bdate_range("2023-01-02", periods=6)
    twr = pd.Series([1.0, 1.01, 1.02, 1.03, 1.04, 1.05], index=dates)
    stale_series = twr * 100_000.0

    result = {
        "is_local": True,
        "series": stale_series,
        "twr_series": twr,
    }

    assert has_stale_local_cashflow_series(
        [result],
        {"amount": 1000.0, "pay_down_margin": False},
    )


def test_does_not_flag_money_weighted_local_cashflow_series():
    dates = pd.bdate_range("2023-01-02", periods=6)
    twr = pd.Series([1.0, 1.01, 1.02, 1.03, 1.04, 1.05], index=dates)
    money_weighted = (twr * 100_000.0) + pd.Series(
        [0.0, 0.0, 1000.0, 1000.0, 2000.0, 2000.0],
        index=dates,
    )

    result = {
        "is_local": True,
        "series": money_weighted,
        "twr_series": twr,
    }

    assert not has_stale_local_cashflow_series(
        [result],
        {"amount": 1000.0, "pay_down_margin": False},
    )


def test_old_retirement_results_refresh_even_when_they_contain_some_dca():
    config = {"draw_monthly_retirement": 3333.0}
    assert has_stale_margin_results([{"is_local": True}], config)
    assert has_stale_margin_results([{"is_local": False}], config)
    assert not has_stale_margin_results(
        [{"performance_cashflow_policy": PERFORMANCE_CASHFLOW_POLICY}], config,
    )
    assert not has_stale_margin_results([{}], {})


def test_current_baseline_before_first_deposit_does_not_refresh_forever():
    dates = pd.bdate_range("2023-01-02", periods=6)
    twr = pd.Series([1.0, 1.01, 1.02, 1.03, 1.04, 1.05], index=dates)
    result = {
        "is_local": True, "series": twr * 100_000.0, "twr_series": twr,
        "performance_cashflow_policy": PERFORMANCE_CASHFLOW_POLICY,
    }
    assert not has_stale_local_cashflow_series([result], {"amount": 1000.0})

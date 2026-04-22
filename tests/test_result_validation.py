"""Regression tests for serialized backtest result validation."""

import pandas as pd

from app.core.result_validation import has_stale_local_cashflow_series


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

"""Tests for app.core.monte_carlo."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.core.monte_carlo import run_monte_carlo, run_seasonal_monte_carlo


@pytest.fixture
def daily_returns():
    """252 days of small positive returns (~10% annual)."""
    np.random.seed(42)
    dates = pd.bdate_range("2023-01-02", periods=252)
    rets = np.random.normal(0.0004, 0.01, size=252)
    return pd.Series(rets, index=dates, name="returns")


# ---------------------------------------------------------------------------
# run_monte_carlo
# ---------------------------------------------------------------------------

def test_run_mc_empty():
    result = run_monte_carlo(pd.Series(dtype=float))
    assert result["percentiles"].empty
    assert result["metrics"] == {}


def test_run_mc_basic(daily_returns):
    result = run_monte_carlo(daily_returns, n_sims=100, n_years=1)

    df = result["percentiles"]
    assert not df.empty
    assert "P10" in df.columns
    assert "Median" in df.columns
    assert "P90" in df.columns
    # 1 year = 252 trading days + 1 start row = 253 rows
    assert len(df) == 253

    m = result["metrics"]
    assert "median_final" in m
    assert "cagr_median" in m
    assert "max_dd_median" in m
    assert "prob_loss" in m


def test_run_mc_with_cashflow(daily_returns):
    result_no_cf = run_monte_carlo(daily_returns, n_sims=50, n_years=1, monthly_cashflow=0)
    result_cf = run_monte_carlo(daily_returns, n_sims=50, n_years=1, monthly_cashflow=500)

    # With positive cashflows, total invested is higher so median final should generally be higher
    assert result_cf["metrics"]["total_invested"] > result_no_cf["metrics"]["total_invested"]


def test_run_mc_custom_params(daily_returns):
    result = run_monte_carlo(
        daily_returns, n_sims=50, n_years=1,
        custom_mean_annual=0.10, custom_vol_annual=0.15,
    )
    assert not result["percentiles"].empty
    assert result["metrics"]["median_final"] > 0


def test_run_mc_block_bootstrap(daily_returns):
    result = run_monte_carlo(daily_returns, n_sims=50, n_years=1, block_size=5)
    assert not result["percentiles"].empty
    assert len(result["percentiles"]) == 253


# ---------------------------------------------------------------------------
# run_seasonal_monte_carlo
# ---------------------------------------------------------------------------

def test_seasonal_mc_basic(daily_returns):
    result = run_seasonal_monte_carlo(daily_returns, n_sims=100)
    assert isinstance(result, pd.DataFrame)
    assert not result.empty
    # 12 months * 21 days + 1 start = 253 rows
    assert len(result) == 253
    assert "P10" in result.columns
    assert "Median" in result.columns
    assert "P90" in result.columns


def test_seasonal_mc_empty():
    result = run_seasonal_monte_carlo(pd.Series(dtype=float))
    assert isinstance(result, pd.DataFrame)
    assert result.empty

import pandas as pd
import numpy as np
import pytest


@pytest.fixture
def flat_prices():
    """DataFrame with constant $100 for TEST ticker, business days 2023."""
    dates = pd.bdate_range("2023-01-02", "2023-12-29")
    return pd.DataFrame({"TEST": 100.0}, index=dates)


@pytest.fixture
def growing_prices():
    """TEST goes $100 -> $200 linearly over 2020-2024."""
    dates = pd.bdate_range("2020-01-02", "2024-12-31")
    prices = np.linspace(100, 200, len(dates))
    return pd.DataFrame({"TEST": prices}, index=dates)


@pytest.fixture
def two_ticker_prices():
    """SPY (growing) + BND (flat) for rebalance testing."""
    dates = pd.bdate_range("2020-01-02", "2024-12-31")
    spy = np.linspace(100, 200, len(dates))
    bnd = np.full(len(dates), 100.0)
    return pd.DataFrame({"SPY": spy, "BND": bnd}, index=dates)


@pytest.fixture
def sample_allocation():
    """Simple 100% TEST allocation."""
    return {"TEST": 100.0}

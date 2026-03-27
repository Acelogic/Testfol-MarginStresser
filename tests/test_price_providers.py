"""Tests for app.services.price_providers module."""

import os
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from app.services.price_providers import (
    ChainedProvider,
    PolygonProvider,
    YFinanceProvider,
    get_price_provider,
    reset_provider,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _reset_singleton():
    """Reset the provider singleton before each test."""
    reset_provider()
    yield
    reset_provider()


def _make_price_df(tickers, n=5):
    """Helper to build a fake price DataFrame."""
    dates = pd.bdate_range("2024-01-02", periods=n)
    data = {t: range(100, 100 + n) for t in tickers}
    return pd.DataFrame(data, index=dates)


# ---------------------------------------------------------------------------
# YFinanceProvider
# ---------------------------------------------------------------------------

class TestYFinanceProvider:
    def test_is_available(self):
        p = YFinanceProvider()
        assert p.is_available() is True

    def test_name(self):
        assert YFinanceProvider.name == "yfinance"

    def test_fetch_single_ticker(self):
        hist = pd.DataFrame(
            {"Close": [100, 101, 102]},
            index=pd.bdate_range("2024-01-02", periods=3),
        )
        mock_ticker = MagicMock()
        mock_ticker.history.return_value = hist

        with patch("yfinance.Ticker", return_value=mock_ticker):
            p = YFinanceProvider()
            result = p.fetch_prices(["AAPL"], "2024-01-01", "2024-01-10")

        assert "AAPL" in result.columns
        assert len(result) == 3

    def test_fetch_empty_list(self):
        p = YFinanceProvider()
        result = p.fetch_prices([], "2024-01-01", "2024-12-31")
        assert result.empty


# ---------------------------------------------------------------------------
# PolygonProvider
# ---------------------------------------------------------------------------

class TestPolygonProvider:
    def test_is_available_no_key(self, monkeypatch):
        monkeypatch.delenv("POLYGON_API_KEY", raising=False)
        p = PolygonProvider(api_key=None)
        assert p.is_available() is False

    def test_is_available_with_key(self):
        p = PolygonProvider(api_key="test_key")
        assert p.is_available() is True

    def test_name(self):
        assert PolygonProvider.name == "polygon"

    def test_fetch_without_key_returns_empty(self, monkeypatch):
        monkeypatch.delenv("POLYGON_API_KEY", raising=False)
        p = PolygonProvider(api_key=None)
        result = p.fetch_prices(["AAPL"], "2024-01-01", "2024-12-31")
        assert result.empty

    def test_fetch_with_mock_client(self):
        """Test that Polygon provider correctly parses agg responses."""
        mock_agg = MagicMock()
        mock_agg.timestamp = 1704153600000  # 2024-01-02 UTC
        mock_agg.close = 150.0

        mock_client = MagicMock()
        mock_client.get_aggs.return_value = [mock_agg]

        with patch("polygon.RESTClient", return_value=mock_client):
            p = PolygonProvider(api_key="test_key", rate_limit_ms=0)
            result = p.fetch_prices(["AAPL"], "2024-01-01", "2024-01-10")

        assert not result.empty
        assert "AAPL" in result.columns


# ---------------------------------------------------------------------------
# ChainedProvider
# ---------------------------------------------------------------------------

class TestChainedProvider:
    def test_requires_at_least_one_available(self):
        """ChainedProvider raises if no providers are available."""
        mock_p = MagicMock()
        mock_p.is_available.return_value = False
        with pytest.raises(ValueError, match="No available"):
            ChainedProvider([mock_p])

    def test_per_ticker_fallback(self):
        """If primary misses a ticker, secondary fills it."""
        primary = MagicMock()
        primary.name = "primary"
        primary.is_available.return_value = True
        primary.fetch_prices.return_value = _make_price_df(["AAPL"])

        secondary = MagicMock()
        secondary.name = "secondary"
        secondary.is_available.return_value = True
        secondary.fetch_prices.return_value = _make_price_df(["MSFT"])

        chain = ChainedProvider([primary, secondary])
        result = chain.fetch_prices(["AAPL", "MSFT"], "2024-01-01", "2024-01-10")

        assert "AAPL" in result.columns
        assert "MSFT" in result.columns

    def test_primary_handles_all(self):
        """If primary has all tickers, secondary is not called with those."""
        primary = MagicMock()
        primary.name = "primary"
        primary.is_available.return_value = True
        primary.fetch_prices.return_value = _make_price_df(["AAPL", "MSFT"])

        secondary = MagicMock()
        secondary.name = "secondary"
        secondary.is_available.return_value = True

        chain = ChainedProvider([primary, secondary])
        result = chain.fetch_prices(["AAPL", "MSFT"], "2024-01-01", "2024-01-10")

        assert "AAPL" in result.columns
        assert "MSFT" in result.columns
        # Secondary should be called with empty remaining list
        if secondary.fetch_prices.called:
            args = secondary.fetch_prices.call_args
            assert args[0][0] == []  # empty ticker list

    def test_primary_failure_falls_through(self):
        """If primary raises, secondary is tried."""
        primary = MagicMock()
        primary.name = "primary"
        primary.is_available.return_value = True
        primary.fetch_prices.side_effect = Exception("API down")

        secondary = MagicMock()
        secondary.name = "secondary"
        secondary.is_available.return_value = True
        secondary.fetch_prices.return_value = _make_price_df(["AAPL"])

        chain = ChainedProvider([primary, secondary])
        result = chain.fetch_prices(["AAPL"], "2024-01-01", "2024-01-10")

        assert "AAPL" in result.columns

    def test_name_reflects_chain(self):
        p1 = MagicMock()
        p1.name = "polygon"
        p1.is_available.return_value = True

        p2 = MagicMock()
        p2.name = "yfinance"
        p2.is_available.return_value = True

        chain = ChainedProvider([p1, p2])
        assert chain.name == "polygon+yfinance"

    def test_unavailable_providers_filtered(self):
        """Unavailable providers are excluded from the chain."""
        p1 = MagicMock()
        p1.name = "unavailable"
        p1.is_available.return_value = False

        p2 = MagicMock()
        p2.name = "yfinance"
        p2.is_available.return_value = True

        chain = ChainedProvider([p1, p2])
        assert chain.name == "yfinance"


# ---------------------------------------------------------------------------
# Factory: get_price_provider
# ---------------------------------------------------------------------------

class TestGetPriceProvider:
    def test_default_yfinance_only(self, monkeypatch):
        monkeypatch.delenv("POLYGON_API_KEY", raising=False)
        provider = get_price_provider()
        assert "yfinance" in provider.name

    def test_polygon_when_key_set(self, monkeypatch):
        monkeypatch.setenv("POLYGON_API_KEY", "test_key")
        provider = get_price_provider()
        assert "polygon" in provider.name
        assert "yfinance" in provider.name

    def test_singleton(self, monkeypatch):
        monkeypatch.delenv("POLYGON_API_KEY", raising=False)
        p1 = get_price_provider()
        p2 = get_price_provider()
        assert p1 is p2


# ---------------------------------------------------------------------------
# Orchestrator failover
# ---------------------------------------------------------------------------

class TestOrchestratorFailover:
    """Test that backtest_orchestrator falls back to local engine on API failure."""

    def test_api_failure_triggers_local_engine(self):
        """When fetch_fn raises ConnectionError, orchestrator uses local engine."""
        import requests
        from app.core.backtest_orchestrator import run_single_backtest

        def failing_fetch(**kwargs):
            raise requests.exceptions.ConnectionError("Testfol is down")

        # Minimal allocation for a simple test
        result = run_single_backtest(
            allocation={"SPY": 100.0},
            maint_pcts={"SPY": 25.0},
            rebalance={"mode": "Standard", "freq": "Yearly"},
            start_date="2024-01-01",
            end_date="2024-03-01",
            start_val=10000,
            cashflow_amount=0,
            cashflow_freq="Monthly",
            invest_div=True,
            pay_down_margin=False,
            tax_config={},
            bearer_token=None,
            name="Test",
            fetch_backtest_fn=failing_fetch,
        )

        # Should succeed via local engine
        assert result is not None
        assert result.get("is_local") is True
        assert not result["series"].empty

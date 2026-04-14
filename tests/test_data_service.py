"""Tests for app.services.data_service performance-sensitive fetch paths."""

from unittest.mock import MagicMock

import pandas as pd

import app.common.cache as cache_mod
from app.services import data_service


def _make_prices(tickers, start="2024-01-02", periods=4):
    dates = pd.bdate_range(start, periods=periods)
    return pd.DataFrame(
        {ticker: range(100, 100 + periods) for ticker in tickers},
        index=dates,
    )


def test_fetch_component_data_batches_standard_provider_requests(monkeypatch, tmp_path):
    monkeypatch.setattr(cache_mod, "CACHE_DIR", str(tmp_path))

    provider = MagicMock()
    provider.fetch_prices.return_value = _make_prices(["SPY", "BND"])
    monkeypatch.setattr(data_service, "get_price_provider", lambda: provider)

    api_fetch = MagicMock(side_effect=AssertionError("API fallback should not run"))
    monkeypatch.setattr(data_service.api, "fetch_backtest", api_fetch)

    result = data_service.fetch_component_data(
        ["SPY", "BND"],
        "2024-01-02",
        "2024-01-10",
    )

    assert list(result.columns) == ["SPY", "BND"]
    assert provider.fetch_prices.call_count == 1

    args = provider.fetch_prices.call_args.args
    assert set(args[0]) == {"SPY", "BND"}
    assert args[1:] == ("2024-01-02", "2024-01-10")


def test_fetch_component_data_reuses_cached_combined_result(monkeypatch, tmp_path):
    monkeypatch.setattr(cache_mod, "CACHE_DIR", str(tmp_path))

    provider = MagicMock()
    provider.fetch_prices.return_value = _make_prices(["SPY", "QQQ"])
    monkeypatch.setattr(data_service, "get_price_provider", lambda: provider)

    api_fetch = MagicMock(side_effect=AssertionError("API fallback should not run"))
    monkeypatch.setattr(data_service.api, "fetch_backtest", api_fetch)

    first = data_service.fetch_component_data(
        ["SPY", "QQQ"],
        "2024-01-02",
        "2024-01-10",
    )
    second = data_service.fetch_component_data(
        ["SPY", "QQQ"],
        "2024-01-02",
        "2024-01-10",
    )

    assert provider.fetch_prices.call_count == 1
    pd.testing.assert_frame_equal(first, second)


def test_fetch_component_data_clips_trailing_unsynced_components(monkeypatch, tmp_path):
    monkeypatch.setattr(cache_mod, "CACHE_DIR", str(tmp_path))

    dates = pd.bdate_range("2024-01-02", periods=4)
    provider_prices = pd.DataFrame(
        {
            "SPY": [100.0, 101.0, 102.0, None],
            "QQQ": [100.0, 101.0, 102.0, 103.0],
        },
        index=dates,
    )
    provider = MagicMock()
    provider.fetch_prices.return_value = provider_prices
    monkeypatch.setattr(data_service, "get_price_provider", lambda: provider)

    api_fetch = MagicMock(side_effect=AssertionError("API fallback should not run"))
    monkeypatch.setattr(data_service.api, "fetch_backtest", api_fetch)

    result = data_service.fetch_component_data(
        ["SPY", "QQQ"],
        "2024-01-02",
        "2024-01-10",
    )

    assert result.index.max() == dates[2]
    assert result.loc[dates[2], "QQQ"] == 102.0


def test_fetch_component_data_can_return_raw_unsynced_components(monkeypatch, tmp_path):
    monkeypatch.setattr(cache_mod, "CACHE_DIR", str(tmp_path))

    dates = pd.bdate_range("2024-01-02", periods=4)
    provider_prices = pd.DataFrame(
        {
            "SPY": [100.0, 101.0, 102.0, None],
            "QQQ": [100.0, 101.0, 102.0, 103.0],
        },
        index=dates,
    )
    provider = MagicMock()
    provider.fetch_prices.return_value = provider_prices
    monkeypatch.setattr(data_service, "get_price_provider", lambda: provider)

    api_fetch = MagicMock(side_effect=AssertionError("API fallback should not run"))
    monkeypatch.setattr(data_service.api, "fetch_backtest", api_fetch)

    result = data_service.fetch_component_data(
        ["SPY", "QQQ"],
        "2024-01-02",
        "2024-01-10",
        sync_end=False,
    )

    assert result.index.max() == dates[3]


def test_fetch_component_data_syncs_only_inside_requested_window(monkeypatch, tmp_path):
    monkeypatch.setattr(cache_mod, "CACHE_DIR", str(tmp_path))

    dates = pd.bdate_range("2024-01-02", periods=6)
    provider_prices = pd.DataFrame(
        {
            "SPY": [100.0, 101.0, 102.0, 103.0, None, None],
            "QQQ": [100.0, 101.0, 102.0, 103.0, 104.0, 105.0],
        },
        index=dates,
    )
    provider = MagicMock()
    provider.fetch_prices.return_value = provider_prices
    monkeypatch.setattr(data_service, "get_price_provider", lambda: provider)

    api_fetch = MagicMock(side_effect=AssertionError("API fallback should not run"))
    monkeypatch.setattr(data_service.api, "fetch_backtest", api_fetch)

    result = data_service.fetch_component_data(
        ["SPY", "QQQ"],
        "2024-01-02",
        dates[3].strftime("%Y-%m-%d"),
    )

    assert result.index.max() == dates[3]
    assert result.loc[dates[3], "SPY"] == 103.0
    assert result.loc[dates[3], "QQQ"] == 103.0

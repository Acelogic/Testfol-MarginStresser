"""Tests for app.services.xray_engine."""
from __future__ import annotations

import pandas as pd
import pytest

from app.services.xray_engine import parse_leverage, compute_xray


# ---------------------------------------------------------------------------
# parse_leverage
# ---------------------------------------------------------------------------

def test_parse_leverage_static_map():
    assert parse_leverage("TQQQ") == 3.0
    assert parse_leverage("SSO") == 2.0
    assert parse_leverage("UPRO") == 3.0


def test_parse_leverage_param():
    assert parse_leverage("SPY?L=2") == 2.0
    assert parse_leverage("AVGO?L=1.5") == 1.5


def test_parse_leverage_default():
    assert parse_leverage("AAPL") == 1.0
    assert parse_leverage("GOOG") == 1.0


def test_parse_leverage_suffix():
    assert parse_leverage("FUND-3X") == 3.0


# ---------------------------------------------------------------------------
# compute_xray
# ---------------------------------------------------------------------------

def test_compute_xray_direct(monkeypatch):
    """When get_etf_holdings returns None, treat ticker as direct holding."""
    monkeypatch.setattr(
        "app.services.xray_engine.etf_holdings_fetcher.get_etf_holdings",
        lambda ticker: None,
    )
    result = compute_xray({"AAPL": 0.6, "GOOG": 0.4})

    assert isinstance(result, pd.DataFrame)
    assert len(result) == 2
    assert abs(result["Weight"].sum() - 1.0) < 0.01
    # AAPL should be the heaviest
    assert result.iloc[0]["Ticker"] == "AAPL"


def test_compute_xray_etf(monkeypatch):
    """When get_etf_holdings returns holdings, weights are decomposed."""
    def mock_holdings(ticker):
        if ticker == "QQQ":
            return pd.DataFrame({
                "name": ["Apple Inc.", "Microsoft Corp"],
                "ticker": ["AAPL", "MSFT"],
                "weight": [0.12, 0.10],
            })
        return None

    monkeypatch.setattr(
        "app.services.xray_engine.etf_holdings_fetcher.get_etf_holdings",
        mock_holdings,
    )

    result = compute_xray({"QQQ": 1.0})
    assert isinstance(result, pd.DataFrame)
    assert not result.empty

    # AAPL weight should be 1.0 * 0.12 = 0.12
    aapl_row = result[result["Ticker"] == "AAPL"]
    assert not aapl_row.empty
    assert aapl_row.iloc[0]["Weight"] == pytest.approx(0.12, abs=0.001)

    msft_row = result[result["Ticker"] == "MSFT"]
    assert not msft_row.empty
    assert msft_row.iloc[0]["Weight"] == pytest.approx(0.10, abs=0.001)

"""Tests for api.routes (FastAPI endpoints)."""
from __future__ import annotations

from unittest.mock import patch

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from api.main import app

client = TestClient(app)


# ---------------------------------------------------------------------------
# /api/backtest
# ---------------------------------------------------------------------------

def test_backtest_endpoint(monkeypatch):
    """POST /api/backtest returns serialized BacktestResult."""
    fake_raw = {
        "name": "TestPort",
        "series": pd.Series([10000, 10100], index=pd.bdate_range("2023-01-02", periods=2)),
        "stats": {"cagr": 0.08},
        "twr_series": None,
        "daily_returns_df": None,
        "trades_df": None,
        "pl_by_year": None,
        "composition_df": None,
        "unrealized_pl_df": None,
        "component_prices": None,
        "allocation": {"SPY": 100.0},
        "logs": [],
        "raw_response": {},
        "is_local": False,
        "start_val": 10000.0,
        "sim_range": "",
        "shadow_range": "",
        "wmaint": 0.25,
    }

    # Bypass cache
    monkeypatch.setattr("api.routes.backtest.cache_get", lambda *a, **kw: None)
    monkeypatch.setattr("api.routes.backtest.cache_set", lambda *a, **kw: None)

    with patch("api.routes.backtest.run_single_backtest", return_value=fake_raw):
        resp = client.post("/api/backtest", json={
            "portfolio": {"name": "TestPort", "allocation": {"SPY": 100.0}},
            "start_date": "2023-01-01",
            "end_date": "2023-12-31",
        })

    assert resp.status_code == 200
    data = resp.json()
    assert data["name"] == "TestPort"
    assert data["stats"]["cagr"] == 0.08


# ---------------------------------------------------------------------------
# /api/backtest/multi
# ---------------------------------------------------------------------------

def test_multi_backtest_endpoint(monkeypatch):
    """POST /api/backtest/multi returns MultiBacktestResponse."""
    idx = pd.bdate_range("2023-01-02", periods=3)
    fake_result = {
        "name": "Port1",
        "series": pd.Series([10000, 10050, 10100], index=idx),
        "stats": {},
        "twr_series": None, "daily_returns_df": None, "trades_df": None,
        "pl_by_year": None, "composition_df": None, "unrealized_pl_df": None,
        "component_prices": None, "allocation": {"SPY": 100.0},
        "logs": [], "raw_response": {}, "is_local": False,
        "start_val": 10000.0, "sim_range": "", "shadow_range": "", "wmaint": 0.25,
    }

    monkeypatch.setattr("api.routes.backtest.cache_get", lambda *a, **kw: None)
    monkeypatch.setattr("api.routes.backtest.cache_set", lambda *a, **kw: None)

    with patch("api.routes.backtest.run_multi_backtest", return_value=([fake_result], [])):
        resp = client.post("/api/backtest/multi", json={
            "portfolios": [{"name": "Port1", "allocation": {"SPY": 100.0}}],
            "start_date": "2023-01-01",
            "end_date": "2023-12-31",
        })

    assert resp.status_code == 200
    data = resp.json()
    assert len(data["results"]) == 1
    assert data["results"][0]["name"] == "Port1"


# ---------------------------------------------------------------------------
# /api/data/prices
# ---------------------------------------------------------------------------

def test_prices_endpoint(monkeypatch):
    """POST /api/data/prices returns prices_json."""
    fake_df = pd.DataFrame(
        {"SPY": [100.0, 101.0]},
        index=pd.bdate_range("2023-01-02", periods=2),
    )

    with patch("api.routes.data.fetch_component_data", return_value=fake_df):
        resp = client.post("/api/data/prices", json={
            "tickers": ["SPY"],
            "start_date": "2023-01-01",
            "end_date": "2023-12-31",
        })

    assert resp.status_code == 200
    assert resp.json()["prices_json"] is not None


# ---------------------------------------------------------------------------
# /api/data/fed-funds
# ---------------------------------------------------------------------------

def test_fed_funds_endpoint(monkeypatch):
    """GET /api/data/fed-funds returns fed_funds_json."""
    fake_series = pd.Series(
        [5.25, 5.25],
        index=pd.date_range("2023-01-01", periods=2),
    )

    with patch("api.routes.data.get_fed_funds_rate", return_value=fake_series):
        resp = client.get("/api/data/fed-funds")

    assert resp.status_code == 200
    assert resp.json()["fed_funds_json"] is not None

"""Retirement affects margin usage without changing comparison compounding."""

import datetime as dt

import pandas as pd
import pytest
import requests
from streamlit.testing.v1 import AppTest

from app.core import backtest_orchestrator as orchestrator
from app.core.calculations import generate_stats
from app.core.shadow_backtest import run_shadow_backtest
from app.core.withdrawals import dca_stop_date
from app.services.testfol_api import simulate_margin


@pytest.fixture
def run_portfolios(monkeypatch):
    dates = pd.bdate_range("1994-12-30", "1997-12-31")
    prices = pd.DataFrame({"TEST": 100.0, "LATE": 100.0}, index=dates)
    prices.loc[prices.index < "1996-01-02", "LATE"] = float("nan")

    def component_prices(tickers, start_date, end_date):
        return prices.loc[start_date:end_date, list(tickers)]

    def shadow(**kwargs):
        kwargs["prices_df"] = prices[list(kwargs["allocation"])]
        return run_shadow_backtest(**kwargs)

    def fetch(**kwargs):
        raw = shadow(
            allocation=kwargs["allocation"],
            start_date=kwargs["start_date"], end_date=kwargs["end_date"],
            start_val=kwargs["start_val"], cashflow=kwargs["cashflow"],
            cashflow_freq=kwargs["cashfreq"],
        )
        return raw[5], generate_stats(raw[6]), {}

    monkeypatch.setattr(orchestrator, "fetch_component_data", component_prices)
    monkeypatch.setattr(orchestrator, "_orchestrator_worker_count", lambda count: 1)

    def run(pm_config=None, engine="local", common_start=False):
        fetch_fn = fetch
        if engine == "failover":
            def fetch_fn(**kwargs):
                raise requests.ConnectionError("offline test")
        tickers = ["TEST", "LATE"] if common_start else ["TEST"]
        results, _ = orchestrator.run_multi_backtest(
            portfolios=[{
                "name": ticker, "allocation": {ticker: 100.0},
                "maint_pcts": {ticker: 25.0},
                "rebalance": {"mode": "None" if engine == "local" else "Standard", "freq": "Yearly"},
            } for ticker in tickers],
            start_date="1995-01-02", end_date="1997-12-31",
            start_val=100_000.0, cashflow_amount=1_000.0, cashflow_freq="Monthly",
            invest_div=True, pay_down_margin=False, tax_config={}, bearer_token=None,
            fetch_backtest_fn=fetch_fn, run_shadow_fn=shadow, pm_config=pm_config,
        )
        return results

    return run


@pytest.mark.parametrize("engine", ["local", "api", "failover"])
@pytest.mark.parametrize("common_start", [False, True])
def test_retirement_keeps_baseline_contributions_and_changes_only_margin(run_portfolios, engine, common_start):
    retirement = dt.date(1996, 7, 1)
    baseline = run_portfolios(engine=engine, common_start=common_start)
    retirement_config = {
        "draw_monthly_retirement": 3_333.0,
        "retirement_date": retirement,
        "dca_in_retirement": False,
    }
    with_retirement = run_portfolios(retirement_config, engine, common_start)

    for before, after in zip(baseline, with_retirement):
        pd.testing.assert_series_equal(before["series"], after["series"])
        pd.testing.assert_series_equal(before["twr_series"], after["twr_series"])
        pd.testing.assert_frame_equal(before["trades_df"], after["trades_df"])
        assert before["stats"] == after["stats"]

        margin = after["margin_result"]
        # No start-of-backtest cutoff: contributions continue until retirement.
        pd.testing.assert_series_equal(
            after["series"].loc[:"1996-06-30"],
            margin["series"].loc[:"1996-06-30"],
        )
        assert margin["series"].iloc[-1] < after["series"].iloc[-1]
        assert margin["series"].loc["1996-07-01":].nunique() == 1
        assert margin["effective_start_date"] == after["effective_start_date"]

        loan, equity, _, usage, _ = simulate_margin(
            margin["series"], 0.0, 0.0, 0.0, 0.25,
            draw_monthly_retirement=3_333.0, retirement_date=retirement,
        )
        assert loan.loc[:"1996-06-30"].eq(0).all()
        assert loan.iloc[-1] > 0
        assert usage.iloc[-1] > 0
        assert equity.iloc[-1] < margin["series"].iloc[-1]


def test_continuing_dca_and_future_retirement_leave_baseline_unchanged(run_portfolios):
    baseline = run_portfolios()[0]
    for continue_dca, retirement in ((True, "1996-07-01"), (False, "2001-01-01")):
        result = run_portfolios({
            "draw_monthly_retirement": 3_333.0,
            "retirement_date": retirement,
            "dca_in_retirement": continue_dca,
        })[0]
        pd.testing.assert_series_equal(baseline["series"], result["series"])
        if continue_dca:
            pd.testing.assert_series_equal(baseline["series"], result["margin_result"]["series"])
            assert any("RetDraw" in line for line in result["margin_result"]["logs"])
        else:
            assert "margin_result" not in result


def test_pm_buy_block_does_not_change_baseline(run_portfolios):
    baseline = run_portfolios(engine="api")[0]
    result = run_portfolios({
        "pm_buy_block": True,
        "pm_buy_block_threshold": 1_000_000.0,
        "draw_monthly_retirement": 3_333.0,
        "retirement_date": "1996-07-01",
        "dca_in_retirement": False,
    }, engine="api")[0]
    pd.testing.assert_series_equal(baseline["series"], result["series"])
    assert result["pm_blocked_dates"] == []
    assert result["margin_result"]["pm_blocked_dates"]


def test_dca_cutoff_uses_active_draws_only():
    assert dca_stop_date(0, "1990-01-01", 3333, "1996-07-01", "1995-01-02") == dt.date(1996, 7, 1)
    assert dca_stop_date(1000, None, 3333, "1996-07-01", "1995-01-02") == dt.date(1995, 1, 2)
    assert dca_stop_date(1000, "1997-01-01", 3333, "1996-07-01", "1995-01-02") == dt.date(1996, 7, 1)
    assert dca_stop_date(0, "1990-01-01", 0, "1996-07-01", "1995-01-02") is None


@pytest.mark.parametrize("view", ["📈 Chart", "📊 Returns Analysis", "💰 Withdrawals"])
def test_ui_keeps_comparison_and_compounding_on_baseline(run_portfolios, view):
    retirement_config = {
        "draw_monthly_retirement": 3_333.0,
        "retirement_date": dt.date(1996, 7, 1),
        "dca_in_retirement": False,
        "rate_annual": 0.0,
        "global_cashflow": {"amount": 1000.0, "freq": "Monthly"},
    }
    result = run_portfolios(retirement_config)[0]
    app = AppTest.from_string(
        """
import streamlit as st
from unittest.mock import patch
from app.ui import results, charts

def capture_plot(fig, **kwargs):
    st.session_state["comparison_plots"].append(list(fig.data[0].y))

def capture_margin(*args, **kwargs):
    st.session_state["margin_chart"] = kwargs

def capture_returns(series, **kwargs):
    st.session_state["performance_returns"] = series

def capture_withdrawals(tab, logs, draw_monthly, draw_start_date, **kwargs):
    st.session_state["withdrawal_logs"] = logs
    st.session_state["withdrawal_amount"] = draw_monthly

st.session_state["comparison_plots"] = []
result = st.session_state["test_result"]
config = st.session_state["test_config"]
with (
    patch("streamlit.segmented_control", return_value=st.session_state["test_view"]),
    patch("streamlit.plotly_chart", side_effect=capture_plot),
    patch.object(results, "render_chart_tab", side_effect=capture_margin),
    patch.object(charts, "render_returns_analysis", side_effect=capture_returns),
    patch("app.ui.results.tabs_withdrawals.render_withdrawals_tab", side_effect=capture_withdrawals),
    patch.object(results.report_generator, "generate_html_report", return_value="<html></html>"),
):
    charts.render_multi_portfolio_chart([result], cashflow_config=config["global_cashflow"])
    results.render(result, config)
""",
        default_timeout=20,
    )
    app.session_state["test_result"] = result
    app.session_state["test_config"] = retirement_config
    app.session_state["test_view"] = view
    app.run()
    assert not app.exception, [exc.message for exc in app.exception]
    assert app.session_state["comparison_plots"][0][-1] == result["series"].iloc[-1]
    portfolio_value = next(metric.value for metric in app.metric if metric.label == "Portfolio Value")
    assert portfolio_value == f"${result['series'].iloc[-1]:,.0f}"
    if view == "📈 Chart":
        chart = app.session_state["margin_chart"]
        assert chart["port_series"].iloc[-1] == result["margin_result"]["series"].iloc[-1]
        assert chart["loan_series"].iloc[-1] > 0
        assert chart["tax_adj_usage_series"].iloc[-1] > 0
    elif view == "📊 Returns Analysis":
        assert app.session_state["performance_returns"].iloc[-1] == result["series"].iloc[-1]
    else:
        assert any("RetDraw" in line for line in app.session_state["withdrawal_logs"])
        assert app.session_state["withdrawal_amount"] == 3333.0


def test_missing_margin_prices_does_not_display_a_fabricated_flat_curve(run_portfolios, monkeypatch):
    original = orchestrator.run_single_backtest

    def missing_scenario(**kwargs):
        if kwargs.get("margin_scenario"):
            return {"series": pd.Series(dtype=float)}
        return original(**kwargs)

    monkeypatch.setattr(orchestrator, "run_single_backtest", missing_scenario)
    with pytest.raises(ValueError, match="No price data available"):
        run_portfolios({"draw_monthly": 1000.0})


def test_serialization_preserves_separate_margin_result(run_portfolios):
    import ast
    from pathlib import Path
    from api.routes.backtest import _serialize_result

    result = run_portfolios({
        "draw_monthly_retirement": 3333.0,
        "retirement_date": "1996-07-01",
        "dca_in_retirement": False,
    })[0]
    payload = _serialize_result(result).model_dump(mode="json")

    # Load the app's deserializers without running its interactive entry point.
    source = Path(__file__).resolve().parents[1] / "testfol_charting.py"
    tree = ast.parse(source.read_text())
    helpers = ast.Module(body=[
        node for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name in {"_deser_series", "_deser_df", "_deserialize_result"}
    ], type_ignores=[])
    namespace = {"pd": pd}
    exec(compile(helpers, str(source), "exec"), namespace)
    restored = namespace["_deserialize_result"](payload)

    assert restored["performance_cashflow_policy"] == result["performance_cashflow_policy"]
    assert restored["series"].iloc[-1] == result["series"].iloc[-1]
    assert restored["margin_result"]["series"].iloc[-1] == result["margin_result"]["series"].iloc[-1]
    assert restored["margin_result"]["series"].iloc[-1] < restored["series"].iloc[-1]

#!/usr/bin/env python3
"""Exercise the complete Streamlit app and write a repeatable JSON smoke artifact.

Failure modes considered before implementing this harness:
* App imports, widget creation, or a rerun fail under the installed Streamlit.
* Date/segmented-control edits disappear or fail to reach the real backtest.
* Run succeeds visually but returns empty, non-finite, or non-local results.
* A results view throws, emits st.error, or destroys the computed result.
* Cache reuse hides a failure, or verification modifies the user's data caches.
* An unintended API/auth/provider request makes the check depend on live data.
* A failed check exits successfully or leaves no inspectable evidence.

Qualification: this is an app-level AppTest check, not browser pixel/layout or
live-service verification. The entry point, widgets, orchestration, local engine,
and result renderers are real. The portfolio uses the existing NDXMEGASIM CSV.
Ancillary asset-explorer/scanner/rolling-benchmark inputs use that same explicitly
synthetic fixture. FRED reads are local-only, credentials are disabled, and disk caches are
temporary. No Testfol request is permitted. Run from any working directory:

    .venv/bin/python scripts/verify_streamlit_upgrade.py --output path/report.json
"""

from __future__ import annotations

import argparse
from contextlib import ExitStack
from datetime import date, datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import socket
import sys
import tempfile
import traceback
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path,
        default=ROOT / "artifacts/streamlit-upgrade/app-smoke.json",
    )
    parser.add_argument("--timeout", type=int, default=120)
    args = parser.parse_args()
    output = args.output.expanduser().resolve()
    report = {
        "schema_version": 1,
        "started_at": datetime.now(timezone.utc).isoformat(),
        "python": platform.python_version(),
        "entry_point": "testfol_charting.py",
        "harness_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "qualification": "Full-entry AppTest with real local engine; no browser pixels or live API proof",
        "fixtures": [
            "Local NDXMEGASIM CSV for portfolio and ancillary market panels/rolling benchmark",
            "Local-only FRED CSV reads; synthetic one-member scanner universe",
            "No authentication; backend health unavailable; isolated temporary disk cache",
        ],
        "assertions": [],
        "views": [],
        "network": {"blocked_requests": [], "testfol_calls": 0},
        "passed": False,
    }

    def check(name, condition, detail=None):
        record = {"name": name, "passed": bool(condition)}
        if detail is not None:
            record["detail"] = detail
        report["assertions"].append(record)
        if not condition:
            raise AssertionError(f"{name}: {detail}")

    try:
        import numpy as np
        import pandas as pd
        import requests
        import streamlit as st
        from streamlit.testing.v1 import AppTest
        from app.common import cache
        from app.services import data_service, testfol_api, testfol_auth
        from app.ui import asset_explorer, ndx_scanner
        from app.ui.charts import rolling

        report["streamlit"] = st.__version__
        source = ROOT / "data/NDXMEGASIM.csv"
        source_bytes = source.read_bytes()
        local = pd.read_csv(source, parse_dates=["Date"]).set_index("Date")["Close"].sort_index()
        fixture = local.loc["2018-01-01":"2021-12-31"]
        report["input"] = {
            "path": str(source.relative_to(ROOT)),
            "sha256": hashlib.sha256(source_bytes).hexdigest(),
            "fixture_start": fixture.index[0].isoformat(),
            "fixture_end": fixture.index[-1].isoformat(),
            "fixture_rows": len(fixture),
            "allocation": {"NDXMEGASIM": 100.0},
            "start_date": "2020-01-02",
            "end_date": "2020-12-31",
            "starting_value": 10000.0,
            "cashflow": 0.0,
        }
        check("local_fixture_available", len(fixture) > 500 and fixture.notna().all())

        def blocked_request(_session, method, url, *a, **kw):
            from urllib.parse import urlsplit
            parsed = urlsplit(str(url))
            target = f"{parsed.scheme}://{parsed.netloc}{parsed.path}"
            report["network"]["blocked_requests"].append(target)
            raise requests.ConnectionError("Network disabled by Streamlit smoke harness")

        def blocked_socket(*a, **kw):
            raise OSError("Network disabled by Streamlit smoke harness")

        def forbidden_testfol(*a, **kw):
            report["network"]["testfol_calls"] += 1
            report["network"].setdefault("testfol_callers", []).append([
                f"{Path(frame.filename).name}:{frame.lineno}:{frame.name}"
                for frame in traceback.extract_stack(limit=10)
            ])
            raise AssertionError("Unexpected Testfol request during local-only smoke")

        def local_fred(series_id, filename, **kw):
            path = ROOT / "data" / filename
            if not path.is_file():
                return None
            table = pd.read_csv(path)
            values = pd.to_numeric(table.iloc[:, 1], errors="coerce")
            return pd.Series(values.to_numpy(), index=pd.to_datetime(table.iloc[:, 0])).dropna()

        def scanner_download(tickers, *a, **kw):
            tickers = [tickers] if isinstance(tickers, str) else list(tickers)
            frame = pd.DataFrame({ticker: fixture for ticker in tickers})
            return pd.concat({"Close": frame}, axis=1)

        def check_render(app, phase):
            exceptions = [str(item.message) for item in app.exception]
            errors = [str(item.value) for item in app.error]
            check(f"{phase}_no_exceptions", not exceptions, exceptions)
            check(f"{phase}_no_errors", not errors, errors)

        with tempfile.TemporaryDirectory(prefix="testfol-streamlit-smoke-") as temp, ExitStack() as stack:
            original_cwd = Path.cwd()
            os.chdir(ROOT)
            stack.callback(os.chdir, original_cwd)
            stack.enter_context(patch.object(cache, "CACHE_DIR", temp))
            stack.enter_context(patch.object(testfol_auth, "get_token", return_value=None))
            stack.enter_context(patch.object(testfol_api, "fetch_backtest", side_effect=forbidden_testfol))
            stack.enter_context(patch.object(data_service, "_get_fred_series", side_effect=local_fred))
            stack.enter_context(patch.object(requests.sessions.Session, "request", blocked_request))
            stack.enter_context(patch.object(socket.socket, "connect", blocked_socket))
            stack.enter_context(patch.object(socket.socket, "connect_ex", blocked_socket))
            stack.enter_context(patch.object(ndx_scanner.yf, "download", side_effect=scanner_download))
            stack.enter_context(patch.object(rolling, "_fetch_spysim_series", return_value=fixture.copy()))
            stack.enter_context(patch.object(ndx_scanner, "get_current_ndx_components", return_value=(
                pd.DataFrame({"Ticker": ["NDXMEGASIM"], "Name": ["Local smoke fixture"], "Weight": [100.0]}),
                pd.Timestamp("2020-12-31"), "Smoke fixture", "One local series", None,
            )))
            # yfinance can use curl directly, bypassing Python sockets.
            import curl_cffi.requests
            stack.enter_context(patch.object(curl_cffi.requests.Session, "request", blocked_request))
            stack.enter_context(patch.dict(os.environ, {
                "TESTFOL_EMAIL": "", "TESTFOL_PASSWORD": "", "TESTFOL_API_KEY": "",
                "POLYGON_API_KEY": "", "TESTFOL_ORCHESTRATOR_WORKERS": "1",
            }))
            st.cache_data.clear()
            st.cache_resource.clear()
            app = AppTest.from_file(str(ROOT / "testfol_charting.py"), default_timeout=args.timeout)
            app.session_state["margin_rate_model"] = "Fixed"
            app.session_state["ae_cache"] = {ticker: fixture.copy() for ticker in asset_explorer.ASSET_CLASSES.values()}
            app.session_state["portfolios"] = [{
                "id": "smoke", "name": "Local smoke portfolio",
                "alloc_df": pd.DataFrame([{
                    "Ticker": "NDXMEGASIM", "Weight %": 100.0,
                    "Maint %": 25.0, "PM Maint %": 15.0,
                }]),
                "rebalance": {"mode": "None", "freq": "Yearly", "compare_std": False},
                "dca": {"mode": "Proportional", "target_ticker": ""},
            }]
            app.run()
            check_render(app, "initial_render")
            report["configuration_tabs"] = [tab.label for tab in app.tabs]
            check("configuration_tabs_present", len(app.tabs) >= 5, report["configuration_tabs"])

            next(w for w in app.date_input if w.label == "Start Date").set_value(date(2020, 1, 2))
            next(w for w in app.date_input if w.label == "End Date").set_value(date(2020, 12, 31))
            app.radio(key="p_rmode_smoke").set_value("None")
            app.run()
            check_render(app, "configuration_rerun")
            check("date_widgets_preserved", [w.value for w in app.date_input if w.label in {"Start Date", "End Date"}] == [date(2020, 1, 2), date(2020, 12, 31)])
            next(w for w in app.button if w.label == "🚀 Run Backtest").click().run()
            check_render(app, "run_backtest")
            results = app.session_state["results_list"]
            check("one_result", len(results) == 1)
            result = results[0]
            series = result["series"]
            check("real_local_engine", result["is_local"] and not result["api_failover"])
            check("finite_result", len(series) > 200 and np.isfinite(series).all())
            check("requested_end_date", series.index[-1].date() == date(2020, 12, 31))
            check("requested_start_date", result["effective_start_date"].date() == date(2020, 1, 2))
            # Independent buy-and-hold oracle for this single-asset, no-flow case.
            anchor = local.loc[:"2020-01-01"].iloc[-1]
            expected_end = 10000.0 * local.loc["2020-12-31"] / anchor
            check("buy_and_hold_value", np.isclose(series.iloc[-1], expected_end, rtol=1e-8), {
                "actual": float(series.iloc[-1]), "expected": float(expected_end),
            })
            initial_fingerprint = hashlib.sha256(series.to_json(date_format="iso").encode()).hexdigest()
            report["result"] = {
                "first_date": series.index[0].isoformat(), "last_date": series.index[-1].isoformat(),
                "rows": len(series), "first_value": float(series.iloc[0]),
                "last_value": float(series.iloc[-1]), "sha256": initial_fingerprint,
                "metrics": [{"label": metric.label, "value": metric.value} for metric in app.metric],
            }
            key = "results_view_Local smoke portfolio"
            views = ["📈 Chart", "📊 Returns Analysis", "⚖️ Rebalancing", "💸 Tax Analysis", "🔍 X-Ray", "🔮 Monte Carlo", "🔧 Debug", "💰 Withdrawals"]
            for view in views:
                app.button_group(key=key).set_value(view).run()
                check_render(app, f"view_{view}")
                check(f"selected_{view}", app.button_group(key=key).value == view)
                report["views"].append({
                    "name": view, "plotly_charts": len(app.get("plotly_chart")),
                    "dataframes": len(app.dataframe), "metrics": len(app.metric),
                })
            app.run()
            check_render(app, "final_rerender")
            final_series = app.session_state["results_list"][0]["series"]
            check("result_persists_across_views", hashlib.sha256(final_series.to_json(date_format="iso").encode()).hexdigest() == initial_fingerprint)
            check("no_testfol_calls", report["network"]["testfol_calls"] == 0)
            unexpected = [url for url in report["network"]["blocked_requests"] if url != "http://localhost:8100/api/health"]
            check("no_unexpected_network_attempts", not unexpected, unexpected)
            report["passed"] = True
    except Exception as exc:
        report["failure"] = {"type": type(exc).__name__, "message": str(exc), "traceback": traceback.format_exc()}
    finally:
        report["finished_at"] = datetime.now(timezone.utc).isoformat()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(f"{'PASS' if report['passed'] else 'FAIL'}: {output}")
    if not report["passed"]:
        print(report.get("failure", {}).get("message", "Unknown failure"))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

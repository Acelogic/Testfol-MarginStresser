"""Read-only, uncached live Testfol wrapper smoke audit with saved evidence.

Failure modes considered before implementation: unavailable upstream, rejected
authentication/plan/quota, request schema drift, changed response shape, invalid
or misaligned values, wrapper/raw disagreement, and cached data masking an outage.
This does not prove financial methodology, browser behavior, or token refresh.

Run from the project root:
    python debug_tools/audit_testfol_api.py --output /tmp/testfol-api-audit.json
Add --authenticated to use the wrapper's configured authentication; tokens and
headers are never included in the evidence. Default execution is anonymous.
The audit disables HTTP retries to bound traffic and stops on the first failure.
"""

from __future__ import annotations

import argparse
from contextlib import nullcontext
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
import time
from datetime import datetime, timezone
from unittest.mock import patch

import numpy as np
import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.services import testfol_api  # noqa: E402


CASES = [
    ("spy_baseline", {"allocation": {"SPY": 100.0}}),
    ("leveraged_sim", {"allocation": {"QQQSIM?L=2": 100.0}}),
    ("mixed_dca_offsets", {
        "allocation": {"SPY": 60.0, "TLT": 40.0},
        "cashflow": 100.0, "cashflow_offset": 1,
        "rebalance": "Quarterly", "rebalance_offset": 1,
    }),
    ("withdrawals_no_dividend_reinvestment", {
        "allocation": {"SPY": 60.0, "TLT": 40.0},
        "cashflow": -100.0, "invest_div": False,
    }),
]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--authenticated", action="store_true")
    args = parser.parse_args()
    evidence = {
        "checked_at_utc": datetime.now(timezone.utc).isoformat(),
        "endpoint": testfol_api.API_URL,
        "git_head": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "source_sha256": {
            name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
            for name in ["app/services/testfol_api.py", "app/services/testfol_auth.py"]
        },
        "authentication": "configured" if args.authenticated else "anonymous",
        "disk_cache": "bypassed for reads and writes",
        "retries": "disabled for bounded audit traffic",
        "qualification": "live wrapper smoke; not browser, auth-refresh, or financial-methodology proof",
        "cases": [],
    }
    original_session = requests.Session

    class AuditSession(original_session):
        def mount(self, prefix, adapter):
            adapter.max_retries = requests.adapters.Retry(total=0)
            return super().mount(prefix, adapter)

        def send(self, request, **kwargs):
            response = super().send(request, **kwargs)
            item["http_status"] = response.status_code
            item["response_sha256"] = hashlib.sha256(response.content).hexdigest()
            item["response_bytes"] = len(response.content)
            # Persist only public backtest inputs, never request headers.
            item["request_payload"] = json.loads(request.body)
            try:
                raw = response.json()
                if isinstance(raw, dict):
                    item["response_keys"] = sorted(raw)
            except ValueError:
                pass
            return response

    auth_patch = (
        patch("app.services.testfol_auth.get_token", return_value=None)
        if not args.authenticated else nullcontext()
    )
    with patch.object(testfol_api, "cache_get", return_value=None), \
         patch.object(testfol_api, "cache_set"), \
         patch.object(requests, "Session", AuditSession), auth_patch:
        for name, overrides in CASES:
            item = {"name": name, "passed": False}
            evidence["cases"].append(item)
            inputs = dict(
                start_date="2024-01-02", end_date="2024-12-31", start_val=10000.0,
                cashflow=0.0, cashfreq="Monthly", rolling=6, invest_div=True,
                rebalance="Yearly", include_raw=True,
            )
            inputs.update(overrides)
            started = time.monotonic()
            try:
                series, stats, extra = testfol_api.fetch_backtest(**inputs)
                raw = extra["raw_response"]
                expected_stats = raw.get("stats", {})
                if isinstance(expected_stats, list):
                    expected_stats = expected_stats[0] if expected_stats else {}
                cashflow_events = raw.get("cashflow_events", [[]])[0]
                recorded_cashflow = sum(event["amount"] for event in cashflow_events)
                expected_rebalances = 4 if name == "mixed_dca_offsets" else 0
                checks = {
                    "http_200": item.get("http_status") == 200,
                    "nonempty": not series.empty,
                    "finite_values": bool(np.isfinite(series.to_numpy(dtype=float)).all()),
                    "positive_values": bool((series > 0).all()),
                    "ordered_unique_dates": bool(series.index.is_monotonic_increasing and series.index.is_unique),
                    "requested_bounds": bool(series.index.min() >= pd.Timestamp(inputs["start_date"]) and series.index.max() <= pd.Timestamp(inputs["end_date"])),
                    "raw_values_equal": bool(np.array_equal(series.to_numpy(), np.asarray(raw["charts"]["history"][1]))),
                    "raw_dates_equal": bool(series.index.equals(pd.to_datetime(raw["charts"]["history"][0], unit="s"))),
                    "stats_equal": stats == expected_stats,
                    "stats_nonempty": bool(stats),
                    "no_api_errors": not bool(raw.get("errors")),
                    "ending_balance_matches_stats": abs(float(series.iloc[-1]) - stats["end_val"]) <= 0.0051,
                    "contributions_match_events": abs(inputs["start_val"] + recorded_cashflow - stats["total_contributions"]) < 1e-6,
                    "cashflow_amounts_match": all(event["amount"] == inputs["cashflow"] for event in cashflow_events),
                    "cashflows_present_when_requested": bool(cashflow_events) == bool(inputs["cashflow"]),
                    "expected_rebalances": raw["rebalancing_stats"][0]["rebalancings"] == expected_rebalances,
                }
                item.update(
                    checks=checks, passed=all(checks.values()), rows=len(series),
                    first_date=str(series.index[0].date()), last_date=str(series.index[-1].date()),
                    first_value=float(series.iloc[0]), last_value=float(series.iloc[-1]),
                    stats_keys=sorted(stats), detail_level=raw.get("detail_level"),
                    locked_tabs=raw.get("locked_tabs"), api_errors=raw.get("errors"),
                    stats_numeric={k: v for k, v in stats.items() if isinstance(v, (int, float)) and math.isfinite(v)},
                    raw_response=raw,
                )
            except Exception as exc:
                # Exception text can contain upstream bodies or credentials.
                item["error_type"] = type(exc).__name__
            item["elapsed_seconds"] = round(time.monotonic() - started, 3)
            if not item["passed"]:
                break
            time.sleep(1)
    evidence["passed"] = len(evidence["cases"]) == len(CASES) and all(c["passed"] for c in evidence["cases"])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(evidence, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"artifact": str(args.output.resolve()), "passed": evidence["passed"],
                      "cases": [{k: v for k, v in c.items() if k in {"name", "http_status", "passed", "rows", "error_type"}} for c in evidence["cases"]]}))
    return 0 if evidence["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

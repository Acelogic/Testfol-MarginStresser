#!/usr/bin/env python3
"""Benchmark the recent speedups against a git baseline."""

from __future__ import annotations

import argparse
import contextlib
import importlib.util
import json
import logging
import subprocess
import sys
import tempfile
import time
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
HEAVY_ARRAY = np.linspace(1.0, 4096.0, 4096, dtype=float)

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

logging.getLogger("streamlit.runtime.caching.cache_data_api").setLevel(logging.ERROR)
logging.getLogger("streamlit.runtime.scriptrunner_utils.script_run_context").setLevel(logging.ERROR)
logging.getLogger("streamlit").setLevel(logging.ERROR)


@dataclass
class BenchmarkSummary:
    name: str
    baseline_median_ms: float
    current_median_ms: float
    speedup_x: float
    baseline_p95_ms: float
    current_p95_ms: float
    baseline_calls: dict[str, int] | None = None
    current_calls: dict[str, int] | None = None


class _DummyColumn:
    def metric(self, *args, **kwargs):
        return None


def _noop(*args, **kwargs):
    return None


def _burn_cpu(multiplier: int = 1) -> float:
    arr = HEAVY_ARRAY.copy()
    for _ in range(140 * multiplier):
        arr = np.sqrt(arr + 1.0)
        arr *= 1.000001
    return float(arr[-1])


def _run_git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=REPO_ROOT, text=True)


def _load_git_module(ref: str, rel_path: str, module_name: str, tmpdir: Path):
    module_source = _run_git("show", f"{ref}:{rel_path}")
    module_path = tmpdir / f"{module_name}.py"
    module_path.write_text(module_source)
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


@contextlib.contextmanager
def _patched_attrs(patches: list[tuple[object, str, object]]):
    originals: list[tuple[object, str, object]] = []
    try:
        for obj, attr, value in patches:
            originals.append((obj, attr, getattr(obj, attr)))
            setattr(obj, attr, value)
        yield
    finally:
        for obj, attr, value in reversed(originals):
            setattr(obj, attr, value)


def _benchmark(fn, iterations: int, warmups: int) -> list[float]:
    for _ in range(warmups):
        fn()

    timings_ms: list[float] = []
    for _ in range(iterations):
        t0 = time.perf_counter()
        fn()
        timings_ms.append((time.perf_counter() - t0) * 1000.0)
    return timings_ms


def _summarize_timings(name: str, baseline: list[float], current: list[float], baseline_calls=None, current_calls=None) -> BenchmarkSummary:
    baseline_median = float(np.median(baseline))
    current_median = float(np.median(current))
    return BenchmarkSummary(
        name=name,
        baseline_median_ms=baseline_median,
        current_median_ms=current_median,
        speedup_x=baseline_median / current_median if current_median else float("inf"),
        baseline_p95_ms=float(np.percentile(baseline, 95)),
        current_p95_ms=float(np.percentile(current, 95)),
        baseline_calls=baseline_calls,
        current_calls=current_calls,
    )


def _build_shadow_inputs() -> dict:
    rng = np.random.default_rng(42)
    tickers = [f"T{i:02d}" for i in range(24)]
    dates = pd.bdate_range("2010-01-04", "2024-12-31")

    drifts = np.linspace(0.00005, 0.00045, len(tickers))
    noise = rng.normal(loc=drifts, scale=0.0075, size=(len(dates), len(tickers)))
    returns = np.clip(noise, -0.18, 0.18)
    prices = 100.0 * np.cumprod(1.0 + returns, axis=0)
    prices_df = pd.DataFrame(prices, index=dates, columns=tickers)

    raw_weights = rng.uniform(0.5, 2.5, size=len(tickers))
    weights = raw_weights / raw_weights.sum() * 100.0
    allocation = {ticker: float(weight) for ticker, weight in zip(tickers, weights)}

    return {
        "allocation": allocation,
        "start_val": 100_000.0,
        "start_date": dates[0].strftime("%Y-%m-%d"),
        "end_date": dates[-1].strftime("%Y-%m-%d"),
        "prices_df": prices_df,
        "rebalance_freq": "Monthly",
        "cashflow": 1_500.0,
        "cashflow_freq": "Monthly",
        "tax_config": {},
    }


def _build_chart_inputs() -> dict:
    dates = pd.bdate_range("2010-01-04", "2024-12-31")
    values = 100_000.0 * np.cumprod(1.0 + np.full(len(dates), 0.00035))
    series = pd.Series(values, index=dates)
    component_prices = pd.DataFrame(
        {
            "SPY": series,
            "QQQ": series * 1.05,
            "TLT": series * 0.65,
            "GLD": series * 0.75,
        },
        index=dates,
    )
    zeros = pd.Series(0.0, index=dates)
    halfs = pd.Series(0.5, index=dates)
    return {
        "tax_adj_port_series": series,
        "final_adj_series": series,
        "loan_series": zeros,
        "tax_adj_equity_pct_series": halfs,
        "tax_adj_usage_series": zeros,
        "equity_series": series,
        "usage_series": zeros,
        "equity_pct_series": halfs,
        "effective_rate_series": zeros,
        "ohlc_data": pd.DataFrame(),
        "equity_resampled": series,
        "loan_resampled": zeros,
        "usage_resampled": zeros,
        "equity_pct_resampled": halfs,
        "effective_rate_resampled": zeros,
        "bench_resampled": None,
        "comp_resampled": None,
        "port_series": series,
        "component_prices": component_prices,
        "portfolio_name": "Benchmark",
        "log_scale": False,
        "show_range_slider": False,
        "show_volume": False,
        "timeframe": "Daily",
        "wmaint": 0.25,
        "stats": {},
        "config": {},
        "pay_tax_cash": False,
        "draw_monthly": 0.0,
        "draw_monthly_retirement": 0.0,
        "draw_start_date": None,
        "retirement_date": None,
        "logs": [],
        "final_tax_series": zeros,
        "tax_payment_series": zeros,
        "start_val": 100_000.0,
        "rate_annual": 0.0,
        "pm_enabled": False,
        "pm_mode": "Off",
        "pm_usage_series": None,
        "wmaint_pm": 0.0,
        "pm_threshold": 110_000.0,
        "pm_blocked_dates": None,
    }


def _build_returns_series() -> pd.Series:
    dates = pd.bdate_range("2005-01-03", "2024-12-31")
    values = 100_000.0 * np.cumprod(1.0 + np.full(len(dates), 0.0003))
    return pd.Series(values, index=dates)


def _benchmark_shadow(ref_module, current_module, iterations: int, warmups: int) -> BenchmarkSummary:
    inputs = _build_shadow_inputs()

    with _patched_attrs([
        (ref_module.log, "disabled", True),
        (current_module.log, "disabled", True),
    ]):
        baseline = _benchmark(lambda: ref_module.run_shadow_backtest(**inputs), iterations, warmups)
        current = _benchmark(lambda: current_module.run_shadow_backtest(**inputs), iterations, warmups)
    return _summarize_timings("shadow_backtest_monthly_dca", baseline, current)


def _benchmark_chart_tab(ref_module, current_module, iterations: int, warmups: int) -> BenchmarkSummary:
    import app.ui.results.tabs_chart as live_module

    inputs = _build_chart_inputs()

    def run_once(module):
        call_counts = Counter()

        def counted(name: str, multiplier: int = 1):
            def _inner(*args, **kwargs):
                call_counts[name] += 1
                _burn_cpu(multiplier)
                return None

            return _inner

        with _patched_attrs([
            (module.st, "segmented_control", lambda *args, **kwargs: "📉 200DMA"),
            (module.st, "tabs", lambda labels: [contextlib.nullcontext() for _ in labels]),
            (module, "_parse_events", lambda logs: pd.DataFrame()),
            (module, "_render_margin_statistics", counted("margin_stats")),
            (module, "_render_cash_statistics", counted("cash_stats")),
            (module.charts, "render_classic_chart", counted("classic_chart", 2)),
            (module.charts, "render_candlestick_chart", counted("candlestick_chart", 2)),
            (module.charts, "render_ma_analysis_tab", counted("ma_analysis", 2)),
            (module.charts, "render_munger_wma_tab", counted("munger_wma", 2)),
            (module.charts, "render_cheat_sheet", counted("cheat_sheet", 2)),
        ]):
            module.render_chart_tab(contextlib.nullcontext(), chart_style="Classic (Combined)", **inputs)

        return dict(call_counts)

    baseline_calls = run_once(ref_module)
    current_calls = run_once(current_module)
    baseline = _benchmark(lambda: run_once(ref_module), iterations, warmups)
    current = _benchmark(lambda: run_once(current_module), iterations, warmups)
    return _summarize_timings("chart_tab_selected_200dma", baseline, current, baseline_calls, current_calls)


def _benchmark_returns(ref_module, current_module, iterations: int, warmups: int) -> BenchmarkSummary:
    import app.core.calculations.stats as stats_module
    import app.services.data_service as data_service
    import app.ui.charts.metrics as metrics_module
    import app.ui.charts.rolling as rolling_module

    series = _build_returns_series()

    def fake_fetch_component_data(*args, **kwargs):
        _burn_cpu(2)
        return pd.DataFrame({"SPYSIM": series}, index=series.index)

    def fake_build_drawdown_table(*args, **kwargs):
        _burn_cpu(2)
        return pd.DataFrame()

    def run_once(module):
        with _patched_attrs([
            (module.st, "segmented_control", lambda *args, **kwargs: "📊 Daily"),
            (module.st, "tabs", lambda labels: [contextlib.nullcontext() for _ in labels]),
            (module.st, "subheader", _noop),
            (module.st, "dataframe", _noop),
            (module.st, "plotly_chart", _noop),
            (module.st, "toggle", lambda *args, **kwargs: False),
            (module.st, "columns", lambda n: [_DummyColumn() for _ in range(n)]),
            (module.st, "info", _noop),
            (module.st, "warning", _noop),
            (module.st, "caption", _noop),
            (module.st, "radio", lambda label, options, **kwargs: options[0]),
            (rolling_module, "render_rolling_metrics", lambda *args, **kwargs: _burn_cpu(2)),
            (metrics_module, "render_risk_return_metrics", lambda *args, **kwargs: _burn_cpu(2)),
            (data_service, "fetch_component_data", fake_fetch_component_data),
            (stats_module, "build_drawdown_table", fake_build_drawdown_table),
        ]):
            module.render_returns_analysis(
                series,
                unique_id="benchmark",
                portfolio_name="Benchmark",
                stats={},
                raw_response={},
            )

    baseline = _benchmark(lambda: run_once(ref_module), iterations, warmups)
    current = _benchmark(lambda: run_once(current_module), iterations, warmups)
    return _summarize_timings("returns_analysis_selected_daily", baseline, current)


def _print_markdown(summaries: list[BenchmarkSummary], baseline_ref: str, baseline_sha: str):
    print(f"# Speed Benchmark\n")
    print(f"- Baseline ref: `{baseline_ref}` ({baseline_sha})")
    print(f"- Current workspace: working tree")
    print()
    print("| Benchmark | Baseline median | Current median | Speedup | Baseline p95 | Current p95 |")
    print("| --- | ---: | ---: | ---: | ---: | ---: |")
    for summary in summaries:
        print(
            f"| {summary.name} | "
            f"{summary.baseline_median_ms:.1f} ms | "
            f"{summary.current_median_ms:.1f} ms | "
            f"{summary.speedup_x:.2f}x | "
            f"{summary.baseline_p95_ms:.1f} ms | "
            f"{summary.current_p95_ms:.1f} ms |"
        )

    extra = [s for s in summaries if s.baseline_calls or s.current_calls]
    if extra:
        print("\n## Call Counts\n")
        for summary in extra:
            print(f"### {summary.name}")
            print(f"- Baseline calls: `{json.dumps(summary.baseline_calls, sort_keys=True)}`")
            print(f"- Current calls: `{json.dumps(summary.current_calls, sort_keys=True)}`")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-ref", default="HEAD", help="Git ref to use as the baseline source.")
    parser.add_argument("--iterations", type=int, default=7, help="Number of timed iterations per benchmark.")
    parser.add_argument("--warmups", type=int, default=1, help="Number of warmup runs before timing.")
    parser.add_argument("--json", action="store_true", help="Print JSON instead of Markdown.")
    args = parser.parse_args()

    baseline_sha = _run_git("rev-parse", "--short", args.baseline_ref).strip()

    import app.core.shadow_backtest as current_shadow
    import app.ui.charts.returns as current_returns
    import app.ui.results.tabs_chart as current_tabs_chart

    with tempfile.TemporaryDirectory(prefix="benchmark_speedups_") as tmp:
        tmpdir = Path(tmp)
        baseline_shadow = _load_git_module(args.baseline_ref, "app/core/shadow_backtest.py", "baseline_shadow_backtest", tmpdir)
        baseline_returns = _load_git_module(args.baseline_ref, "app/ui/charts/returns.py", "baseline_returns", tmpdir)
        baseline_tabs_chart = _load_git_module(args.baseline_ref, "app/ui/results/tabs_chart.py", "baseline_tabs_chart", tmpdir)

        summaries = [
            _benchmark_shadow(baseline_shadow, current_shadow, args.iterations, args.warmups),
            _benchmark_chart_tab(baseline_tabs_chart, current_tabs_chart, args.iterations, args.warmups),
            _benchmark_returns(baseline_returns, current_returns, args.iterations, args.warmups),
        ]

    if args.json:
        payload = {
            "baseline_ref": args.baseline_ref,
            "baseline_sha": baseline_sha,
            "summaries": [asdict(summary) for summary in summaries],
        }
        print(json.dumps(payload, indent=2))
    else:
        _print_markdown(summaries, args.baseline_ref, baseline_sha)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

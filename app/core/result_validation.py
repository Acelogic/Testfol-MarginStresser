"""Validation helpers for serialized backtest results."""

from __future__ import annotations

import pandas as pd


def looks_like_rebased_twr_series(
    result: dict,
    cashflow_amount: float,
    pay_down_margin: bool,
) -> bool:
    """Detect stale local results where money-valued series was cached as rebased TWR."""
    try:
        cashflow_amount = float(cashflow_amount or 0.0)
    except (TypeError, ValueError):
        cashflow_amount = 0.0

    if cashflow_amount <= 0 or pay_down_margin or not result.get("is_local", False):
        return False

    series = result.get("series")
    twr = result.get("twr_series")
    if series is None or twr is None or series.empty or twr.empty:
        return False
    if twr.iloc[0] == 0:
        return False

    try:
        series = series.astype(float)
        twr_rebased = (twr.astype(float) / float(twr.iloc[0])) * float(series.iloc[0])
        aligned = pd.concat(
            [series.rename("series"), twr_rebased.rename("twr_rebased")],
            axis=1,
        ).dropna()
    except (TypeError, ValueError, ZeroDivisionError):
        return False

    if len(aligned) < 3:
        return False

    sample_count = min(10, len(aligned))
    step = max(1, len(aligned) // sample_count)
    sampled = aligned.iloc[::step].head(sample_count)
    if sampled.index[-1] != aligned.index[-1]:
        sampled = pd.concat([sampled, aligned.tail(1)])

    rel_diff = (sampled["series"] - sampled["twr_rebased"]).abs()
    rel_diff = rel_diff / sampled["series"].abs().clip(lower=1.0)
    return bool((rel_diff < 1e-4).all())


def has_stale_local_cashflow_series(results_list: list[dict], cashflow_cfg: dict) -> bool:
    """Return True when any local cashflow result appears to be stale TWR data."""
    amount = cashflow_cfg.get("amount", 0.0)
    pay_down_margin = cashflow_cfg.get("pay_down_margin", False)
    return any(
        looks_like_rebased_twr_series(result, amount, pay_down_margin)
        for result in results_list
    )

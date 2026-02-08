from __future__ import annotations

import logging

import pandas as pd
from fastapi import APIRouter, HTTPException

from api.schemas import (
    BacktestRequest, MultiBacktestRequest, MultiBacktestResponse,
    BacktestResult,
)
from app.core.backtest_orchestrator import run_single_backtest, run_multi_backtest
from app.common.cache import cache_key, cache_get, cache_set

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/backtest", tags=["backtest"])


def _pydantic_cache_key(req: BacktestRequest | MultiBacktestRequest) -> str:
    """Generate cache key from a Pydantic request model (excludes bearer_token)."""
    return cache_key(req.model_dump_json(exclude={"bearer_token"}))


# ---------------------------------------------------------------------------
# Serialization Helpers
# ---------------------------------------------------------------------------

def _serialize_series(s: pd.Series | None) -> str | None:
    """Serialize a pandas Series to JSON string (split orient)."""
    if s is None or (hasattr(s, "empty") and s.empty):
        return None
    if hasattr(s.index, "tz") and s.index.tz is not None:
        s = s.copy()
        s.index = s.index.tz_localize(None)
    return s.to_json(orient="split", date_format="iso")


def _serialize_df(df: pd.DataFrame | None) -> str | None:
    """Serialize a pandas DataFrame to JSON string (split orient)."""
    if df is None or (hasattr(df, "empty") and df.empty):
        return None
    if hasattr(df.index, "tz") and df.index.tz is not None:
        df = df.copy()
        df.index = df.index.tz_localize(None)
    return df.to_json(orient="split", date_format="iso")


def _serialize_result(r: dict) -> BacktestResult:
    """Convert raw result dict to serialized BacktestResult."""
    return BacktestResult(
        name=r["name"],
        series_json=_serialize_series(r.get("series")),
        stats=r.get("stats", {}),
        twr_series_json=_serialize_series(r.get("twr_series")),
        daily_returns_df_json=_serialize_df(r.get("daily_returns_df")),
        trades_df_json=_serialize_df(r.get("trades_df")),
        pl_by_year_json=_serialize_df(r.get("pl_by_year")),
        composition_df_json=_serialize_df(r.get("composition_df")),
        unrealized_pl_df_json=_serialize_df(r.get("unrealized_pl_df")),
        component_prices_json=_serialize_df(r.get("component_prices")),
        allocation=r.get("allocation", {}),
        logs=r.get("logs", []),
        raw_response=r.get("raw_response", {}),
        is_local=r.get("is_local", False),
        start_val=r.get("start_val", 10000.0),
        sim_range=r.get("sim_range", ""),
        shadow_range=r.get("shadow_range", ""),
        wmaint=r.get("wmaint", 0.25),
    )


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.post("", response_model=BacktestResult)
def run_backtest(req: BacktestRequest):
    """Run a single portfolio backtest."""
    key = _pydantic_cache_key(req)
    cached = cache_get(key, prefix="single")
    if cached is not None:
        return cached
    try:
        p = req.portfolio
        raw = run_single_backtest(
            allocation=p.allocation,
            maint_pcts=p.maint_pcts,
            rebalance=p.rebalance.model_dump(),
            start_date=req.start_date,
            end_date=req.end_date,
            start_val=req.cashflow.start_val,
            cashflow_amount=req.cashflow.amount,
            cashflow_freq=req.cashflow.freq,
            invest_div=req.cashflow.invest_div,
            pay_down_margin=req.cashflow.pay_down_margin,
            tax_config=req.tax_config,
            bearer_token=req.bearer_token,
            name=p.name,
        )
        result = _serialize_result(raw)
        cache_set(key, result, prefix="single")
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/multi", response_model=MultiBacktestResponse)
def run_multi_backtest_endpoint(req: MultiBacktestRequest):
    """Run multiple portfolio backtests with common-start alignment."""
    key = _pydantic_cache_key(req)
    cached = cache_get(key, prefix="multi")
    if cached is not None:
        logger.debug("Cache HIT for multi backtest %s", key[:8])
        return cached

    try:
        # Convert Pydantic models to plain dicts for the orchestrator
        portfolios_plain = []
        for p in req.portfolios:
            portfolios_plain.append({
                "name": p.name,
                "allocation": p.allocation,
                "maint_pcts": p.maint_pcts,
                "rebalance": p.rebalance.model_dump(),
            })

        results_raw, bench_raw = run_multi_backtest(
            portfolios=portfolios_plain,
            start_date=req.start_date,
            end_date=req.end_date,
            start_val=req.cashflow.start_val,
            cashflow_amount=req.cashflow.amount,
            cashflow_freq=req.cashflow.freq,
            invest_div=req.cashflow.invest_div,
            pay_down_margin=req.cashflow.pay_down_margin,
            tax_config=req.tax_config,
            bearer_token=req.bearer_token,
        )

        # Serialize results
        serialized_results = [_serialize_result(r) for r in results_raw]

        # Serialize bench series (these are plain pd.Series from the orchestrator)
        serialized_bench = []
        for b in bench_raw:
            if hasattr(b, "empty") and not b.empty:
                # bench_raw contains pd.Series objects
                serialized_bench.append(BacktestResult(
                    name=b.name if hasattr(b, "name") else "Benchmark",
                    series_json=_serialize_series(b),
                    stats={},
                ))

        # Determine common_start from results
        start_dates = []
        for res in results_raw:
            s = res.get("series")
            if s is not None and not s.empty:
                start_dates.append(s.index.min())
        common_start = max(start_dates) if start_dates else None

        response = MultiBacktestResponse(
            results=serialized_results,
            bench_results=serialized_bench,
            common_start=common_start.isoformat() if common_start else None,
        )
        cache_set(key, response, prefix="multi")
        logger.debug("Cached multi backtest %s", key[:8])
        return response
    except Exception as e:
        logger.exception("Multi backtest failed")
        raise HTTPException(status_code=500, detail=str(e))

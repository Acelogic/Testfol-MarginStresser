"""Live quote helpers for returns analysis."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd

from app.common.special_tickers import (
    clean_ticker_symbol,
    live_price_fallback_ticker,
    live_price_uses_native_leverage,
)


QuoteFetcher = Callable[[tuple[str, ...], pd.Timestamp], dict[str, dict[str, Any]]]


def _parse_query_params(ticker: str) -> dict[str, str]:
    if "?" not in str(ticker):
        return {}

    query = str(ticker).split("?", 1)[1]
    params: dict[str, str] = {}
    for pair in query.split("&"):
        if "=" not in pair:
            continue
        key, value = pair.split("=", 1)
        params[key.upper()] = value
    return params


def _float_param(params: dict[str, str], key: str, default: float) -> float:
    try:
        return float(params.get(key, default))
    except (TypeError, ValueError):
        return default


def _normalize_price_series(series: pd.Series | None) -> pd.Series:
    if series is None or series.empty:
        return pd.Series(dtype=float)

    result = pd.Series(series).dropna().astype(float).sort_index()
    if result.empty:
        return pd.Series(dtype=float)

    result.index = pd.to_datetime(result.index)
    if result.index.tz is not None:
        result.index = result.index.tz_localize(None)
    return result[~result.index.duplicated(keep="last")].sort_index()


def _latest_position_values(
    portfolio_value: float,
    allocation: dict | None,
    composition_df: pd.DataFrame | None,
    reference_date: pd.Timestamp,
) -> pd.Series:
    if composition_df is not None and not composition_df.empty:
        required = {"Date", "Ticker", "Value"}
        if required.issubset(composition_df.columns):
            comp = composition_df[["Date", "Ticker", "Value"]].copy()
            comp["Date"] = pd.to_datetime(comp["Date"], errors="coerce")
            comp["Value"] = pd.to_numeric(comp["Value"], errors="coerce")
            comp = comp.dropna(subset=["Date", "Ticker", "Value"])

            eligible = comp[comp["Date"] <= reference_date]
            if eligible.empty:
                eligible = comp

            if not eligible.empty:
                snap_date = eligible["Date"].max()
                positions = (
                    eligible[eligible["Date"] == snap_date]
                    .groupby("Ticker", sort=False)["Value"]
                    .sum()
                )
                total = float(positions.sum())
                if total > 0:
                    return positions * (portfolio_value / total)

    if allocation:
        weights = pd.Series(allocation, dtype=float)
        total_weight = float(weights.sum())
        if total_weight > 0:
            return weights * (portfolio_value / total_weight)

    return pd.Series(dtype=float)


def _effective_return(
    price_path: pd.Series,
    reference_date: pd.Timestamp,
    leverage: float,
    expense_ratio_pct: float,
) -> tuple[float, float, float, pd.Timestamp | None]:
    prices = _normalize_price_series(price_path)
    if prices.empty:
        return 0.0, np.nan, np.nan, None

    reference_prices = prices[prices.index <= reference_date]
    if reference_prices.empty:
        return 0.0, np.nan, np.nan, None

    reference_idx = reference_prices.index[-1]

    reference_price = float(prices.loc[reference_idx])
    if reference_price == 0 or not np.isfinite(reference_price):
        return 0.0, reference_price, np.nan, reference_idx

    path = prices[prices.index >= reference_idx].copy()
    if path.empty or path.index[0] != reference_idx:
        path = pd.concat([pd.Series([reference_price], index=[reference_idx]), path])
    path = path[~path.index.duplicated(keep="last")].sort_index()

    if len(path) < 2:
        return 0.0, reference_price, float(path.iloc[-1]), reference_idx

    raw_daily = path.pct_change(fill_method=None).dropna()
    daily_er = (expense_ratio_pct / 100.0) / 252.0 if expense_ratio_pct else 0.0
    effective_daily = raw_daily * leverage - daily_er
    effective_total = float((1.0 + effective_daily).prod() - 1.0)
    return effective_total, reference_price, float(path.iloc[-1]), reference_idx


def fetch_yahoo_live_price_series(
    symbols: Iterable[str],
    reference_date: pd.Timestamp,
) -> dict[str, dict[str, Any]]:
    """Fetch recent adjusted closes plus a current Yahoo Finance quote."""
    import yfinance as yf

    unique_symbols = tuple(dict.fromkeys(str(s).strip().upper() for s in symbols if str(s).strip()))
    ref = pd.Timestamp(reference_date)
    start = (ref - pd.Timedelta(days=14)).strftime("%Y-%m-%d")
    end = (pd.Timestamp.now().normalize() + pd.Timedelta(days=2)).strftime("%Y-%m-%d")

    quote_data: dict[str, dict[str, Any]] = {}
    for symbol in unique_symbols:
        try:
            ticker = yf.Ticker(symbol)
            hist = ticker.history(start=start, end=end, auto_adjust=True, timeout=10)
            close = _normalize_price_series(hist["Close"] if "Close" in hist else None)

            live_price = np.nan
            live_time: pd.Timestamp | None = None
            try:
                fast_info = getattr(ticker, "fast_info", {}) or {}
                live_price = float(fast_info.get("last_price", np.nan))
            except Exception:
                live_price = np.nan

            if not np.isfinite(live_price):
                try:
                    intraday = ticker.history(period="1d", interval="1m", auto_adjust=True, timeout=10)
                    intraday_close = _normalize_price_series(
                        intraday["Close"] if "Close" in intraday else None
                    )
                    if not intraday_close.empty:
                        live_price = float(intraday_close.iloc[-1])
                        live_time = pd.Timestamp(intraday_close.index[-1])
                except Exception:
                    pass

            if np.isfinite(live_price):
                if live_time is None:
                    live_time = pd.Timestamp(datetime.now())
                close.loc[live_time] = live_price
                close = close.sort_index()

            if close.empty:
                quote_data[symbol] = {"prices": close, "error": "No Yahoo Finance price data"}
            else:
                quote_data[symbol] = {
                    "prices": close,
                    "live_price": live_price if np.isfinite(live_price) else float(close.iloc[-1]),
                    "live_time": live_time or pd.Timestamp(close.index[-1]),
                    "error": None,
                }
        except Exception as exc:
            quote_data[symbol] = {
                "prices": pd.Series(dtype=float),
                "error": str(exc),
            }

    return quote_data


def build_live_returns_snapshot(
    port_series: pd.Series,
    allocation: dict | None = None,
    composition_df: pd.DataFrame | None = None,
    price_fetcher: QuoteFetcher = fetch_yahoo_live_price_series,
) -> dict[str, Any]:
    """Estimate live portfolio value from current quotes and last composition."""
    if port_series is None or port_series.empty:
        return {"ok": False, "error": "No portfolio series is available."}

    series = pd.Series(port_series).dropna()
    if series.empty:
        return {"ok": False, "error": "No portfolio values are available."}

    if not isinstance(series.index, pd.DatetimeIndex):
        series.index = pd.to_datetime(series.index)
    if series.index.tz is not None:
        series.index = series.index.tz_localize(None)
    series = series.sort_index()

    reference_date = pd.Timestamp(series.index[-1])
    reference_value = float(series.iloc[-1])
    positions = _latest_position_values(reference_value, allocation, composition_df, reference_date)
    if positions.empty:
        return {"ok": False, "error": "No allocation or composition data is available."}

    position_specs: list[dict[str, Any]] = []
    for ticker, value in positions.items():
        params = _parse_query_params(str(ticker))
        configured_leverage = _float_param(params, "L", 1.0)
        expense_ratio = _float_param(params, "E", _float_param(params, "D", 0.0))
        uses_native_leverage = live_price_uses_native_leverage(str(ticker))
        position_specs.append(
            {
                "ticker": str(ticker),
                "base": clean_ticker_symbol(str(ticker)),
                "live_symbol": live_price_fallback_ticker(str(ticker)),
                "position_value": float(value),
                "leverage": configured_leverage,
                "live_return_multiplier": 1.0 if uses_native_leverage else configured_leverage,
                "expense_ratio": 0.0 if uses_native_leverage else expense_ratio,
            }
        )

    live_symbols = tuple(sorted({spec["live_symbol"] for spec in position_specs}))
    quotes = price_fetcher(live_symbols, reference_date)

    rows: list[dict[str, Any]] = []
    total_live_value = 0.0
    quote_times: list[pd.Timestamp] = []
    updated_count = 0

    for spec in position_specs:
        quote = quotes.get(spec["live_symbol"], {})
        prices = _normalize_price_series(quote.get("prices"))
        status = quote.get("error") or "OK"

        if prices.empty:
            effective_ret = 0.0
            underlying_ret = np.nan
            reference_price = np.nan
            live_price = np.nan
            quote_reference_date = None
        else:
            effective_ret, reference_price, live_price, quote_reference_date = _effective_return(
                prices,
                reference_date,
                spec["live_return_multiplier"],
                spec["expense_ratio"],
            )
            if quote_reference_date is None:
                status = "No quote at reference date"
            underlying_ret = (
                float(live_price / reference_price - 1.0)
                if reference_price and np.isfinite(reference_price) and np.isfinite(live_price)
                else np.nan
            )
            if status == "OK":
                updated_count += 1

            live_time = quote.get("live_time")
            if live_time is not None:
                quote_times.append(pd.Timestamp(live_time))

        current_value = spec["position_value"] * (1.0 + effective_ret)
        contribution = current_value - spec["position_value"]
        total_live_value += current_value

        rows.append(
            {
                "Ticker": spec["ticker"],
                "Live Ticker": spec["live_symbol"],
                "Weight": spec["position_value"] / reference_value if reference_value else np.nan,
                "Reference Price": reference_price,
                "Live Price": live_price,
                "Underlying Return": underlying_ret,
                "Effective Return": effective_ret,
                "Position Value": spec["position_value"],
                "Estimated Value": current_value,
                "Contribution": contribution,
                "Leverage": spec["leverage"],
                "Reference Date": quote_reference_date,
                "Status": status,
            }
        )

    live_return = total_live_value / reference_value - 1.0 if reference_value else np.nan
    asof = max(quote_times) if quote_times else None

    return {
        "ok": True,
        "reference_date": reference_date,
        "reference_value": reference_value,
        "live_value": total_live_value,
        "live_return": live_return,
        "live_change": total_live_value - reference_value,
        "asof": asof,
        "updated_count": updated_count,
        "position_count": len(rows),
        "rows": pd.DataFrame(rows),
    }

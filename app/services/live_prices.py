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

EXTENDED_HOURS_SESSION_BY_STATE = {
    "PRE": "Pre-market",
    "PREPRE": "Pre-market",
    "REGULAR": "Regular",
    "POST": "After-hours",
    "POSTPOST": "After-hours",
    "CLOSED": "Closed",
}


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


def _finite_price(value: Any) -> float:
    try:
        price = float(value)
    except (TypeError, ValueError):
        return np.nan
    return price if np.isfinite(price) else np.nan


def _quote_timestamp(value: Any) -> pd.Timestamp | None:
    if value in (None, ""):
        return None

    try:
        if isinstance(value, pd.Timestamp):
            ts = value
        elif isinstance(value, datetime):
            ts = pd.Timestamp(value)
        elif isinstance(value, (int, float, np.integer, np.floating)):
            numeric = float(value)
            unit = "ms" if numeric > 10_000_000_000 else "s"
            return (
                pd.to_datetime(numeric, unit=unit, utc=True)
                .tz_convert("America/New_York")
                .tz_localize(None)
            )
        else:
            ts = pd.Timestamp(value)
    except (TypeError, ValueError, OverflowError):
        return None

    if pd.isna(ts):
        return None
    if ts.tzinfo is not None:
        ts = ts.tz_convert("America/New_York").tz_localize(None)
    return ts


def _session_from_market_state(market_state: Any) -> str | None:
    state = str(market_state or "").strip().upper()
    return EXTENDED_HOURS_SESSION_BY_STATE.get(state)


def _session_from_timestamp(timestamp: pd.Timestamp | None) -> str | None:
    if timestamp is None:
        return None

    ts = pd.Timestamp(timestamp)
    minute = ts.hour * 60 + ts.minute
    if 4 * 60 <= minute < 9 * 60 + 30:
        return "Pre-market"
    if 9 * 60 + 30 <= minute < 16 * 60:
        return "Regular"
    if 16 * 60 <= minute < 20 * 60:
        return "After-hours"
    return "Closed"


def _quote_from_yahoo_info(info: dict[str, Any]) -> tuple[float, pd.Timestamp | None, str | None]:
    market_state = str(info.get("marketState") or "").strip().upper()
    quote_specs = {
        "Pre-market": ("preMarketPrice", "preMarketTime"),
        "Regular": ("regularMarketPrice", "regularMarketTime"),
        "After-hours": ("postMarketPrice", "postMarketTime"),
        "Current": ("currentPrice", "regularMarketTime"),
    }

    priority_by_state = {
        "PRE": ("Pre-market", "Regular", "After-hours", "Current"),
        "PREPRE": ("Pre-market", "Regular", "After-hours", "Current"),
        "REGULAR": ("Regular", "Current", "Pre-market", "After-hours"),
        "POST": ("After-hours", "Regular", "Current", "Pre-market"),
        "POSTPOST": ("After-hours", "Regular", "Current", "Pre-market"),
    }

    candidates: dict[str, tuple[float, pd.Timestamp | None]] = {}
    for session, (price_key, time_key) in quote_specs.items():
        price = _finite_price(info.get(price_key))
        if np.isfinite(price):
            candidates[session] = (price, _quote_timestamp(info.get(time_key)))

    for session in priority_by_state.get(market_state, ()):
        if session in candidates:
            price, timestamp = candidates[session]
            label = session if session != "Current" else _session_from_market_state(market_state)
            return price, timestamp, label or session

    timestamped = [
        (session, price, timestamp)
        for session, (price, timestamp) in candidates.items()
        if timestamp is not None
    ]
    if timestamped:
        session, price, timestamp = max(timestamped, key=lambda item: item[2])
        label = session if session != "Current" else _session_from_timestamp(timestamp)
        return price, timestamp, label

    for session, (price, timestamp) in candidates.items():
        return price, timestamp, session if session != "Current" else None

    return np.nan, None, _session_from_market_state(market_state)


def _quote_from_fast_info(fast_info: Any) -> tuple[float, pd.Timestamp | None]:
    price = np.nan
    timestamp = None
    try:
        price = _finite_price(fast_info.get("last_price", np.nan))
    except Exception:
        price = np.nan

    for key in ("last_timestamp", "last_time", "last_trade_time"):
        try:
            timestamp = _quote_timestamp(fast_info.get(key))
        except Exception:
            timestamp = None
        if timestamp is not None:
            break

    return price, timestamp


def _prefer_intraday_quote(
    current_price: float,
    current_time: pd.Timestamp | None,
    current_session: str | None,
    intraday_close: pd.Series,
) -> tuple[float, pd.Timestamp | None, str | None]:
    if intraday_close.empty:
        return current_price, current_time, current_session

    intraday_time = pd.Timestamp(intraday_close.index[-1])
    intraday_price = _finite_price(intraday_close.iloc[-1])
    if not np.isfinite(intraday_price):
        return current_price, current_time, current_session

    if current_time is None or intraday_time >= current_time:
        return intraday_price, intraday_time, _session_from_timestamp(intraday_time)

    return current_price, current_time, current_session


def _combined_session_label(sessions: Iterable[str | None]) -> str | None:
    labels = sorted({str(session) for session in sessions if session})
    if not labels:
        return None
    if len(labels) == 1:
        return labels[0]
    if set(labels).issubset({"Pre-market", "After-hours"}):
        return "Extended-hours"
    return "Mixed"


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
    """Fetch recent adjusted closes plus current Yahoo Finance regular/extended quotes."""
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
            market_session: str | None = None

            try:
                info = getattr(ticker, "info", {}) or {}
                if isinstance(info, dict):
                    live_price, live_time, market_session = _quote_from_yahoo_info(info)
            except Exception:
                live_price = np.nan
                live_time = None
                market_session = None

            try:
                fast_info = getattr(ticker, "fast_info", {}) or {}
                fast_price, fast_time = _quote_from_fast_info(fast_info)
                if (
                    np.isfinite(fast_price)
                    and (
                        not np.isfinite(live_price)
                        or live_time is None
                        or (fast_time is not None and fast_time > live_time)
                    )
                ):
                    live_price = fast_price
                    live_time = fast_time
                    market_session = _session_from_timestamp(fast_time) or market_session
            except Exception:
                pass

            try:
                intraday = ticker.history(
                    period="1d",
                    interval="1m",
                    auto_adjust=True,
                    prepost=True,
                    timeout=10,
                )
                intraday_close = _normalize_price_series(
                    intraday["Close"] if "Close" in intraday else None
                )
                live_price, live_time, market_session = _prefer_intraday_quote(
                    live_price,
                    live_time,
                    market_session,
                    intraday_close,
                )
            except Exception:
                pass

            if not np.isfinite(live_price):
                try:
                    intraday = ticker.history(
                        period="1d",
                        interval="1m",
                        auto_adjust=True,
                        prepost=False,
                        timeout=10,
                    )
                    intraday_close = _normalize_price_series(
                        intraday["Close"] if "Close" in intraday else None
                    )
                    if not intraday_close.empty:
                        live_price = float(intraday_close.iloc[-1])
                        live_time = pd.Timestamp(intraday_close.index[-1])
                        market_session = _session_from_timestamp(live_time)
                except Exception:
                    pass

            if np.isfinite(live_price):
                if live_time is None:
                    live_time = pd.Timestamp(datetime.now())
                if market_session is None:
                    market_session = _session_from_timestamp(live_time)
                close.loc[live_time] = live_price
                close = close.sort_index()

            if close.empty:
                quote_data[symbol] = {"prices": close, "error": "No Yahoo Finance price data"}
            else:
                quote_data[symbol] = {
                    "prices": close,
                    "live_price": live_price if np.isfinite(live_price) else float(close.iloc[-1]),
                    "live_time": live_time or pd.Timestamp(close.index[-1]),
                    "market_session": market_session,
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
    quote_sessions: list[str] = []
    updated_count = 0

    for spec in position_specs:
        quote = quotes.get(spec["live_symbol"], {})
        prices = _normalize_price_series(quote.get("prices"))
        status = quote.get("error") or "OK"
        market_session = quote.get("market_session")

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
            if market_session:
                quote_sessions.append(str(market_session))

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
                "Market Session": market_session or "Unknown",
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
        "market_session": _combined_session_label(quote_sessions),
        "updated_count": updated_count,
        "position_count": len(rows),
        "rows": pd.DataFrame(rows),
    }

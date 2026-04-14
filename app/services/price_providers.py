"""
Multi-provider price data abstraction layer.

Provides a unified interface for fetching historical close prices from
Polygon.io, yfinance, or a chained combination with per-ticker fallback.

Usage:
    from app.services.price_providers import get_price_provider
    provider = get_price_provider()
    prices_df = provider.fetch_prices(["AAPL", "MSFT"], "2020-01-01", "2024-12-31")
"""

from __future__ import annotations

import json
import logging
import os
import time
from typing import Protocol, runtime_checkable

import pandas as pd

from app.common.cache import cache_get, cache_key, cache_set

logger = logging.getLogger(__name__)

PRICE_CACHE_TTL = 86400  # 24 hours — daily close prices only change once per trading day
PRICE_CACHE_PREFIX = "prices"


# ---------------------------------------------------------------------------
# Protocol
# ---------------------------------------------------------------------------

@runtime_checkable
class PriceProvider(Protocol):
    name: str

    def fetch_prices(
        self,
        tickers: list[str],
        start_date: str,
        end_date: str,
        adjusted: bool = True,
    ) -> pd.DataFrame:
        """Return a DataFrame with DatetimeIndex and one column per ticker (close prices)."""
        ...

    def is_available(self) -> bool:
        """Quick check — e.g. API key is present."""
        ...


# ---------------------------------------------------------------------------
# YFinance Provider
# ---------------------------------------------------------------------------

class YFinanceProvider:
    name = "yfinance"

    def is_available(self) -> bool:
        return True  # No API key required

    def fetch_prices(
        self,
        tickers: list[str],
        start_date: str,
        end_date: str,
        adjusted: bool = True,
    ) -> pd.DataFrame:
        import yfinance as yf

        if not tickers:
            return pd.DataFrame()

        sd = str(start_date)
        # yfinance treats end= as exclusive; callers pass inclusive app dates.
        ed = (pd.Timestamp(end_date) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")

        result = pd.DataFrame()

        if len(tickers) == 1:
            # Single-ticker path: use Ticker.history() for reliability
            ticker = tickers[0]
            try:
                hist = yf.Ticker(ticker).history(
                    start=sd, end=ed, auto_adjust=adjusted, timeout=30,
                )
                if not hist.empty and "Close" in hist.columns:
                    if hist.index.tz is not None:
                        hist.index = hist.index.tz_localize(None)
                    result[ticker] = hist["Close"]
            except Exception as e:
                logger.warning(f"[yfinance] Failed to fetch {ticker}: {e}")
        else:
            # Multi-ticker batch download
            try:
                data = yf.download(
                    tickers, start=sd, end=ed,
                    auto_adjust=adjusted, progress=False, timeout=30,
                )
                if data.empty:
                    return pd.DataFrame()

                if "Close" in data.columns if isinstance(data.columns, pd.Index) else "Close" in data.columns.get_level_values(0):
                    closes = data["Close"] if len(tickers) > 1 else data[["Close"]].rename(columns={"Close": tickers[0]})
                else:
                    closes = data

                if isinstance(closes, pd.Series):
                    closes = closes.to_frame(name=tickers[0])

                if closes.index.tz is not None:
                    closes.index = closes.index.tz_localize(None)

                result = closes
            except Exception as e:
                logger.warning(f"[yfinance] Batch download failed: {e}")

        return result


# ---------------------------------------------------------------------------
# Polygon.io Provider
# ---------------------------------------------------------------------------

class PolygonProvider:
    name = "polygon"

    def __init__(self, api_key: str | None = None, rate_limit_ms: int = 200):
        self._api_key = api_key or os.environ.get("POLYGON_API_KEY")
        self._rate_limit_s = rate_limit_ms / 1000.0

    def is_available(self) -> bool:
        return bool(self._api_key)

    def fetch_prices(
        self,
        tickers: list[str],
        start_date: str,
        end_date: str,
        adjusted: bool = True,
    ) -> pd.DataFrame:
        if not self._api_key:
            return pd.DataFrame()

        try:
            from polygon import RESTClient
        except ImportError:
            logger.warning("[polygon] polygon-api-client not installed")
            return pd.DataFrame()

        client = RESTClient(self._api_key)
        sd = str(start_date)
        ed = str(end_date)
        all_data: dict[str, pd.Series] = {}

        for i, ticker in enumerate(tickers):
            if i > 0 and self._rate_limit_s > 0:
                time.sleep(self._rate_limit_s)
            try:
                aggs = client.get_aggs(
                    ticker=ticker,
                    multiplier=1,
                    timespan="day",
                    from_=sd,
                    to=ed,
                    adjusted=adjusted,
                    sort="asc",
                    limit=50000,
                )
                if aggs:
                    dates = [pd.to_datetime(a.timestamp, unit="ms").tz_localize(None) for a in aggs]
                    closes = [a.close for a in aggs]
                    all_data[ticker] = pd.Series(closes, index=dates, name=ticker)
                else:
                    logger.debug(f"[polygon] No data for {ticker}")
            except Exception as e:
                logger.warning(f"[polygon] Error fetching {ticker}: {e}")

        if not all_data:
            return pd.DataFrame()

        df = pd.DataFrame(all_data)
        df.index = pd.to_datetime(df.index)
        df = df.sort_index()
        return df


# ---------------------------------------------------------------------------
# Chained Provider (per-ticker fallback)
# ---------------------------------------------------------------------------

class ChainedProvider:
    name = "chained"

    def __init__(self, providers: list[PriceProvider]):
        self._providers = [p for p in providers if p.is_available()]
        if not self._providers:
            raise ValueError("No available price providers")
        self.name = "+".join(p.name for p in self._providers)

    def is_available(self) -> bool:
        return len(self._providers) > 0

    def fetch_prices(
        self,
        tickers: list[str],
        start_date: str,
        end_date: str,
        adjusted: bool = True,
    ) -> pd.DataFrame:
        if not tickers:
            return pd.DataFrame()

        # Check cache first
        cache_payload = json.dumps({
            "provider": self.name,
            "tickers": sorted(tickers),
            "start_date": str(start_date),
            "end_date": str(end_date),
            "adjusted": adjusted,
            "cache_version": "inclusive-yfinance-end-v3",
        }, sort_keys=True)
        ck = cache_key(cache_payload)
        cached = cache_get(ck, prefix=PRICE_CACHE_PREFIX, ttl=PRICE_CACHE_TTL)
        if cached is not None:
            cached_cols = set(cached.columns) if hasattr(cached, "columns") else set()
            missing_cached = [
                ticker for ticker in tickers
                if ticker not in cached_cols or cached[ticker].dropna().empty
            ]
            if not missing_cached:
                return cached
            logger.info("Ignoring incomplete price cache; missing tickers: %s", missing_cached)

        result = pd.DataFrame()
        remaining = list(tickers)

        for provider in self._providers:
            if not remaining:
                break
            try:
                prices = provider.fetch_prices(remaining, start_date, end_date, adjusted)
                if prices.empty:
                    continue

                # Merge successful columns
                for col in prices.columns:
                    if col in remaining and not prices[col].dropna().empty:
                        result[col] = prices[col]
                        remaining.remove(col)

            except Exception as e:
                logger.warning(f"[{provider.name}] Failed: {e}")

        if remaining:
            logger.info(f"No data found for tickers: {remaining}")

        if not result.empty:
            result = result.sort_index()
            cache_set(ck, result, prefix=PRICE_CACHE_PREFIX)

        return result


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

_singleton: ChainedProvider | None = None


def get_price_provider() -> ChainedProvider:
    """Return a configured ChainedProvider singleton.

    Provider order: Polygon (if API key set) → yfinance.
    """
    global _singleton
    if _singleton is not None:
        return _singleton

    providers: list[PriceProvider] = []

    polygon_key = os.environ.get("POLYGON_API_KEY")
    if polygon_key:
        providers.append(PolygonProvider(api_key=polygon_key))

    providers.append(YFinanceProvider())

    _singleton = ChainedProvider(providers)
    logger.info(f"Price provider chain: {_singleton.name}")
    return _singleton


def reset_provider():
    """Reset the singleton (for testing or config changes)."""
    global _singleton
    _singleton = None

from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
import warnings

import pandas as pd

from app.common.cache import cache_get, cache_key, cache_set
from app.common.constants import Tickers
from app.services import testfol_api as api
from app.services.price_providers import get_price_provider

logger = logging.getLogger(__name__)

COMPONENT_DATA_CACHE_TTL = 86400
COMPONENT_DATA_CACHE_PREFIX = "component_prices"


def _normalize_date_str(value) -> str:
    """Return a stable YYYY-MM-DD-style string for cache/provider keys."""
    return value.strftime("%Y-%m-%d") if hasattr(value, "strftime") else str(value)


def clip_component_data_to_synced_end(
    prices: pd.DataFrame,
    required_columns: list[str],
) -> pd.DataFrame:
    """Drop trailing dates until every requested component has real data.

    This prevents mixed-date portfolios, e.g. live single names updating to a
    new trading day while SIM/proxy sleeves are still one close behind.
    """
    if prices.empty or not required_columns:
        return prices

    result = prices.copy()
    if not isinstance(result.index, pd.DatetimeIndex):
        result.index = pd.to_datetime(result.index)
    if result.index.tz is not None:
        result.index = result.index.tz_localize(None)
    result = result.sort_index()

    columns = [col for col in dict.fromkeys(required_columns) if col in result.columns]
    if len(columns) < 2:
        return result

    last_valid_by_col: dict[str, pd.Timestamp] = {}
    for col in columns:
        series = result[col].dropna()
        if not series.empty:
            last_valid_by_col[col] = pd.Timestamp(series.index[-1])

    if len(last_valid_by_col) < 2:
        return result

    synced_end = min(last_valid_by_col.values())
    latest_end = max(last_valid_by_col.values())
    if synced_end >= latest_end:
        return result

    stale_cols = sorted(
        col for col, last_date in last_valid_by_col.items()
        if last_date == synced_end
    )
    fresh_cols = sorted(
        col for col, last_date in last_valid_by_col.items()
        if last_date == latest_end
    )
    logger.warning(
        "Clipping component data to %s because not all sleeves are synced "
        "(stale: %s; latest: %s at %s)",
        synced_end.date(),
        ", ".join(stale_cols),
        ", ".join(fresh_cols),
        latest_end.date(),
    )
    return result.loc[result.index <= synced_end]


def _component_cache_is_complete(
    prices: pd.DataFrame,
    required_columns: list[str],
    start_date: str,
    end_date: str,
) -> bool:
    if prices.empty:
        return False

    if not isinstance(prices.index, pd.DatetimeIndex):
        prices = prices.copy()
        prices.index = pd.to_datetime(prices.index)

    window = prices.loc[pd.Timestamp(start_date):pd.Timestamp(end_date)]
    if window.empty:
        return False

    for col in dict.fromkeys(required_columns):
        if col not in window.columns or window[col].dropna().empty:
            logger.info("Ignoring component cache with missing/empty column: %s", col)
            return False

    return True


def fetch_component_data(tickers: list[str], start_date, end_date, *, sync_end: bool = True) -> pd.DataFrame:
    """
    Fetches historical data for each ticker individually via Testfol API (or local).
    Handles composite tickers like NDXMEGASIM by splicing local CSVs with live data.
    """
    if not tickers:
        return pd.DataFrame()

    unique_tickers = list(dict.fromkeys(tickers))
    unique_bases = list(dict.fromkeys(ticker.split("?")[0] for ticker in unique_tickers))
    sd_str = _normalize_date_str(start_date)
    ed_str = _normalize_date_str(end_date)

    cache_payload = json.dumps(
        {
            "tickers": unique_tickers,
            "start_date": sd_str,
            "end_date": ed_str,
            "sync_end": sync_end,
            "sync_policy": "requested-window-v3",
        },
        sort_keys=True,
    )
    ck = cache_key(cache_payload)
    cached = cache_get(ck, prefix=COMPONENT_DATA_CACHE_PREFIX, ttl=COMPONENT_DATA_CACHE_TTL)
    if cached is not None and _component_cache_is_complete(cached, unique_bases, sd_str, ed_str):
        return cached

    combined_prices: dict[str, pd.Series] = {}
    provider = get_price_provider()
    today_str = pd.Timestamp.now().strftime("%Y-%m-%d")

    proxy_tickers: list[str] = []
    if Tickers.NDX30SIM in unique_bases:
        proxy_tickers.append("QTOP")
    if any(base in (Tickers.NDXMEGASIM, Tickers.NDXMEGA2SIM) for base in unique_bases):
        proxy_tickers.append(Tickers.QBIG)

    proxy_prices = pd.DataFrame()
    if proxy_tickers:
        try:
            proxy_prices = provider.fetch_prices(sorted(set(proxy_tickers)), "2000-01-01", today_str)
        except Exception as e:
            warnings.warn(f"Failed to fetch proxy data for synthetic tickers: {e}")
            proxy_prices = pd.DataFrame()

    from app.core.shadow_backtest import parse_ticker

    mapped_provider_tickers: dict[str, str] = {}
    for base in unique_bases:
        is_sim_request = base.upper().endswith("SIM") or base.upper().endswith("TR")
        if not is_sim_request:
            mapped_ticker, _ = parse_ticker(base)
            mapped_provider_tickers[base] = mapped_ticker

    batched_provider_prices = pd.DataFrame()
    if mapped_provider_tickers:
        try:
            batched_provider_prices = provider.fetch_prices(
                sorted(set(mapped_provider_tickers.values())),
                sd_str,
                ed_str,
            )
        except Exception as e:
            warnings.warn(f"Failed batched provider fetch: {e}")
            batched_provider_prices = pd.DataFrame()

    for base in unique_bases:
        try:
            # SPECIAL: NDX30SIM — load simulation CSV + splice with QTOP
            if base == Tickers.NDX30SIM:
                try:
                    csv_path = f"data/{base}.csv"
                    df_sim = pd.Series(dtype=float)
                    if os.path.exists(csv_path):
                        _df = pd.read_csv(csv_path)
                        if "Date" in _df.columns:
                            _df["Date"] = pd.to_datetime(_df["Date"])
                            _df = _df.set_index("Date")
                        if "Close" in _df.columns:
                            df_sim = _df["Close"].sort_index()
                    else:
                        warnings.warn(f"{base} requested but {csv_path} not found.")

                    qtop_series = pd.Series(dtype=float)
                    if not proxy_prices.empty and "QTOP" in proxy_prices.columns:
                        qtop_series = proxy_prices["QTOP"].dropna().sort_index()

                    if not qtop_series.empty and not df_sim.empty:
                        splice_date = qtop_series.index[0]
                        sim_part = df_sim[df_sim.index < splice_date]
                        if not sim_part.empty:
                            sim_end_val = sim_part.iloc[-1]
                            qtop_start_val = qtop_series.iloc[0]
                            scale_factor = qtop_start_val / sim_end_val if sim_end_val != 0 else 1.0
                            combined_prices[base] = pd.concat([sim_part * scale_factor, qtop_series])
                        else:
                            combined_prices[base] = qtop_series
                    elif not df_sim.empty:
                        combined_prices[base] = df_sim
                    elif not qtop_series.empty:
                        combined_prices[base] = qtop_series

                    continue
                except Exception as e:
                    raise RuntimeError(f"Failed to load/splice NDX30SIM: {e}")

            # SPECIAL: Load NDX Mega simulations from local CSV + splice with QBIG
            if base in [Tickers.NDXMEGASIM, Tickers.NDXMEGA2SIM]:
                try:
                    csv_path = f"data/{base}.csv"
                    df_sim = pd.DataFrame()
                    if os.path.exists(csv_path):
                        try:
                            df_sim = pd.read_csv(csv_path)
                        except (pd.errors.ParserError, pd.errors.EmptyDataError, ValueError) as e:
                            warnings.warn(f"Corruption detected in {base}.csv ({e}). Attempting auto-rebuild...")
                            rebuild_script = os.path.join("data", "ndx_simulation", "scripts", "rebuild_all.py")
                            if os.path.exists(rebuild_script):
                                try:
                                    subprocess.run([sys.executable, rebuild_script], check=True, timeout=1800)
                                    logger.info("Rebuild complete. Reloading data.")
                                    df_sim = pd.read_csv(csv_path)
                                except Exception as rebuild_err:
                                    raise RuntimeError(f"Rebuild failed: {rebuild_err}")
                            else:
                                raise FileNotFoundError(f"Cannot rebuild: Script not found at {rebuild_script}")

                        if "Date" in df_sim.columns:
                            df_sim["Date"] = pd.to_datetime(df_sim["Date"])
                            df_sim = df_sim.set_index("Date")
                        if "Close" not in df_sim.columns:
                            warnings.warn(f"{base}.csv missing 'Close' column")
                            df_sim = pd.DataFrame()
                        else:
                            df_sim = df_sim["Close"].sort_index()
                    else:
                        warnings.warn(f"{base} requested but {csv_path} not found.")

                    qbig_series = pd.Series(dtype=float)
                    if not proxy_prices.empty and Tickers.QBIG in proxy_prices.columns:
                        qbig_series = proxy_prices[Tickers.QBIG].dropna().sort_index()

                    if not qbig_series.empty and not df_sim.empty:
                        splice_date = qbig_series.index[0]
                        sim_part = df_sim[df_sim.index < splice_date]

                        if not sim_part.empty:
                            sim_end_val = sim_part.iloc[-1]
                            qbig_start_val = qbig_series.iloc[0]
                            scale_factor = qbig_start_val / sim_end_val if sim_end_val != 0 else 1.0
                            combined_prices[base] = pd.concat([sim_part * scale_factor, qbig_series])
                        else:
                            combined_prices[base] = qbig_series
                    elif not df_sim.empty:
                        combined_prices[base] = df_sim
                    elif not qbig_series.empty:
                        combined_prices[base] = qbig_series

                    continue
                except Exception as e:
                    raise RuntimeError(f"Failed to load/splice local {base}: {e}")

            is_sim_request = base.upper().endswith("SIM") or base.upper().endswith("TR")
            mapped_ticker = mapped_provider_tickers.get(base)

            if mapped_ticker and not batched_provider_prices.empty:
                if mapped_ticker in batched_provider_prices.columns and not batched_provider_prices[mapped_ticker].dropna().empty:
                    combined_prices[base] = batched_provider_prices[mapped_ticker]
                    continue

            # Testfol API fallback (for SIM tickers or provider chain misses)
            try:
                series, _, _ = api.fetch_backtest(
                    start_date="1900-01-01",
                    end_date=today_str,
                    start_val=10000,
                    cashflow=0,
                    cashfreq="Monthly",
                    rolling=1,
                    invest_div=True,
                    rebalance="Yearly",
                    allocation={base: 100.0},
                )
                combined_prices[base] = series
            except Exception as api_err:
                # Last resort: try provider chain even for SIM tickers
                if is_sim_request:
                    try:
                        mapped_ticker, _ = parse_ticker(base)
                        prices = provider.fetch_prices([mapped_ticker], sd_str, ed_str)
                        if not prices.empty and mapped_ticker in prices.columns:
                            combined_prices[base] = prices[mapped_ticker]
                            continue
                    except Exception:
                        pass
                raise api_err

        except Exception as e:
            warnings.warn(f"Failed to fetch data for {base}: {e}")

    result = pd.DataFrame(combined_prices)
    if not result.empty:
        result = result.sort_index()
        if not isinstance(result.index, pd.DatetimeIndex):
            result.index = pd.to_datetime(result.index)
        if result.index.tz is not None:
            result.index = result.index.tz_localize(None)
        result = result.loc[pd.Timestamp(sd_str):pd.Timestamp(ed_str)]
        if sync_end:
            result = clip_component_data_to_synced_end(result, unique_bases)
    cache_set(ck, result, prefix=COMPONENT_DATA_CACHE_PREFIX)
    return result

import time

def get_fed_funds_rate() -> pd.Series | None:
    """
    Fetches historical Fed Funds Rate (daily) from FRED or local cache.
    Returns: pd.Series with DatetimeIndex and rate as float (e.g. 5.25 for 5.25%)
    """
    data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../data")
    file_path = os.path.join(data_dir, "FEDFUNDS.csv")

    # Download if missing or old (>30 days)
    should_download = True
    if os.path.exists(file_path):
        mtime = os.path.getmtime(file_path)
        if (time.time() - mtime) < (30 * 86400):
            should_download = False

    if should_download:
        try:
            url = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=FEDFUNDS"
            # Fake User-Agent to avoid 403
            headers = {'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)'}
            import requests
            r = requests.get(url, headers=headers)
            r.raise_for_status()
            with open(file_path, "wb") as f:
                f.write(r.content)
            logger.info("Downloaded fresh FEDFUNDS.csv")
        except Exception as e:
            logger.warning(f"Failed to download FEDFUNDS: {e}. Using cached if available.")

    if os.path.exists(file_path):
        try:
            # FRED CSV format: observation_date, FEDFUNDS
            df = pd.read_csv(file_path, parse_dates=["observation_date"], index_col="observation_date")
            # Resample to daily (forward fill monthlies)
            full_idx = pd.date_range(start=df.index.min(), end=pd.Timestamp.today(), freq='D')
            daily_series = df['FEDFUNDS'].reindex(full_idx).ffill()
            return daily_series
        except Exception as e:
            raise RuntimeError(f"Error reading FEDFUNDS.csv: {e}")
    return None

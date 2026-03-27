from __future__ import annotations

import logging
import os
import subprocess
import sys
import warnings

import pandas as pd

from app.common.constants import Tickers
from app.services import testfol_api as api
from app.services.price_providers import get_price_provider

logger = logging.getLogger(__name__)


def fetch_component_data(tickers: list[str], start_date, end_date) -> pd.DataFrame:
    """
    Fetches historical data for each ticker individually via Testfol API (or local).
    Handles composite tickers like NDXMEGASIM by splicing local CSVs with live data.
    """
    combined_prices = pd.DataFrame()
    unique_tickers = list(set(tickers))

    for ticker in unique_tickers:
        try:
            # Parse Ticker
            base = ticker.split("?")[0]
            if base in combined_prices.columns:
                continue

            # SPECIAL: NDX30SIM — load simulation CSV + splice with QTOP
            if base == Tickers.NDX30SIM:
                try:
                    # 1. Load Simulation CSV
                    csv_path = f"data/{base}.csv"
                    df_sim = pd.Series(dtype=float)
                    if os.path.exists(csv_path):
                        _df = pd.read_csv(csv_path)
                        if 'Date' in _df.columns:
                            _df['Date'] = pd.to_datetime(_df['Date'])
                            _df = _df.set_index('Date')
                        if 'Close' in _df.columns:
                            df_sim = _df['Close'].sort_index()
                    else:
                        warnings.warn(f"{base} requested but {csv_path} not found.")

                    # 2. Fetch QTOP (Live ETF) via provider chain
                    qtop_series = pd.Series(dtype=float)
                    try:
                        provider = get_price_provider()
                        qtop_prices = provider.fetch_prices(["QTOP"], "2000-01-01", pd.Timestamp.now().strftime("%Y-%m-%d"))
                        if not qtop_prices.empty and "QTOP" in qtop_prices.columns:
                            qtop_series = qtop_prices["QTOP"].dropna().sort_index()
                    except Exception as e:
                        warnings.warn(f"Failed to fetch QTOP data: {e}")

                    # 3. Splice
                    if not qtop_series.empty and not df_sim.empty:
                        splice_date = qtop_series.index[0]
                        sim_part = df_sim[df_sim.index < splice_date]
                        if not sim_part.empty:
                            sim_end_val = sim_part.iloc[-1]
                            qtop_start_val = qtop_series.iloc[0]
                            scale_factor = qtop_start_val / sim_end_val if sim_end_val != 0 else 1.0
                            sim_part_scaled = sim_part * scale_factor
                            combined_prices[base] = pd.concat([sim_part_scaled, qtop_series])
                        else:
                            combined_prices[base] = qtop_series
                    elif not df_sim.empty:
                        combined_prices[base] = df_sim
                    elif not qtop_series.empty:
                        combined_prices[base] = qtop_series

                    continue

                except Exception as e:
                    raise RuntimeError(f"Failed to load/splice NDX30SIM: {e}")

            # SPECIAL: Load NDX Mega simulations from local CSV + Splice with QBIG
            if base in [Tickers.NDXMEGASIM, Tickers.NDXMEGA2SIM]:
                try:
                    # 1. Load Simulation Data (dynamic path based on ticker)
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
                                    subprocess.run([sys.executable, rebuild_script], check=True, timeout=120)
                                    logger.info("Rebuild complete. Reloading data.")
                                    df_sim = pd.read_csv(csv_path) # Retry load
                                except Exception as rebuild_err:
                                    raise RuntimeError(f"Rebuild failed: {rebuild_err}")
                            else:
                                raise FileNotFoundError(f"Cannot rebuild: Script not found at {rebuild_script}")

                        if 'Date' in df_sim.columns:
                            df_sim['Date'] = pd.to_datetime(df_sim['Date'])
                            df_sim = df_sim.set_index('Date')
                        if 'Close' not in df_sim.columns:
                            warnings.warn(f"{base}.csv missing 'Close' column")
                            df_sim = pd.DataFrame()
                        else:
                            df_sim = df_sim['Close'].sort_index() # Convert to Series
                    else:
                        warnings.warn(f"{base} requested but {csv_path} not found.")

                    # 2. Fetch QBIG (Live Proxy) via provider chain
                    try:
                        provider = get_price_provider()
                        qbig_prices = provider.fetch_prices([Tickers.QBIG], "2000-01-01", pd.Timestamp.now().strftime("%Y-%m-%d"))
                        qbig_series = pd.Series(dtype=float)
                        if not qbig_prices.empty and Tickers.QBIG in qbig_prices.columns:
                            qbig_series = qbig_prices[Tickers.QBIG].dropna().sort_index()
                    except Exception as e:
                        warnings.warn(f"Failed to fetch QBIG data: {e}")
                        qbig_series = pd.Series(dtype=float)

                    # 3. Splice
                    if not qbig_series.empty and not df_sim.empty:
                        # Find splice point (Start of QBIG)
                        splice_date = qbig_series.index[0]

                        # Get Sim Data UP TO Splice Date
                        sim_part = df_sim[df_sim.index < splice_date]

                        if not sim_part.empty:
                            # Align Sim End to QBIG Start
                            sim_end_val = sim_part.iloc[-1]
                            qbig_start_val = qbig_series.iloc[0]
                            scale_factor = qbig_start_val / sim_end_val if sim_end_val != 0 else 1.0

                            sim_part_scaled = sim_part * scale_factor

                            combined = pd.concat([sim_part_scaled, qbig_series])
                            combined_prices[base] = combined
                        else:
                             combined_prices[base] = qbig_series

                    elif not df_sim.empty:
                        combined_prices[base] = df_sim
                    elif not qbig_series.empty:
                        combined_prices[base] = qbig_series

                    continue

                except Exception as e:
                    raise RuntimeError(f"Failed to load/splice local {base}: {e}")

            # Standard ticker fetch via provider chain (Polygon → yfinance)
            # SIM tickers need extended history from Testfol, not real-ticker history
            is_sim_request = base.upper().endswith("SIM") or base.upper().endswith("TR")

            if not is_sim_request:
                try:
                    from app.core.shadow_backtest import parse_ticker
                    mapped_ticker, _ = parse_ticker(base)

                    sd_str = start_date.strftime("%Y-%m-%d") if hasattr(start_date, 'strftime') else str(start_date)
                    ed_str = end_date.strftime("%Y-%m-%d") if hasattr(end_date, 'strftime') else str(end_date)

                    provider = get_price_provider()
                    prices = provider.fetch_prices([mapped_ticker], sd_str, ed_str)
                    if not prices.empty and mapped_ticker in prices.columns and not prices[mapped_ticker].dropna().empty:
                        combined_prices[base] = prices[mapped_ticker]
                        continue
                except Exception:
                    pass  # Fall through to Testfol API

            # Testfol API fallback (for SIM tickers or provider chain failures)
            try:
                broad_start = "1900-01-01"
                broad_end = pd.Timestamp.now().strftime("%Y-%m-%d")

                series, _, _ = api.fetch_backtest(
                    start_date=broad_start,
                    end_date=broad_end,
                    start_val=10000,
                    cashflow=0,
                    cashfreq="Monthly",
                    rolling=1,
                    invest_div=True,
                    rebalance="Yearly",
                    allocation={base: 100.0}
                )
                combined_prices[base] = series
            except Exception as api_err:
                # Last resort: try provider chain even for SIM tickers
                if is_sim_request:
                    try:
                        from app.core.shadow_backtest import parse_ticker
                        mapped_ticker, _ = parse_ticker(base)
                        provider = get_price_provider()
                        prices = provider.fetch_prices([mapped_ticker], str(start_date), str(end_date))
                        if not prices.empty and mapped_ticker in prices.columns:
                            combined_prices[base] = prices[mapped_ticker]
                            continue
                    except Exception:
                        pass
                raise api_err

        except Exception as e:
            warnings.warn(f"Failed to fetch data for {ticker}: {e}")

    return combined_prices

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

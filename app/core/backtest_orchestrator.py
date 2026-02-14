"""
Shared backtest orchestration logic.

Pure Python module — no Streamlit, no FastAPI, no Pydantic imports.
All params are plain types (dict, str, float, bool).
Returns raw dicts containing pandas objects.

Dependency injection: callers can pass cached versions of fetch_backtest
and run_shadow_backtest via the `fetch_backtest_fn` / `run_shadow_fn` params.
"""

import logging

import pandas as pd

from app.common.constants import Freq, RebalMode, Tickers
from app.core import calculations
from app.core.shadow_backtest import run_shadow_backtest as _default_shadow
from app.services.data_service import fetch_component_data
from app.services.testfol_api import fetch_backtest as _default_fetch

logger = logging.getLogger(__name__)


def calc_rebal_offset(reb: dict, r_freq: str) -> int:
    """Calculate rebalance offset from custom rebalance config."""
    r_mode = reb.get("mode", RebalMode.STANDARD)
    if r_mode != RebalMode.CUSTOM:
        return 0
    r_month = reb.get("month", 1)
    r_day = reb.get("day", 1)
    try:
        if r_freq == Freq.YEARLY:
            end_of_year = pd.Timestamp("2024-12-31")
            target_date = pd.Timestamp(f"2024-{r_month}-{r_day}")
            days_remaining = (end_of_year - target_date).days
            return max(0, int(days_remaining * (252.0 / 366.0)))
        else:
            days_remaining = 31 - r_day
            return int(days_remaining * (21.0 / 31.0))
    except Exception:
        return 0


def run_single_backtest(
    allocation: dict,
    maint_pcts: dict,
    rebalance: dict,
    start_date: str,
    end_date: str,
    start_val: float,
    cashflow_amount: float,
    cashflow_freq: str,
    invest_div: bool,
    pay_down_margin: bool,
    tax_config: dict,
    bearer_token: str | None,
    name: str = "Portfolio",
    fetch_backtest_fn=None,
    run_shadow_fn=None,
) -> dict:
    """
    Run a single portfolio backtest (API or local engine).

    Returns a raw dict with all result fields (not yet serialized).
    Uses dependency injection for the two expensive functions so callers
    can pass @st.cache_data-wrapped versions.
    """
    fetch_fn = fetch_backtest_fn or _default_fetch
    shadow_fn = run_shadow_fn or _default_shadow

    alloc_map = allocation

    # Weighted maintenance
    total_w = sum(alloc_map.values())
    d_maint = 25.0
    current_wmaint = 0.0
    if total_w > 0:
        for ticker, weight in alloc_map.items():
            m = maint_pcts.get(ticker.split("?")[0], d_maint)
            current_wmaint += (weight / 100) * (m / 100)
    else:
        current_wmaint = d_maint / 100.0

    # Rebalance
    reb = rebalance
    r_mode = reb.get("mode", RebalMode.STANDARD)
    r_freq = reb.get("freq", Freq.YEARLY)
    r_threshold = reb.get("threshold_pct", 5.0)

    # Determine engine
    has_ndxmega = any(
        (Tickers.NDXMEGASIM in t or Tickers.NDXMEGA2SIM in t or Tickers.NDX30SIM in t)
        for t in alloc_map
    )
    uses_threshold = r_mode in (RebalMode.THRESHOLD, RebalMode.THRESHOLD_CALENDAR)
    use_local_engine = has_ndxmega or uses_threshold

    # Cashflow settings
    bt_cashflow = 0.0 if pay_down_margin else cashflow_amount
    shadow_cashflow = 0.0 if pay_down_margin else cashflow_amount

    port_series = pd.Series(dtype=float)
    stats: dict = {}
    trades_df = pd.DataFrame()
    extra_data: dict = {}
    df_rets = None
    twr_series = None
    pl_by_year = pd.DataFrame()
    composition_df = pd.DataFrame()
    unrealized_pl_df = pd.DataFrame()
    logs: list = []
    prices_df = pd.DataFrame()

    if not use_local_engine:
        # --- API Path ---
        rebal_offset = calc_rebal_offset(reb, r_freq)

        port_series, stats_api, extra_data = fetch_fn(
            start_date=start_date,
            end_date=end_date,
            start_val=max(1.0, start_val),
            cashflow=bt_cashflow,
            cashfreq=cashflow_freq,
            rolling=60,
            invest_div=invest_div,
            rebalance=r_freq,
            rebalance_offset=rebal_offset,
            allocation=alloc_map,
            return_raw=False,
            include_raw=True,
            bearer_token=bearer_token,
        )
        stats = stats_api
        logger.debug(f"API: Tickers={list(alloc_map.keys())} | API Stats CAGR={stats.get('cagr')}")

        # Extract TWR Series from API
        api_twr_series = None
        if extra_data and "daily_returns" in extra_data:
            d_rets = extra_data["daily_returns"]
            if d_rets:
                try:
                    df_rets = pd.DataFrame(d_rets, columns=["Date", "Pct", "Val"])
                    df_rets["Date"] = pd.to_datetime(df_rets["Date"])
                    df_rets = df_rets.set_index("Date").sort_index()
                    df_rets["Factor"] = 1 + (df_rets["Pct"] / 100.0)
                    api_twr_series = df_rets["Factor"].cumprod()
                    api_twr_series.name = "TWR (API)"
                except Exception as e:
                    logger.warning(f"Failed to build API TWR: {e}")

        # Fetch component prices for cheat sheet
        try:
            prices_df = fetch_component_data(list(alloc_map.keys()), start_date, end_date)
        except Exception as e:
            logger.warning(f"Failed to fetch component prices: {e}")
            prices_df = pd.DataFrame()

        # Run shadow backtest (tax tracking)
        if not port_series.empty:
            trades_df, pl_by_year, composition_df, unrealized_pl_df, logs, _, twr_series = shadow_fn(
                allocation=alloc_map,
                start_val=start_val,
                start_date=start_date,
                end_date=end_date,
                api_port_series=port_series,
                rebalance_freq="Custom",
                cashflow=shadow_cashflow,
                cashflow_freq=cashflow_freq,
                invest_dividends=invest_div,
                pay_down_margin=pay_down_margin,
                tax_config=tax_config,
                custom_rebal_config=reb if r_mode == "Custom" else {},
            )
            if api_twr_series is not None:
                twr_series = api_twr_series
    else:
        # --- Pure Local Path (NDXMEGASIM) ---
        tickers = list(alloc_map.keys())
        prices_df = fetch_component_data(tickers, start_date, end_date)

        trades_df, pl_by_year, composition_df, unrealized_pl_df, logs, port_series, twr_series = shadow_fn(
            allocation=alloc_map,
            start_val=start_val,
            start_date=start_date,
            end_date=end_date,
            api_port_series=None,
            rebalance_freq=r_mode if uses_threshold else reb.get("freq", "Yearly"),
            cashflow=shadow_cashflow,
            cashflow_freq=cashflow_freq,
            invest_dividends=invest_div,
            pay_down_margin=pay_down_margin,
            tax_config=tax_config,
            custom_rebal_config=reb if r_mode == "Custom" else {},
            prices_df=prices_df,
            rebalance_month=reb.get("month", 1),
            rebalance_day=reb.get("day", 1),
            custom_freq=reb.get("freq", "Yearly"),
            threshold_pct=r_threshold,
        )

        if not port_series.empty:
            stats = calculations.generate_stats(twr_series if twr_series is not None else port_series)

    return {
        "name": name,
        "series": port_series,
        "port_series": port_series,
        "stats": stats,
        "twr_series": twr_series,
        "daily_returns_df": df_rets,
        "is_local": use_local_engine,
        "trades": trades_df,
        "trades_df": trades_df,
        "pl_by_year": pl_by_year,
        "unrealized_pl_df": unrealized_pl_df,
        "component_prices": prices_df,
        "allocation": alloc_map,
        "logs": logs,
        "composition": composition_df,
        "composition_df": composition_df,
        "raw_response": extra_data,
        "start_val": start_val,
        "sim_range": f"{start_date} to {end_date}",
        "shadow_range": f"{start_date} to {end_date}",
        "wmaint": current_wmaint,
        # Internal: kept for pass-2 re-fetch
        "_reb": reb,
        "_r_mode": r_mode,
    }


def run_multi_backtest(
    portfolios: list[dict],
    start_date: str,
    end_date: str,
    start_val: float,
    cashflow_amount: float,
    cashflow_freq: str,
    invest_div: bool,
    pay_down_margin: bool,
    tax_config: dict,
    bearer_token: str | None,
    fetch_backtest_fn=None,
    run_shadow_fn=None,
) -> tuple[list[dict], list]:
    """
    Run backtests for multiple portfolios with common-start alignment.

    Each entry in `portfolios` is a plain dict with keys:
        name, allocation, maint_pcts, rebalance

    Returns (results_list, bench_series_list).
    """
    fetch_fn = fetch_backtest_fn or _default_fetch
    shadow_fn = run_shadow_fn or _default_shadow

    results_list: list[dict] = []
    bench_series_list: list = []

    # --- Pass 1: Run each portfolio ---
    for p in portfolios:
        raw = run_single_backtest(
            allocation=p["allocation"],
            maint_pcts=p.get("maint_pcts", {}),
            rebalance=p.get("rebalance", {}),
            start_date=start_date,
            end_date=end_date,
            start_val=start_val,
            cashflow_amount=cashflow_amount,
            cashflow_freq=cashflow_freq,
            invest_div=invest_div,
            pay_down_margin=pay_down_margin,
            tax_config=tax_config,
            bearer_token=bearer_token,
            name=p.get("name", "Portfolio"),
            fetch_backtest_fn=fetch_fn,
            run_shadow_fn=shadow_fn,
        )
        results_list.append(raw)

        # Comparison (vs Standard)
        reb = p.get("rebalance", {})
        if reb.get("compare_std", False) and reb.get("mode") == "Custom":
            try:
                std_series, std_stats, _ = fetch_fn(
                    start_date=start_date,
                    end_date=end_date,
                    start_val=start_val,
                    cashflow=0.0 if pay_down_margin else cashflow_amount,
                    cashfreq=cashflow_freq,
                    rolling=60,
                    invest_div=invest_div,
                    rebalance="Yearly",
                    allocation=p["allocation"],
                    return_raw=False,
                    bearer_token=bearer_token,
                )
                std_series.name = f"{p.get('name')} (Standard)"
                bench_series_list.append(std_series)
            except Exception as e:
                logger.warning(f"Failed standard comparison: {e}")

    # --- Pass 2: Common start date alignment ---
    start_dates = []
    for res in results_list:
        s = res.get("series")
        if s is not None and not s.empty:
            start_dates.append(s.index.min())
    for b in bench_series_list:
        if b is not None and not b.empty:
            start_dates.append(b.index.min())

    common_start = max(start_dates) if start_dates else None
    global_start_val = start_val

    if common_start:
        for i, res in enumerate(results_list):
            series = res.get("series")
            if series is None or series.empty:
                continue
            if series.index[0] >= common_start - pd.Timedelta(days=3):
                continue

            alloc_map = res["allocation"]
            reb = res["_reb"]
            r_mode = res["_r_mode"]
            r_freq = reb.get("freq", "Yearly")
            r_threshold = reb.get("threshold_pct", 5.0)
            uses_threshold = r_mode in (RebalMode.THRESHOLD, RebalMode.THRESHOLD_CALENDAR)

            if not res.get("is_local", False):
                # --- API Portfolio: Re-fetch with common_start ---
                rebal_offset = calc_rebal_offset(reb, r_freq)
                try:
                    new_series, new_stats, new_extra = fetch_fn(
                        start_date=common_start.strftime("%Y-%m-%d"),
                        end_date=end_date,
                        start_val=global_start_val,
                        cashflow=0.0 if pay_down_margin else cashflow_amount,
                        cashfreq=cashflow_freq,
                        rolling=60,
                        invest_div=invest_div,
                        rebalance=r_freq,
                        rebalance_offset=rebal_offset,
                        allocation=alloc_map,
                        return_raw=False,
                        include_raw=True,
                        bearer_token=bearer_token,
                    )
                    if not new_series.empty:
                        res["series"] = new_series
                        res["port_series"] = new_series
                        res["original_api_stats"] = res.get("stats", {})
                        res["stats"] = new_stats
                        res["raw_response"] = new_extra
                        logger.debug(f"Re-fetched {res['name']} from {common_start.date()} - CAGR: {new_stats.get('cagr')}")

                        # Re-run shadow backtest aligned
                        try:
                            shadow_cf = 0.0 if pay_down_margin else cashflow_amount
                            new_trades, new_pl, new_comp, new_unrealized, new_logs, _, new_twr = shadow_fn(
                                allocation=alloc_map,
                                start_val=global_start_val,
                                start_date=common_start.strftime("%Y-%m-%d"),
                                end_date=end_date,
                                api_port_series=new_series,
                                rebalance_freq="Custom",
                                cashflow=shadow_cf,
                                cashflow_freq=cashflow_freq,
                                invest_dividends=invest_div,
                                pay_down_margin=pay_down_margin,
                                tax_config=tax_config,
                                custom_rebal_config=reb if r_mode == "Custom" else {},
                            )
                            res["trades_df"] = new_trades
                            res["trades"] = new_trades
                            res["pl_by_year"] = new_pl
                            res["composition_df"] = new_comp
                            res["composition"] = new_comp
                            res["unrealized_pl_df"] = new_unrealized
                            res["logs"] = new_logs
                            res["twr_series"] = new_twr
                            res["shadow_range"] = f"{common_start.date()} to {end_date}"
                        except Exception as shadow_e:
                            logger.warning(f"Failed to re-run shadow for {res['name']}: {shadow_e}")

                except Exception as e:
                    logger.warning(f"Failed to re-fetch {res['name']}: {e}")
                    # Fallback: TWR-based rebasing
                    twr = res.get("twr_series")
                    if twr is not None and not twr.empty:
                        twr_slice = twr[twr.index >= common_start]
                        if not twr_slice.empty:
                            scale = twr_slice / twr_slice.iloc[0]
                            res["series"] = scale * global_start_val
                            res["port_series"] = res["series"]
                            res["stats"] = calculations.generate_stats(res["series"])
            else:
                # --- Local Portfolio (NDXMEGASIM): TWR-based rebasing ---
                twr = res.get("twr_series")
                if twr is not None and not twr.empty:
                    twr_slice = twr[twr.index >= common_start]
                    if not twr_slice.empty:
                        scale = twr_slice / twr_slice.iloc[0]
                        new_series = scale * global_start_val
                        res["series"] = new_series
                        res["port_series"] = new_series
                        res["stats"] = calculations.generate_stats(new_series)

                        # Re-run shadow for local portfolio
                        try:
                            prices_df_new = fetch_component_data(
                                list(alloc_map.keys()),
                                common_start.strftime("%Y-%m-%d"),
                                end_date,
                            )
                            shadow_cf = 0.0 if pay_down_margin else cashflow_amount
                            new_trades, new_pl, new_comp, new_unrealized, new_logs, _, new_twr = shadow_fn(
                                allocation=alloc_map,
                                start_val=global_start_val,
                                start_date=common_start.strftime("%Y-%m-%d"),
                                end_date=end_date,
                                api_port_series=None,
                                rebalance_freq=r_mode if uses_threshold else reb.get("freq", "Yearly"),
                                cashflow=shadow_cf,
                                cashflow_freq=cashflow_freq,
                                invest_dividends=invest_div,
                                pay_down_margin=pay_down_margin,
                                tax_config=tax_config,
                                custom_rebal_config=reb if r_mode == "Custom" else {},
                                prices_df=prices_df_new,
                                rebalance_month=reb.get("month", 1),
                                rebalance_day=reb.get("day", 1),
                                custom_freq=reb.get("freq", "Yearly"),
                                threshold_pct=r_threshold,
                            )
                            res["trades_df"] = new_trades
                            res["trades"] = new_trades
                            res["pl_by_year"] = new_pl
                            res["composition_df"] = new_comp
                            res["composition"] = new_comp
                            res["unrealized_pl_df"] = new_unrealized
                            res["logs"] = new_logs
                            res["twr_series"] = new_twr
                            res["shadow_range"] = f"{common_start.date()} to {end_date}"
                            if new_twr is not None and not new_twr.empty:
                                synced = (new_twr / new_twr.iloc[0]) * global_start_val
                                res["series"] = synced
                                res["port_series"] = synced
                            res["stats"] = calculations.generate_stats(
                                new_twr if new_twr is not None and not new_twr.empty else new_series
                            )
                        except Exception as shadow_e:
                            logger.warning(f"Failed to re-run shadow for local {res['name']}: {shadow_e}")

    return results_list, bench_series_list

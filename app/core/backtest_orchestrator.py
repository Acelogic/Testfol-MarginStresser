"""
Shared backtest orchestration logic.

Pure Python module — no Streamlit, no FastAPI, no Pydantic imports.
All params are plain types (dict, str, float, bool).
Returns raw dicts containing pandas objects.

Dependency injection: callers can pass cached versions of fetch_backtest
and run_shadow_backtest via the `fetch_backtest_fn` / `run_shadow_fn` params.
"""

import datetime
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
    pm_maint_pcts: dict | None = None,
    pm_config: dict | None = None,
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
            current_wmaint += (weight / total_w) * (m / 100)
    else:
        current_wmaint = d_maint / 100.0

    # PM weighted maintenance
    current_wmaint_pm = 0.0
    if pm_maint_pcts and total_w > 0:
        for ticker, weight in alloc_map.items():
            m_pm = pm_maint_pcts.get(ticker.split("?")[0], 0.0)
            if m_pm > 0:
                current_wmaint_pm += (weight / total_w) * (m_pm / 100)

    # PM buy block config
    _pm_cfg = pm_config or {}
    pm_buy_block = _pm_cfg.get("pm_buy_block", False)
    pm_buy_block_threshold = _pm_cfg.get("pm_buy_block_threshold", 100000.0)
    pm_draw_monthly = _pm_cfg.get("draw_monthly", 0.0)
    pm_draw_monthly_retirement = _pm_cfg.get("draw_monthly_retirement", 0.0)
    pm_dca_in_retirement = _pm_cfg.get("dca_in_retirement", True)
    _raw_retirement_date = _pm_cfg.get("retirement_date", None)
    _raw_draw_start = _pm_cfg.get("draw_start_date", None)
    # Clamp draw_start_date to backtest range
    _bt_start = pd.Timestamp(start_date).date() if isinstance(start_date, str) else start_date
    _bt_end = pd.Timestamp(end_date).date() if isinstance(end_date, str) else end_date
    if _raw_draw_start is not None:
        if isinstance(_raw_draw_start, str):
            _raw_draw_start = pd.Timestamp(_raw_draw_start).date()
        elif isinstance(_raw_draw_start, datetime.datetime):
            _raw_draw_start = _raw_draw_start.date()
        pm_draw_start_date = max(_raw_draw_start, _bt_start)
        pm_draw_start_date = min(pm_draw_start_date, _bt_end)
    else:
        pm_draw_start_date = _bt_start
    # Clamp retirement_date to backtest range
    if _raw_retirement_date is not None:
        if isinstance(_raw_retirement_date, str):
            _raw_retirement_date = pd.Timestamp(_raw_retirement_date).date()
        elif isinstance(_raw_retirement_date, datetime.datetime):
            _raw_retirement_date = _raw_retirement_date.date()
        pm_retirement_date = max(_raw_retirement_date, _bt_start)
        pm_retirement_date = min(pm_retirement_date, _bt_end)
    else:
        pm_retirement_date = None
    pm_margin_rate = _pm_cfg.get("margin_rate_annual", 8.0)
    pm_cf_for_loan = _pm_cfg.get("cashflow_for_loan", 0.0)
    pm_cf_freq = _pm_cfg.get("cashflow_freq", "Monthly")

    _orch_log = logging.getLogger("orchestrator")
    _orch_log.info("Orchestrator draw config: draw=$%,.0f/mo ret_draw=$%,.0f/mo draw_start=%s ret_date=%s dca_in_ret=%s",
                   pm_draw_monthly, pm_draw_monthly_retirement, pm_draw_start_date, pm_retirement_date, pm_dca_in_retirement)

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
    pm_blocked_dates: list = []

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
            trades_df, pl_by_year, composition_df, unrealized_pl_df, logs, _, twr_series, *_ = shadow_fn(
                allocation=alloc_map,
                start_val=start_val,
                start_date=start_date,
                end_date=end_date,
                api_port_series=port_series,
                rebalance_freq="Custom" if r_mode == "Custom" else r_freq,
                cashflow=shadow_cashflow,
                cashflow_freq=cashflow_freq,
                invest_dividends=invest_div,
                pay_down_margin=pay_down_margin,
                tax_config=tax_config,
                custom_rebal_config=reb if r_mode == "Custom" else {},
                rebalance_month=reb.get("month", 1),
                rebalance_day=reb.get("day", 1),
                custom_freq=reb.get("freq", "Yearly"),
                pm_buy_block=pm_buy_block,
                pm_buy_block_threshold=pm_buy_block_threshold,
                starting_loan=_pm_cfg.get("starting_loan", 0.0),
                margin_rate_annual=pm_margin_rate,
                draw_monthly=pm_draw_monthly,
                draw_start_date=pm_draw_start_date,
                draw_monthly_retirement=pm_draw_monthly_retirement,
                retirement_date=pm_retirement_date,
                dca_in_retirement=pm_dca_in_retirement,
                loan_repayment=pm_cf_for_loan if pay_down_margin else 0.0,
                loan_repayment_freq=pm_cf_freq,
            )
            if api_twr_series is not None:
                twr_series = api_twr_series
    else:
        # --- Pure Local Path (NDXMEGASIM) ---
        tickers = list(alloc_map.keys())
        prices_df = fetch_component_data(tickers, start_date, end_date)

        _shadow_result = shadow_fn(
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
            pm_buy_block=pm_buy_block,
            pm_buy_block_threshold=pm_buy_block_threshold,
            starting_loan=_pm_cfg.get("starting_loan", 0.0),
            margin_rate_annual=pm_margin_rate,
            draw_monthly=pm_draw_monthly,
            draw_start_date=pm_draw_start_date,
            draw_monthly_retirement=pm_draw_monthly_retirement,
            retirement_date=pm_retirement_date,
            dca_in_retirement=pm_dca_in_retirement,
            loan_repayment=pm_cf_for_loan if pay_down_margin else 0.0,
            loan_repayment_freq=pm_cf_freq,
        )
        # Unpack: shadow backtest returns 7-tuple or 8-tuple (with pm_blocked_dates)
        if isinstance(_shadow_result, tuple) and len(_shadow_result) == 8:
            trades_df, pl_by_year, composition_df, unrealized_pl_df, logs, port_series, twr_series, pm_blocked_dates = _shadow_result
        else:
            trades_df, pl_by_year, composition_df, unrealized_pl_df, logs, port_series, twr_series = _shadow_result
            pm_blocked_dates = []

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
        "wmaint_pm": current_wmaint_pm,
        "pm_blocked_dates": pm_blocked_dates,
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
    pm_config: dict | None = None,
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
            pm_maint_pcts=p.get("pm_maint_pcts"),
            pm_config=pm_config,
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

    # Resolve draw_start_date for Pass 2 re-runs
    _p2_cfg = pm_config or {}
    _p2_raw_draw_start = _p2_cfg.get("draw_start_date", None)
    if _p2_raw_draw_start is not None:
        if isinstance(_p2_raw_draw_start, str):
            _p2_raw_draw_start = pd.Timestamp(_p2_raw_draw_start).date()
        elif isinstance(_p2_raw_draw_start, datetime.datetime):
            _p2_raw_draw_start = _p2_raw_draw_start.date()
        _p2_bt_start = pd.Timestamp(start_date).date() if isinstance(start_date, str) else start_date
        _p2_bt_end = pd.Timestamp(end_date).date() if isinstance(end_date, str) else end_date
        pm_draw_start_date = max(_p2_raw_draw_start, _p2_bt_start)
        pm_draw_start_date = min(pm_draw_start_date, _p2_bt_end)
    else:
        pm_draw_start_date = pd.Timestamp(start_date).date() if isinstance(start_date, str) else start_date
    # Resolve retirement_date for Pass 2
    _p2_raw_ret = _p2_cfg.get("retirement_date", None)
    if _p2_raw_ret is not None:
        if isinstance(_p2_raw_ret, str):
            _p2_raw_ret = pd.Timestamp(_p2_raw_ret).date()
        elif isinstance(_p2_raw_ret, datetime.datetime):
            _p2_raw_ret = _p2_raw_ret.date()
        pm_retirement_date = _p2_raw_ret
    else:
        pm_retirement_date = None
    pm_draw_monthly_retirement = _p2_cfg.get("draw_monthly_retirement", 0.0)

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
                            _pm_cfg_p2 = pm_config or {}
                            new_trades, new_pl, new_comp, new_unrealized, new_logs, _, new_twr, *_ = shadow_fn(
                                allocation=alloc_map,
                                start_val=global_start_val,
                                start_date=common_start.strftime("%Y-%m-%d"),
                                end_date=end_date,
                                api_port_series=new_series,
                                rebalance_freq="Custom" if r_mode == "Custom" else r_freq,
                                cashflow=shadow_cf,
                                cashflow_freq=cashflow_freq,
                                invest_dividends=invest_div,
                                pay_down_margin=pay_down_margin,
                                tax_config=tax_config,
                                custom_rebal_config=reb if r_mode == "Custom" else {},
                                rebalance_month=reb.get("month", 1),
                                rebalance_day=reb.get("day", 1),
                                custom_freq=reb.get("freq", "Yearly"),
                                threshold_pct=r_threshold,
                                pm_buy_block=_pm_cfg_p2.get("pm_buy_block", False),
                                pm_buy_block_threshold=_pm_cfg_p2.get("pm_buy_block_threshold", 100000.0),
                                starting_loan=_pm_cfg_p2.get("starting_loan", 0.0),
                                margin_rate_annual=_pm_cfg_p2.get("margin_rate_annual", 8.0),
                                draw_monthly=_pm_cfg_p2.get("draw_monthly", 0.0),
                                draw_start_date=pm_draw_start_date,
                                draw_monthly_retirement=_pm_cfg_p2.get("draw_monthly_retirement", 0.0),
                                retirement_date=pm_retirement_date,
                                dca_in_retirement=_pm_cfg_p2.get("dca_in_retirement", True),
                                loan_repayment=_pm_cfg_p2.get("cashflow_for_loan", 0.0) if pay_down_margin else 0.0,
                                loan_repayment_freq=_pm_cfg_p2.get("cashflow_freq", "Monthly"),
                            )
                            res["trades_df"] = new_trades
                            res["trades"] = new_trades
                            res["pl_by_year"] = new_pl
                            res["composition_df"] = new_comp
                            res["composition"] = new_comp
                            res["unrealized_pl_df"] = new_unrealized
                            res["logs"] = new_logs
                            res["twr_series"] = new_twr
                            # Rebuild API TWR from re-fetched response
                            new_extra = res.get("raw_response")
                            if new_extra and "daily_returns" in new_extra:
                                d_rets = new_extra["daily_returns"]
                                if d_rets:
                                    try:
                                        df_tmp = pd.DataFrame(d_rets, columns=["Date", "Pct", "Val"])
                                        df_tmp["Date"] = pd.to_datetime(df_tmp["Date"])
                                        df_tmp = df_tmp.set_index("Date").sort_index()
                                        api_twr = (1 + df_tmp["Pct"] / 100.0).cumprod()
                                        api_twr.name = "TWR (API)"
                                        res["twr_series"] = api_twr
                                    except Exception:
                                        pass
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
                            _pm_cfg2 = pm_config or {}
                            new_trades, new_pl, new_comp, new_unrealized, new_logs, new_port, new_twr, *_ = shadow_fn(
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
                                pm_buy_block=_pm_cfg2.get("pm_buy_block", False),
                                pm_buy_block_threshold=_pm_cfg2.get("pm_buy_block_threshold", 100000.0),
                                starting_loan=_pm_cfg2.get("starting_loan", 0.0),
                                margin_rate_annual=_pm_cfg2.get("margin_rate_annual", 8.0),
                                draw_monthly=_pm_cfg2.get("draw_monthly", 0.0),
                                draw_start_date=pm_draw_start_date,
                                draw_monthly_retirement=_pm_cfg2.get("draw_monthly_retirement", 0.0),
                                retirement_date=pm_retirement_date,
                                dca_in_retirement=_pm_cfg2.get("dca_in_retirement", True),
                                loan_repayment=_pm_cfg2.get("cashflow_for_loan", 0.0) if pay_down_margin else 0.0,
                                loan_repayment_freq=_pm_cfg2.get("cashflow_freq", "Monthly"),
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
                            # Use port_series (includes cashflows) for chart,
                            # TWR (strips cashflows) for stats
                            if new_port is not None and not new_port.empty:
                                res["series"] = new_port
                                res["port_series"] = new_port
                            elif new_twr is not None and not new_twr.empty:
                                synced = (new_twr / new_twr.iloc[0]) * global_start_val
                                res["series"] = synced
                                res["port_series"] = synced
                            res["stats"] = calculations.generate_stats(
                                new_twr if new_twr is not None and not new_twr.empty else new_series
                            )
                        except Exception as shadow_e:
                            logger.warning(f"Failed to re-run shadow for local {res['name']}: {shadow_e}")

    # Align benchmarks to common_start (Bug 11 fix)
    if common_start:
        for j, b in enumerate(bench_series_list):
            if b is not None and not b.empty and b.index[0] < common_start:
                bench_series_list[j] = b[b.index >= common_start]

    return results_list, bench_series_list

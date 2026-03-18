#!/usr/bin/env python3
"""
Run ACTUAL backtests for both NDXMEGASPLIT portfolios and trace
the exact tax → loan → usage pipeline to find the margin bug.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
from app.core.shadow_backtest import run_shadow_backtest
from app.services.testfol_api import simulate_margin
from app.services.data_service import fetch_component_data
from app.core.tax_library import (
    calculate_tax_series_with_carryforward,
    calculate_state_tax_series_with_carryforward,
)

# ─────────────────────────────────────────────────────────────────────────────
# Portfolio definitions (exact from presets.json)
# ─────────────────────────────────────────────────────────────────────────────

BASE_ALLOC = {
    "NDXMEGASIM?L=2": 60.0,
    "GLDSIM": 20.0,
    "VXUSSIM": 15.0,
    "QQQSIM?L=3": 5.0,
}
BASE_MAINT = {"NDXMEGASIM": 50.0, "GLDSIM": 25.0, "VXUSSIM": 25.0, "QQQSIM": 75.0}
BASE_REBAL = {"mode": "Custom", "freq": "Yearly", "month": 1, "day": 1}

NIGGED_ALLOC = {
    "NDXMEGASIM?L=2": 30.0,
    "GLDSIM": 20.0,
    "VXUSSIM": 15.0,
    "QQQSIM?L=3": 5.0,
    "AAPL?L=2": 3.75,
    "MSFT?L=2": 3.75,
    "AVGO?L=2": 3.75,
    "AMZN?L=2": 3.75,
    "META?L=2": 3.75,
    "NVDA?L=2": 3.75,
    "GOOG?L=2": 3.75,
    "TSLA?L=2": 3.75,
}
NIGGED_MAINT = {
    "NDXMEGASIM": 50.0, "GLDSIM": 25.0, "VXUSSIM": 25.0, "QQQSIM": 75.0,
    "AAPL": 50.0, "MSFT": 50.0, "AVGO": 50.0, "AMZN": 50.0,
    "META": 50.0, "NVDA": 50.0, "GOOG": 50.0, "TSLA": 50.0,
}
NIGGED_REBAL = {"mode": "Standard", "freq": "Yearly", "month": 1, "day": 1}

# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────
START_DATE = "2016-01-01"
END_DATE = "2026-03-10"
START_VAL = 10_000.0
OTHER_INCOME = 100_000.0
FILING_STATUS = "Single"
TAX_METHOD = "historical"
STATE_CODE = "NY"

RATE_CONFIG = {
    "type": "Tiered",
    "tiers": [(0, 1.50), (100000, 1.00), (1000000, 0.75), (50000000, 0.50)],
    "base_series": None,
}


def compute_wmaint(alloc, maint_pcts):
    wm = 0.0
    for ticker, weight in alloc.items():
        base = ticker.split("?")[0]
        m = maint_pcts.get(base, 25.0)
        wm += (weight / 100) * (m / 100)
    return wm


def run_and_trace(name, alloc, maint_pcts, rebal_cfg):
    """Run shadow backtest, compute taxes, run margin sim, dump everything."""
    print(f"\n{'='*80}")
    print(f"  RUNNING: {name}")
    print(f"{'='*80}")

    wmaint = compute_wmaint(alloc, maint_pcts)
    print(f"  wmaint = {wmaint:.4f} ({wmaint*100:.2f}%)")
    print(f"  Tickers: {len(alloc)}")
    print(f"  Rebalance: {rebal_cfg['mode']} / {rebal_cfg['freq']}")

    r_mode = rebal_cfg["mode"]
    if r_mode == "Custom":
        rebalance_freq = "Custom"
        custom_rebal_config = rebal_cfg
    else:
        rebalance_freq = rebal_cfg["freq"]
        custom_rebal_config = {}

    # Pre-fetch prices via data service (handles simulated tickers)
    print(f"  Fetching price data...")
    prices_df = fetch_component_data(list(alloc.keys()), START_DATE, END_DATE)
    print(f"  Got {len(prices_df)} rows, {len(prices_df.columns)} columns: {list(prices_df.columns)}")

    trades_df, pl_by_year, comp_df, unrealized_df, logs, port_series, twr_series, pm_blocked = \
        run_shadow_backtest(
            allocation=alloc,
            start_val=START_VAL,
            start_date=START_DATE,
            end_date=END_DATE,
            api_port_series=None,
            rebalance_freq=rebalance_freq,
            cashflow=0.0,
            cashflow_freq="Monthly",
            prices_df=prices_df,
            invest_dividends=True,
            pay_down_margin=False,
            tax_config=None,
            custom_rebal_config=custom_rebal_config,
            rebalance_month=rebal_cfg.get("month", 1),
            rebalance_day=rebal_cfg.get("day", 1),
            custom_freq=rebal_cfg.get("freq", "Yearly"),
        )

    print(f"\n--- Portfolio Series ---")
    if port_series.empty:
        print("  EMPTY! Backtest produced no data.")
        return None
    print(f"  Start: {port_series.index[0].date()} = ${port_series.iloc[0]:,.2f}")
    print(f"  End:   {port_series.index[-1].date()} = ${port_series.iloc[-1]:,.2f}")

    # Check Nov 7, 2022
    target = pd.Timestamp("2022-11-07")
    if target in port_series.index:
        nov7_val = port_series.loc[target]
    else:
        nov7_val = port_series.asof(target)
    print(f"  Nov 7 2022: ${nov7_val:,.2f}")

    # ── Trades summary ──
    print(f"\n--- Trades Summary ---")
    if trades_df.empty:
        print("  No trades recorded!")
    else:
        print(f"  Total trades: {len(trades_df)}")
        sell_trades = trades_df[trades_df["Trade Amount"] < 0]
        print(f"  Sell trades:  {len(sell_trades)}")
        print(f"  Total Realized ST P&L: ${trades_df['Realized ST P&L'].sum():>14,.2f}")
        print(f"  Total Realized LT P&L: ${trades_df['Realized LT P&L'].sum():>14,.2f}")
        total_realized = trades_df['Realized ST P&L'].sum() + trades_df['Realized LT P&L'].sum()
        print(f"  Total Realized P&L:    ${total_realized:>14,.2f}")

        # Per-year breakdown
        if "Year" not in trades_df.columns:
            trades_df["Year"] = trades_df["Date"].dt.year
        yearly = trades_df.groupby("Year")[["Realized ST P&L", "Realized LT P&L", "Trade Amount"]].agg({
            "Realized ST P&L": "sum",
            "Realized LT P&L": "sum",
            "Trade Amount": lambda x: x[x < 0].sum(),  # total sold
        }).rename(columns={"Trade Amount": "Total Sold"})
        yearly["Total P&L"] = yearly["Realized ST P&L"] + yearly["Realized LT P&L"]
        print(f"\n--- Realized P&L by Year (from trades_df) ---")
        for year, row in yearly.iterrows():
            print(f"  {year}: ST ${row['Realized ST P&L']:>12,.2f}  "
                  f"LT ${row['Realized LT P&L']:>12,.2f}  "
                  f"Total ${row['Total P&L']:>12,.2f}  "
                  f"Sold ${row['Total Sold']:>12,.2f}")

    # ── pl_by_year (as computed by shadow backtest) ──
    print(f"\n--- pl_by_year (shadow backtest output) ---")
    if pl_by_year.empty:
        print("  EMPTY!")
    else:
        print(pl_by_year.to_string())

    # ── Tax computation (exact same code as results/__init__.py) ──
    print(f"\n--- Tax Computation ---")
    if pl_by_year.empty:
        print("  No P&L → no tax")
        return None

    fed_tax = calculate_tax_series_with_carryforward(
        pl_by_year, OTHER_INCOME, FILING_STATUS,
        method=TAX_METHOD, use_standard_deduction=True,
    )
    state_tax = calculate_state_tax_series_with_carryforward(
        pl_by_year, OTHER_INCOME, STATE_CODE, FILING_STATUS,
        use_standard_deduction=True,
    )
    annual_total = fed_tax + state_tax

    tax_payment_series = pd.Series(0.0, index=port_series.index)
    print(f"\n  Year-by-Year Tax:")
    total_tax_all = 0.0
    for year, amount in annual_total.items():
        f = fed_tax.get(year, 0.0)
        s = state_tax.get(year, 0.0)
        if amount > 0:
            pay_date = pd.Timestamp(year + 1, 4, 15)
            idx = port_series.index.searchsorted(pay_date)
            if idx < len(port_series.index):
                actual_date = port_series.index[idx]
                tax_payment_series[actual_date] += amount
        realized = pl_by_year.loc[year, "Realized P&L"] if year in pl_by_year.index else 0
        eff_rate = (amount / realized * 100) if realized > 0 else 0
        print(f"  {year}: P&L ${realized:>12,.2f} → Fed ${f:>10,.2f} State ${s:>8,.2f} "
              f"Total ${amount:>10,.2f} (eff rate: {eff_rate:.1f}%)")
        total_tax_all += amount

    print(f"  {'─'*70}")
    print(f"  TOTAL TAX PAYMENTS: ${total_tax_all:,.2f}")

    # ── Margin simulation ──
    print(f"\n--- Margin Simulation ---")
    loan_series, equity_series, eq_pct, usage_series, eff_rate_series = simulate_margin(
        port_series,
        starting_loan=0.0,
        rate_annual=RATE_CONFIG,
        draw_monthly=0.0,
        maint_pct=wmaint,
        tax_series=tax_payment_series,
    )

    # Tax-adjusted usage (for pay_tax_margin: tax_adj_port = port)
    tax_adj_usage = pd.Series(0.0, index=port_series.index)
    if wmaint < 1.0:
        max_loan = port_series * (1 - wmaint)
        valid = max_loan > 0
        tax_adj_usage[valid] = loan_series[valid] / max_loan[valid]

    # Nov 7 snapshot
    if target in loan_series.index:
        l = loan_series.loc[target]
        e = equity_series.loc[target]
        u = tax_adj_usage.loc[target]
    else:
        l = loan_series.asof(target)
        e = equity_series.asof(target)
        u = tax_adj_usage.asof(target)
    p = nov7_val

    print(f"\n  Snapshot on Nov 7, 2022:")
    print(f"    Portfolio:    ${p:>12,.2f}")
    print(f"    Loan:         ${l:>12,.2f}")
    print(f"    Net Equity:   ${e:>12,.2f}")
    print(f"    Reg-T Usage:  {u:>12.2%}")
    print(f"    Eq + Loan:    ${e + l:>12,.2f} (delta: ${abs(e + l - p):.2f})")

    # Loan decomposition
    tax_paid_before_nov7 = tax_payment_series[tax_payment_series.index <= target].sum()
    print(f"\n    Tax paid by Nov 7: ${tax_paid_before_nov7:,.2f}")
    print(f"    Interest accrued:  ${l - tax_paid_before_nov7:,.2f}")
    print(f"    Loan breakdown:    {tax_paid_before_nov7/l*100:.1f}% tax / {(l-tax_paid_before_nov7)/l*100:.1f}% interest" if l > 0 else "")

    return {
        "name": name,
        "port_nov7": p,
        "loan_nov7": l,
        "equity_nov7": e,
        "usage_nov7": u,
        "wmaint": wmaint,
        "total_tax": total_tax_all,
        "tax_by_nov7": tax_paid_before_nov7,
        "pl_by_year": pl_by_year,
        "trades_df": trades_df,
    }


def main():
    print("=" * 80)
    print("  ACTUAL BACKTEST: NDXMEGASPLIT vs NDXMEGASPLIT Nigged")
    print("  Running real shadow backtests with real price data")
    print("=" * 80)

    r_base = run_and_trace("NDXMEGASPLIT (Base)", BASE_ALLOC, BASE_MAINT, BASE_REBAL)
    r_nigged = run_and_trace("NDXMEGASPLIT Nigged", NIGGED_ALLOC, NIGGED_MAINT, NIGGED_REBAL)

    if r_base and r_nigged:
        print(f"\n\n{'#'*80}")
        print(f"  COMPARISON: BASE vs NIGGED on Nov 7, 2022")
        print(f"{'#'*80}")

        print(f"\n  {'Metric':<25} {'Base':>15} {'Nigged':>15} {'Ratio':>10}")
        print(f"  {'─'*65}")
        for key, label in [
            ("port_nov7", "Portfolio"),
            ("loan_nov7", "Loan"),
            ("equity_nov7", "Net Equity"),
            ("total_tax", "Total Tax (all yrs)"),
            ("tax_by_nov7", "Tax by Nov 7"),
        ]:
            b = r_base[key]
            n = r_nigged[key]
            ratio = n / b if b != 0 else float('inf')
            print(f"  {label:<25} ${b:>13,.2f} ${n:>13,.2f} {ratio:>9.2f}x")

        print(f"  {'Usage (Reg-T)':<25} {r_base['usage_nov7']:>14.2%} {r_nigged['usage_nov7']:>14.2%} {r_nigged['usage_nov7']/r_base['usage_nov7']:>9.2f}x")
        print(f"  {'wmaint':<25} {r_base['wmaint']:>14.4f} {r_nigged['wmaint']:>14.4f}")

        # Detailed P&L comparison
        print(f"\n  {'─'*80}")
        print(f"  Year-by-Year Realized P&L Comparison:")
        print(f"  {'Year':<6} {'Base P&L':>14} {'Nigged P&L':>14} {'Ratio':>8} {'Base Sold':>14} {'Nigged Sold':>14}")
        print(f"  {'─'*80}")

        pl_b = r_base["pl_by_year"]
        pl_n = r_nigged["pl_by_year"]
        all_years = sorted(set(pl_b.index) | set(pl_n.index))

        # Get trade volumes
        trades_b = r_base["trades_df"]
        trades_n = r_nigged["trades_df"]
        if "Year" not in trades_b.columns and not trades_b.empty:
            trades_b["Year"] = trades_b["Date"].dt.year
        if "Year" not in trades_n.columns and not trades_n.empty:
            trades_n["Year"] = trades_n["Date"].dt.year

        for y in all_years:
            bp = pl_b.loc[y, "Realized P&L"] if y in pl_b.index else 0
            np_ = pl_n.loc[y, "Realized P&L"] if y in pl_n.index else 0
            ratio = np_ / bp if bp != 0 else float('inf')

            bs = trades_b[trades_b["Year"] == y]["Trade Amount"].clip(upper=0).sum() if not trades_b.empty else 0
            ns = trades_n[trades_n["Year"] == y]["Trade Amount"].clip(upper=0).sum() if not trades_n.empty else 0

            print(f"  {y:<6} ${bp:>12,.2f} ${np_:>12,.2f} {ratio:>7.2f}x ${bs:>12,.2f} ${ns:>12,.2f}")

        # Per-ticker P&L for Nigged (which ticker drives the most gains?)
        print(f"\n  {'─'*80}")
        print(f"  Nigged: Realized P&L by TICKER (cumulative):")
        if not trades_n.empty:
            ticker_pl = trades_n.groupby("Ticker")[["Realized ST P&L", "Realized LT P&L"]].sum()
            ticker_pl["Total"] = ticker_pl["Realized ST P&L"] + ticker_pl["Realized LT P&L"]
            ticker_pl = ticker_pl.sort_values("Total", ascending=False)
            for t, row in ticker_pl.iterrows():
                print(f"    {t:<25} ST: ${row['Realized ST P&L']:>12,.2f}  "
                      f"LT: ${row['Realized LT P&L']:>12,.2f}  "
                      f"Total: ${row['Total']:>12,.2f}")

        print(f"\n  Base: Realized P&L by TICKER (cumulative):")
        if not trades_b.empty:
            ticker_pl_b = trades_b.groupby("Ticker")[["Realized ST P&L", "Realized LT P&L"]].sum()
            ticker_pl_b["Total"] = ticker_pl_b["Realized ST P&L"] + ticker_pl_b["Realized LT P&L"]
            ticker_pl_b = ticker_pl_b.sort_values("Total", ascending=False)
            for t, row in ticker_pl_b.iterrows():
                print(f"    {t:<25} ST: ${row['Realized ST P&L']:>12,.2f}  "
                      f"LT: ${row['Realized LT P&L']:>12,.2f}  "
                      f"Total: ${row['Total']:>12,.2f}")


if __name__ == "__main__":
    main()

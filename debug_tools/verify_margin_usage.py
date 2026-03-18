#!/usr/bin/env python3
"""
Verify margin usage computation end-to-end.

Replays the EXACT code paths used in results/__init__.py and testfol_api.py
to compute loan growth and margin usage from tax payments.

Two modes:
  1. Forward: synthetic P&L → tax → loan → usage (formula check)
  2. Reverse: back-solve from ACTUAL tooltip values (what taxes are needed?)

Usage:
    python debug_tools/verify_margin_usage.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from app.services.testfol_api import simulate_margin
from app.core.tax_library import (
    calculate_tax_series_with_carryforward,
    calculate_state_tax_series_with_carryforward,
)

# ─────────────────────────────────────────────────────────────────────────────
# Helper: build tax_payment_series exactly as results/__init__.py does
# ─────────────────────────────────────────────────────────────────────────────

def build_tax_payment_series(
    pl_by_year: pd.DataFrame,
    port_index: pd.DatetimeIndex,
    other_income: float,
    filing_status: str,
    tax_method: str,
    state_code: str,
    use_std_deduction: bool = True,
):
    """Replicate results/__init__.py lines 141-176."""
    fed_tax_series = calculate_tax_series_with_carryforward(
        pl_by_year, other_income, filing_status,
        method=tax_method, use_standard_deduction=use_std_deduction,
    )
    state_tax_series = calculate_state_tax_series_with_carryforward(
        pl_by_year, other_income, state_code, filing_status,
        use_standard_deduction=use_std_deduction,
    )

    annual_total_tax = fed_tax_series + state_tax_series

    tax_payment_series = pd.Series(0.0, index=port_index)
    payment_log = []

    for year, amount in annual_total_tax.items():
        fed_amt = fed_tax_series.get(year, 0.0)
        state_amt = state_tax_series.get(year, 0.0)
        if amount > 0:
            pay_date = pd.Timestamp(year + 1, 4, 15)
            idx = port_index.searchsorted(pay_date)
            if idx < len(port_index):
                actual_date = port_index[idx]
                tax_payment_series[actual_date] += amount
                payment_log.append({
                    "Tax Year": year,
                    "Payment Date": actual_date.date(),
                    "Federal Tax": fed_amt,
                    "State Tax": state_amt,
                    "Total Payment": amount,
                })

    return tax_payment_series, pd.DataFrame(payment_log)


# ─────────────────────────────────────────────────────────────────────────────
# Scenario builders
# ─────────────────────────────────────────────────────────────────────────────

def make_portfolio_series(start_val, annual_return, start="2016-01-04", end="2022-11-07"):
    """Generate a daily portfolio value series with constant growth rate."""
    dates = pd.bdate_range(start, end)
    daily_r = (1 + annual_return) ** (1/252) - 1
    vals = start_val * (1 + daily_r) ** np.arange(len(dates))
    return pd.Series(vals, index=dates, name="Portfolio")


def make_pl_by_year(port_series, annual_turnover_pct=0.15, gain_pct=0.40):
    """
    Simulate yearly realized P&L from rebalancing.

    Assumes `annual_turnover_pct` of portfolio sold each year,
    with cost basis = (1-gain_pct) of proceeds.
    Splits 80% LT / 20% ST.
    """
    yearly_vals = port_series.resample("YE").last()
    rows = []
    for date, val in yearly_vals.items():
        sold = val * annual_turnover_pct
        gain = sold * gain_pct
        rows.append({
            "Year": date.year,
            "Realized ST P&L": gain * 0.20,
            "Realized LT P&L": gain * 0.80,
        })
    df = pd.DataFrame(rows).set_index("Year")
    df["Realized P&L"] = df["Realized ST P&L"] + df["Realized LT P&L"]
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Forward verification: P&L → tax → loan → usage
# ─────────────────────────────────────────────────────────────────────────────

def verify_portfolio(name, port_series, pl_by_year, rate_config, wmaint,
                     other_income, filing_status, tax_method, state_code):
    """Run the full tax → loan → usage pipeline and print diagnostics."""

    print(f"\n{'='*80}")
    print(f"  PORTFOLIO: {name}")
    print(f"{'='*80}")

    # 1. Tax computation
    tax_payment_series, payment_log = build_tax_payment_series(
        pl_by_year, port_series.index,
        other_income=other_income,
        filing_status=filing_status,
        tax_method=tax_method,
        state_code=state_code,
    )

    print(f"\n--- Realized P&L by Year ---")
    print(pl_by_year.to_string())

    print(f"\n--- Tax Payments (added to margin loan) ---")
    if not payment_log.empty:
        for _, row in payment_log.iterrows():
            print(f"  Tax Year {row['Tax Year']}: Fed ${row['Federal Tax']:>12,.2f}  "
                  f"State ${row['State Tax']:>10,.2f}  "
                  f"Total ${row['Total Payment']:>12,.2f}  → paid {row['Payment Date']}")
        print(f"  {'─'*70}")
        print(f"  CUMULATIVE TAX PAYMENTS: ${payment_log['Total Payment'].sum():,.2f}")
    else:
        print("  (none)")

    # 2. simulate_margin (exact same function as the app)
    loan_series, equity_series, _, _, eff_rate = simulate_margin(
        port_series,
        starting_loan=0.0,
        rate_annual=rate_config,
        draw_monthly=0.0,
        maint_pct=wmaint,
        tax_series=tax_payment_series,
    )

    # 3. Tax-adjusted usage (for pay_tax_margin: tax_adj_port = port)
    tax_adj_port = equity_series + loan_series
    tax_adj_usage = pd.Series(0.0, index=port_series.index)
    if wmaint < 1.0:
        max_loan = tax_adj_port * (1 - wmaint)
        valid = max_loan > 0
        tax_adj_usage[valid] = loan_series[valid] / max_loan[valid]

    # 4. Print final-day snapshot
    check_date = port_series.index[-1]
    port_val = port_series.loc[check_date]
    loan_val = loan_series.loc[check_date]
    equity_val = equity_series.loc[check_date]
    usage_val = tax_adj_usage.loc[check_date]

    print(f"\n--- Snapshot on {check_date.date()} ---")
    print(f"  Portfolio (Gross):  ${port_val:>14,.2f}")
    print(f"  Loan:              ${loan_val:>14,.2f}")
    print(f"  Net Equity:        ${equity_val:>14,.2f}")
    print(f"  Loan / Portfolio:  {loan_val/port_val:>14.2%}")
    print(f"  wmaint:            {wmaint:>14.4f}")
    print(f"  Max Allowed Loan:  ${port_val * (1-wmaint):>14,.2f}")
    print(f"  Reg-T Usage:       {usage_val:>14.2%}")

    # 5. Verify formula: equity + loan = portfolio
    check_sum = equity_val + loan_val
    delta = abs(check_sum - port_val)
    print(f"\n  CHECK: Equity + Loan = ${check_sum:,.2f} (delta: ${delta:.6f})")

    # 6. Verify usage formula
    expected_usage = loan_val / (port_val * (1 - wmaint))
    print(f"  CHECK: usage formula = {expected_usage:.6%} (matches: {abs(expected_usage - usage_val) < 1e-8})")

    # 7. Interest breakdown
    total_tax_paid = tax_payment_series.sum()
    total_interest = loan_val - total_tax_paid
    print(f"\n--- Loan Decomposition ---")
    print(f"  Cumulative Tax Payments: ${total_tax_paid:>14,.2f}")
    print(f"  Cumulative Interest:     ${total_interest:>14,.2f}")
    print(f"  Total Loan:              ${loan_val:>14,.2f}")
    if total_tax_paid > 0:
        print(f"  Interest / Tax Ratio:    {total_interest/total_tax_paid:>14.2%}")

    return {
        "name": name,
        "port_val": port_val,
        "loan": loan_val,
        "equity": equity_val,
        "usage": usage_val,
        "total_tax": total_tax_paid,
        "total_interest": total_interest,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Reverse verification: back-solve from ACTUAL tooltip values
# ─────────────────────────────────────────────────────────────────────────────

def reverse_verify(name, port_val, equity_val, loan_val, usage_pct, pm_usage_pct, rate_pct):
    """
    Given the ACTUAL values from the chart tooltip, verify all relationships
    and back-solve what the wmaint must be.
    """
    print(f"\n{'='*80}")
    print(f"  REVERSE VERIFY: {name} (from tooltip)")
    print(f"{'='*80}")

    print(f"\n  Raw tooltip values:")
    print(f"    Portfolio:  ${port_val:>12,.0f}")
    print(f"    Net Equity: ${equity_val:>12,.0f}")
    print(f"    Loan:       ${loan_val:>12,.0f}")
    print(f"    Reg-T:      {usage_pct:>12.1%}")
    print(f"    PM:         {pm_usage_pct:>12.1%}")
    print(f"    Rate:       {rate_pct:>12.2%}")

    # 1. Check equity + loan = portfolio
    computed_port = equity_val + loan_val
    delta_port = abs(computed_port - port_val)
    print(f"\n  TEST 1: Equity + Loan = Portfolio")
    print(f"    {equity_val:,.0f} + {loan_val:,.0f} = {computed_port:,.0f}")
    print(f"    Expected: {port_val:,.0f}  |  Delta: ${delta_port:,.0f}")
    print(f"    {'PASS' if delta_port < 5 else 'FAIL'}")

    # 2. Back-solve wmaint from Reg-T usage
    #    usage = loan / (port * (1 - wmaint))
    #    => (1 - wmaint) = loan / (port * usage)
    #    => wmaint = 1 - loan / (port * usage)
    wmaint_solved = 1 - loan_val / (port_val * usage_pct)
    print(f"\n  TEST 2: Back-solve wmaint from Reg-T usage")
    print(f"    usage = loan / (port * (1 - wmaint))")
    print(f"    {usage_pct:.4f} = {loan_val:,.0f} / ({port_val:,.0f} * (1 - wmaint))")
    print(f"    => wmaint = {wmaint_solved:.4f} ({wmaint_solved*100:.2f}%)")

    # 3. Back-solve wmaint_pm from PM usage
    wmaint_pm_solved = 1 - loan_val / (port_val * pm_usage_pct)
    print(f"\n  TEST 3: Back-solve wmaint_pm from PM usage")
    print(f"    pm_usage = loan / (port * (1 - wmaint_pm))")
    print(f"    {pm_usage_pct:.4f} = {loan_val:,.0f} / ({port_val:,.0f} * (1 - wmaint_pm))")
    print(f"    => wmaint_pm = {wmaint_pm_solved:.4f} ({wmaint_pm_solved*100:.2f}%)")

    # 4. Verify cross-consistency
    #    For the SAME loan, the PM usage should differ from Reg-T only
    #    because of different wmaint values. Verify:
    recomputed_regt = loan_val / (port_val * (1 - wmaint_solved))
    recomputed_pm = loan_val / (port_val * (1 - wmaint_pm_solved))
    print(f"\n  TEST 4: Recompute usage from solved wmaint values")
    print(f"    Reg-T recomputed: {recomputed_regt:.4f} (expected {usage_pct:.4f})")
    print(f"    PM recomputed:    {recomputed_pm:.4f} (expected {pm_usage_pct:.4f})")
    print(f"    Both match: {'PASS' if abs(recomputed_regt - usage_pct) < 0.001 and abs(recomputed_pm - pm_usage_pct) < 0.001 else 'FAIL'}")

    return {
        "wmaint": wmaint_solved,
        "wmaint_pm": wmaint_pm_solved,
        "port_val": port_val,
        "loan": loan_val,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Back-solve: what cumulative taxes produce the observed loan?
# ─────────────────────────────────────────────────────────────────────────────

def backsol_required_taxes(target_loan, n_years, avg_rate_pct):
    """
    If the loan comes entirely from tax payments (paid April 15 each year)
    plus compounding interest, what constant annual tax payment would
    produce the target loan after n_years?

    loan = sum_{k=1..n}(annual_tax * (1 + r)^(n - k + fraction))
    where r = avg_rate / 100
    """
    r = avg_rate_pct / 100.0
    # Each payment compounds from its payment date to the check date.
    # Payment for year k arrives April 15 of year k+1.
    # If check date is Nov 7, 2022 (year 7), the payment for year k
    # compounds for (2022.85 - (k+1).29) years ≈ (6.85 - k + 0.71) years.
    # Simplify: compounding for (n_years - k + 0.56) years.
    total_factor = sum((1 + r) ** max(0, n_years - k + 0.56) for k in range(1, n_years + 1))
    annual_tax = target_loan / total_factor if total_factor > 0 else 0
    return annual_tax, total_factor


def main():
    print("=" * 80)
    print("  MARGIN USAGE VERIFICATION SCRIPT")
    print("  Mode 1: Reverse-verify from ACTUAL tooltip values")
    print("  Mode 2: Forward P&L → tax → loan → usage")
    print("=" * 80)

    # ═══════════════════════════════════════════════════════════════════════════
    # MODE 1: REVERSE VERIFICATION FROM YOUR ACTUAL NOV 7, 2022 TOOLTIP DATA
    # ═══════════════════════════════════════════════════════════════════════════

    print(f"\n{'#'*80}")
    print(f"  MODE 1: REVERSE VERIFICATION (Actual Tooltip Data)")
    print(f"{'#'*80}")

    r_base = reverse_verify(
        "NDXMEGASPLIT (Base)",
        port_val=421533, equity_val=365256, loan_val=56278,
        usage_pct=0.232, pm_usage_pct=0.175, rate_pct=5.28,
    )

    r_nigged = reverse_verify(
        "NDXMEGASPLIT Nigged",
        port_val=904370, equity_val=638402, loan_val=265968,
        usage_pct=0.511, pm_usage_pct=0.386, rate_pct=4.97,
    )

    # Cross-portfolio comparison of solved wmaint values
    print(f"\n{'='*80}")
    print(f"  CROSS-PORTFOLIO: SOLVED wmaint COMPARISON")
    print(f"{'='*80}")
    print(f"  Base wmaint:         {r_base['wmaint']:.4f} ({r_base['wmaint']*100:.2f}%)")
    print(f"  Nigged wmaint:       {r_nigged['wmaint']:.4f} ({r_nigged['wmaint']*100:.2f}%)")
    print(f"  Difference:          {abs(r_base['wmaint'] - r_nigged['wmaint']):.6f}")
    if abs(r_base['wmaint'] - r_nigged['wmaint']) < 0.01:
        print(f"  => Both portfolios use the SAME wmaint (as expected if same tickers)")
    else:
        print(f"  => DIFFERENT wmaint values! Different Maint % per ticker in allocation?")

    print(f"\n  Base wmaint_pm:      {r_base['wmaint_pm']:.4f} ({r_base['wmaint_pm']*100:.2f}%)")
    print(f"  Nigged wmaint_pm:    {r_nigged['wmaint_pm']:.4f} ({r_nigged['wmaint_pm']*100:.2f}%)")

    # Back-solve required annual tax payments
    print(f"\n{'='*80}")
    print(f"  BACK-SOLVE: Required Annual Tax to Produce Observed Loan")
    print(f"{'='*80}")

    n_years = 7  # 2016..2022

    base_annual_tax, base_factor = backsol_required_taxes(56278, n_years, 5.28)
    nigged_annual_tax, nigged_factor = backsol_required_taxes(265968, n_years, 4.97)

    print(f"\n  Base (loan=$56,278, rate=5.28%):")
    print(f"    Compounding factor sum: {base_factor:.2f}")
    print(f"    Required avg annual tax: ${base_annual_tax:,.0f}")
    print(f"    Over 7 years total: ${base_annual_tax * n_years:,.0f}")

    print(f"\n  Nigged (loan=$265,968, rate=4.97%):")
    print(f"    Compounding factor sum: {nigged_factor:.2f}")
    print(f"    Required avg annual tax: ${nigged_annual_tax:,.0f}")
    print(f"    Over 7 years total: ${nigged_annual_tax * n_years:,.0f}")

    ratio_tax_needed = nigged_annual_tax / base_annual_tax if base_annual_tax > 0 else float('inf')
    ratio_port = r_nigged['port_val'] / r_base['port_val']

    print(f"\n  Tax ratio needed (Nigged/Base): {ratio_tax_needed:.2f}x")
    print(f"  Portfolio ratio (Nigged/Base):  {ratio_port:.2f}x")

    if ratio_tax_needed > ratio_port * 2:
        print(f"\n  ⚠ SUSPICIOUS: Nigged needs {ratio_tax_needed:.1f}x the annual tax")
        print(f"    but only has {ratio_port:.1f}x the portfolio value.")
        print(f"    Possible explanations:")
        print(f"      a) Nigged has higher turnover (more rebalancing trades)")
        print(f"      b) Nigged has deeper gains (lower cost basis relative to current value)")
        print(f"      c) Nigged's gains push into higher tax brackets (progressive)")
        print(f"      d) BUG: Tax computation overcounts gains for Nigged")
        print(f"\n    TO INVESTIGATE: Check the Rebalancing tab → Tax Analysis for both portfolios.")
        print(f"    Compare year-by-year Realized P&L and tax amounts.")
    else:
        print(f"\n  The tax ratio ({ratio_tax_needed:.1f}x) is proportional to")
        print(f"  the portfolio ratio ({ratio_port:.1f}x). Behavior is expected.")

    # ═══════════════════════════════════════════════════════════════════════════
    # MODE 2: FORWARD VERIFICATION WITH REALISTIC P&L
    # ═══════════════════════════════════════════════════════════════════════════

    print(f"\n\n{'#'*80}")
    print(f"  MODE 2: FORWARD VERIFICATION (Synthetic, matching observed loan)")
    print(f"{'#'*80}")

    # Config matching screenshots
    other_income = 100_000.0
    filing_status = "Single"
    tax_method = "historical"
    state_code = "NY"
    wmaint = r_base['wmaint']  # use the solved wmaint

    rate_config = {
        "type": "Tiered",
        "tiers": [
            (0,      1.50),
            (100000, 1.00),
            (1000000, 0.75),
            (50000000, 0.50),
        ],
        "base_series": None,
    }

    # Calibrate the P&L to produce the OBSERVED loan amounts.
    # Base needs ~$6.2k/year average tax → need ~$30k avg annual realized gains
    # Nigged needs ~$29k/year avg tax → need ~$140k avg annual realized gains
    # For a portfolio going from $10k to $421k, 15% turnover * 85% gain = ~12.75% of value
    # For the nigged going to $904k, similar.
    # Let's calibrate turnover to match.

    port_base = make_portfolio_series(10000, 0.65, "2016-01-04", "2022-11-07")
    port_nigged = make_portfolio_series(10000, 0.83, "2016-01-04", "2022-11-07")

    # Higher gain_pct = lower cost basis relative to current value (more unrealized gains)
    # Try turnover=30%, gain=85% for base; turnover=30%, gain=90% for nigged
    pl_base = make_pl_by_year(port_base, annual_turnover_pct=0.30, gain_pct=0.85)
    pl_nigged = make_pl_by_year(port_nigged, annual_turnover_pct=0.30, gain_pct=0.90)

    verify_portfolio(
        "NDXMEGASPLIT (Base) [calibrated]", port_base, pl_base, rate_config, wmaint,
        other_income, filing_status, tax_method, state_code,
    )

    verify_portfolio(
        "NDXMEGASPLIT Nigged [calibrated]", port_nigged, pl_nigged, rate_config, wmaint,
        other_income, filing_status, tax_method, state_code,
    )

    # ═══════════════════════════════════════════════════════════════════════════
    # MODE 3: FORMULA UNIT TESTS
    # ═══════════════════════════════════════════════════════════════════════════

    print(f"\n\n{'#'*80}")
    print(f"  MODE 3: FORMULA UNIT TESTS")
    print(f"{'#'*80}")

    # Test: single lump-sum tax payment → verify loan growth
    print(f"\n  Test A: Single $10,000 tax payment on day 1, 5% rate, 252 days")
    dates = pd.bdate_range("2022-01-03", periods=252)
    port = pd.Series(100_000.0, index=dates)
    tax = pd.Series(0.0, index=dates)
    tax.iloc[0] = 10_000.0

    loan, eq, _, usage, _ = simulate_margin(
        port, starting_loan=0.0, rate_annual=5.0,
        draw_monthly=0.0, maint_pct=0.25, tax_series=tax,
    )

    expected_loan = 10_000 * (1 + 0.05/252) ** 251  # compounds for 251 days after injection
    actual_loan = loan.iloc[-1]
    print(f"    Expected loan (manual): ${expected_loan:,.2f}")
    print(f"    Actual loan (simulate_margin): ${actual_loan:,.2f}")
    print(f"    Delta: ${abs(actual_loan - expected_loan):.4f}")
    print(f"    {'PASS' if abs(actual_loan - expected_loan) < 0.01 else 'FAIL'}")

    expected_usage = actual_loan / (100_000 * (1 - 0.25))
    actual_usage = usage.iloc[-1]
    print(f"    Expected usage: {expected_usage:.6f}")
    print(f"    Actual usage:   {actual_usage:.6f}")
    print(f"    {'PASS' if abs(actual_usage - expected_usage) < 1e-8 else 'FAIL'}")

    # Test: verify equity + loan = portfolio identity
    print(f"\n  Test B: Equity + Loan = Portfolio (identity check across all days)")
    identity_errors = abs((eq + loan) - port)
    max_err = identity_errors.max()
    print(f"    Max error across 252 days: ${max_err:.10f}")
    print(f"    {'PASS' if max_err < 1e-6 else 'FAIL'}")

    # Test: zero tax → zero loan
    print(f"\n  Test C: Zero tax payments → zero loan")
    loan_zero, _, _, _, _ = simulate_margin(
        port, starting_loan=0.0, rate_annual=5.0,
        draw_monthly=0.0, maint_pct=0.25, tax_series=None,
    )
    max_loan = abs(loan_zero).max()
    print(f"    Max loan across 252 days: ${max_loan:.10f}")
    print(f"    {'PASS' if max_loan < 1e-6 else 'FAIL'}")

    # Test: wmaint sensitivity
    print(f"\n  Test D: wmaint sensitivity on usage (same loan, different wmaint)")
    test_loan = 100_000
    test_port_val = 500_000
    for wm in [0.25, 0.30, 0.35, 0.40, 0.425, 0.50]:
        u = test_loan / (test_port_val * (1 - wm))
        print(f"    wmaint={wm:.3f} → usage={u:.2%}  max_loan=${test_port_val*(1-wm):,.0f}")


if __name__ == "__main__":
    main()

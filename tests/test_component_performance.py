"""Tests for component performance position tracking logic in rebalancing.py."""

import numpy as np
import pandas as pd
import pytest

from app.ui.charts.rebalancing import (
    _build_portfolio_allocation_data,
    _get_cached_portfolio_allocation_data,
    render_portfolio_allocation,
)


def _compute_positions(raw_prices, allocation, start_val, rebal_dates=None):
    """
    Extracted position tracking logic from render_portfolio_allocation.
    Returns (positions DataFrame, row_totals Series).
    """
    # Parse allocation
    weights = {}
    leverage = {}
    expense_ratio = {}
    for full_tk, weight in allocation.items():
        base = full_tk.split("?")[0]
        weights[base] = weights.get(base, 0) + weight
        if "?" in full_tk:
            query = full_tk.split("?", 1)[1]
            for pair in query.split("&"):
                if "=" in pair:
                    k, v = pair.split("=", 1)
                    try:
                        if k.upper() == "L":
                            leverage[base] = float(v)
                        elif k.upper() in ("E", "D"):
                            expense_ratio[base] = float(v)
                    except ValueError:
                        pass

    available = [t for t in weights if t in raw_prices.columns]
    prices_clean = raw_prices[available].dropna(how="all").ffill().dropna(how="any")

    # Apply leverage and ER
    daily_returns = prices_clean.pct_change()
    modified_returns = daily_returns.copy()
    for t in available:
        lev = leverage.get(t, 1.0)
        er = expense_ratio.get(t, 0.0)
        daily_er = (er / 100.0) / 252.0 if er > 0 else 0.0
        if lev != 1.0 or daily_er > 0:
            modified_returns[t] = daily_returns[t] * lev - daily_er

    # Reconstruct modified prices
    prices = (1 + modified_returns).cumprod() * prices_clean.iloc[0]
    prices.iloc[0] = prices_clean.iloc[0]

    total_weight = sum(weights[t] for t in available)

    # Segment boundaries
    seg_starts = [prices.index[0]]
    if rebal_dates:
        for rd in sorted(rebal_dates):
            idx_loc = prices.index.searchsorted(rd)
            if 0 < idx_loc < len(prices.index):
                snapped = prices.index[idx_loc]
                if snapped > seg_starts[-1]:
                    seg_starts.append(snapped)

    # Position tracking
    all_positions = []
    for i, seg_start in enumerate(seg_starts):
        if i + 1 < len(seg_starts):
            seg_prices = prices.loc[seg_start:seg_starts[i + 1]].iloc[:-1]
        else:
            seg_prices = prices.loc[seg_start:]

        if seg_prices.empty:
            continue

        if i == 0:
            total_val = start_val
        else:
            prev_end = all_positions[-1].iloc[-1] if all_positions else None
            total_val = prev_end.sum() if prev_end is not None else start_val

        start_prices = seg_prices.iloc[0]
        seg_pos = pd.DataFrame(index=seg_prices.index, columns=available, dtype=float)
        for t in available:
            if start_prices[t] > 0:
                alloc_val = total_val * (weights[t] / total_weight)
                seg_pos[t] = alloc_val * (seg_prices[t] / start_prices[t])
            else:
                seg_pos[t] = 0.0

        all_positions.append(seg_pos)

    positions = pd.concat(all_positions)
    positions = positions[~positions.index.duplicated(keep="first")]
    row_totals = positions.sum(axis=1)
    return positions, row_totals


def _make_prices(daily_return=0.001, n_days=252, start_price=100.0):
    """Create a simple price series with constant daily return."""
    dates = pd.bdate_range("2020-01-02", periods=n_days)
    prices = start_price * (1 + daily_return) ** np.arange(n_days)
    return dates, prices


class TestNoLeverage:
    """Baseline: no modifiers, single ticker."""

    def test_single_ticker_no_modifiers(self):
        dates, prices_arr = _make_prices(daily_return=0.001, n_days=100)
        raw = pd.DataFrame({"SPY": prices_arr}, index=dates)
        alloc = {"SPY": 100.0}

        positions, totals = _compute_positions(raw, alloc, start_val=10000)

        # Total should track: 10000 * (1.001)^99 = 10000 * 1.1041 ≈ 11041
        expected_final = 10000 * (1.001) ** 99
        assert abs(totals.iloc[-1] - expected_final) / expected_final < 0.001

    def test_two_tickers_equal_weight(self):
        dates, _ = _make_prices(n_days=100)
        raw = pd.DataFrame({
            "SPY": 100 * (1.001) ** np.arange(100),
            "TLT": 50 * (1.002) ** np.arange(100),
        }, index=dates)
        alloc = {"SPY": 50.0, "TLT": 50.0}

        positions, totals = _compute_positions(raw, alloc, start_val=10000)

        assert abs(totals.iloc[0] - 10000) < 1.0
        # SPY position: 5000 * (1.001)^99, TLT position: 5000 * (1.002)^99
        expected = 5000 * (1.001) ** 99 + 5000 * (1.002) ** 99
        assert abs(totals.iloc[-1] - expected) / expected < 0.001


class TestWithLeverage:
    """Leverage modifier applied correctly."""

    def test_2x_leverage_single_ticker(self):
        dates, prices_arr = _make_prices(daily_return=0.001, n_days=100)
        raw = pd.DataFrame({"SPY": prices_arr}, index=dates)
        alloc = {"SPY?L=2": 100.0}

        positions, totals = _compute_positions(raw, alloc, start_val=10000)

        # With 2x leverage, daily return = 0.002
        # Final = 10000 * (1.002)^99
        expected = 10000 * (1.002) ** 99
        assert abs(totals.iloc[-1] - expected) / expected < 0.001

    def test_3x_leverage(self):
        dates, prices_arr = _make_prices(daily_return=0.001, n_days=50)
        raw = pd.DataFrame({"QQQ": prices_arr}, index=dates)
        alloc = {"QQQ?L=3": 100.0}

        positions, totals = _compute_positions(raw, alloc, start_val=10000)

        expected = 10000 * (1.003) ** 49
        assert abs(totals.iloc[-1] - expected) / expected < 0.001

    def test_leverage_with_expense_ratio(self):
        dates, prices_arr = _make_prices(daily_return=0.001, n_days=100)
        raw = pd.DataFrame({"SPY": prices_arr}, index=dates)
        alloc = {"SPY?L=2&E=0.95": 100.0}

        positions, totals = _compute_positions(raw, alloc, start_val=10000)

        daily_er = 0.0095 / 252
        # Daily modified return = 0.001 * 2 - daily_er
        mod_ret = 0.002 - daily_er
        expected = 10000 * (1 + mod_ret) ** 99
        assert abs(totals.iloc[-1] - expected) / expected < 0.001


class TestWithRebalancing:
    """Position tracking across rebalance boundaries."""

    def test_rebalance_preserves_total_value(self):
        """At a rebalance, total value should be continuous (no jump)."""
        dates = pd.bdate_range("2020-01-02", periods=200)
        raw = pd.DataFrame({
            "A": 100 * (1.002) ** np.arange(200),  # strong
            "B": 100 * (1.0005) ** np.arange(200),  # weak
        }, index=dates)
        alloc = {"A": 50.0, "B": 50.0}
        rebal = [dates[100]]  # rebalance at midpoint

        positions, totals = _compute_positions(raw, alloc, start_val=10000, rebal_dates=rebal)

        # Check continuity at rebalance boundary
        pre_rebal = totals.iloc[99]
        post_rebal = totals.iloc[100]
        # Should be equal (rebalance preserves total value)
        assert abs(pre_rebal - post_rebal) / pre_rebal < 0.01

    def test_leveraged_rebalance(self):
        """Leverage + rebalancing: verify total tracks correctly."""
        dates = pd.bdate_range("2020-01-02", periods=200)
        raw = pd.DataFrame({
            "A": 100 * (1.001) ** np.arange(200),
            "B": 50 * (1.001) ** np.arange(200),
        }, index=dates)
        alloc = {"A?L=2": 60.0, "B": 40.0}
        rebal = [dates[100]]

        positions, totals = _compute_positions(raw, alloc, start_val=10000, rebal_dates=rebal)

        # Segment 1 (0-99): A at 2x leverage, B at 1x
        # A daily = 0.002, B daily = 0.001
        a_val_99 = 6000 * (1.002) ** 99
        b_val_99 = 4000 * (1.001) ** 99
        seg1_total = a_val_99 + b_val_99

        # Segment 2 (100-199): reallocate
        a_val_199 = seg1_total * 0.6 * (1.002) ** 99
        b_val_199 = seg1_total * 0.4 * (1.001) ** 99
        expected_final = a_val_199 + b_val_199

        actual_final = totals.iloc[-1]
        assert abs(actual_final - expected_final) / expected_final < 0.01, \
            f"Expected ~{expected_final:.0f}, got {actual_final:.0f}"


class TestEdgeCases:
    """Edge cases: NaN handling, single day segments, etc."""

    def test_no_nan_in_output(self):
        dates, prices_arr = _make_prices(n_days=50)
        raw = pd.DataFrame({"SPY": prices_arr}, index=dates)
        alloc = {"SPY?L=2": 100.0}

        positions, totals = _compute_positions(raw, alloc, start_val=10000)
        assert not totals.isna().any(), f"Found NaN in totals: {totals[totals.isna()]}"

    def test_expense_ratio_only(self):
        dates, prices_arr = _make_prices(daily_return=0.001, n_days=100)
        raw = pd.DataFrame({"GLD": prices_arr}, index=dates)
        alloc = {"GLD?E=0.40": 100.0}

        positions, totals = _compute_positions(raw, alloc, start_val=10000)

        daily_er = 0.004 / 252
        mod_ret = 0.001 - daily_er
        expected = 10000 * (1 + mod_ret) ** 99
        assert abs(totals.iloc[-1] - expected) / expected < 0.001

    def test_mixed_leveraged_and_plain(self):
        """Mix of leveraged and unleveraged tickers."""
        dates = pd.bdate_range("2020-01-02", periods=100)
        raw = pd.DataFrame({
            "SPY": 100 * (1.001) ** np.arange(100),
            "GLD": 50 * (1.0005) ** np.arange(100),
        }, index=dates)
        alloc = {"SPY?L=2&E=0.95": 70.0, "GLD": 30.0}

        positions, totals = _compute_positions(raw, alloc, start_val=10000)

        daily_er = 0.0095 / 252
        spy_mod_ret = 0.001 * 2 - daily_er
        spy_expected = 7000 * (1 + spy_mod_ret) ** 99
        gld_expected = 3000 * (1.0005) ** 99
        expected = spy_expected + gld_expected
        assert abs(totals.iloc[-1] - expected) / expected < 0.005, \
            f"Expected ~{expected:.0f}, got {totals.iloc[-1]:.0f}"


class TestRebalancingAllocationHelper:
    def test_helper_matches_expected_growth_without_rebalances(self):
        dates, prices_arr = _make_prices(daily_return=0.001, n_days=100)
        raw = pd.DataFrame({"SPY": prices_arr}, index=dates)

        result = _build_portfolio_allocation_data(
            component_prices=raw,
            allocation={"SPY": 100.0},
            composition_df=pd.DataFrame(),
            start_val=10000,
        )

        assert result is not None
        totals = result["row_totals"]
        expected_final = 10000 * (1.001) ** 99
        assert abs(totals.iloc[-1] - expected_final) / expected_final < 0.001

    def test_helper_resets_allocations_at_rebalance_snapshots(self):
        dates = pd.bdate_range("2020-01-02", periods=10)
        raw = pd.DataFrame(
            {
                "A": 100 * (1.10) ** np.arange(10),
                "B": 100.0,
            },
            index=dates,
        )
        rebal_date = dates[5]
        composition = pd.DataFrame(
            [
                {"Date": rebal_date, "Ticker": "A", "Value": 5000.0},
                {"Date": rebal_date, "Ticker": "B", "Value": 5000.0},
                # Final composition snapshot should not become a rebalance marker.
                {"Date": dates[-1], "Ticker": "A", "Value": 9000.0},
                {"Date": dates[-1], "Ticker": "B", "Value": 1000.0},
            ]
        )

        result = _build_portfolio_allocation_data(
            component_prices=raw,
            allocation={"A": 50.0, "B": 50.0},
            composition_df=composition,
            start_val=10000,
        )

        assert result is not None
        positions = result["positions"]
        pct = positions.div(positions.sum(axis=1), axis=0)

        assert positions.loc[dates[4], "A"] > 5000
        assert positions.loc[rebal_date, "A"] == pytest.approx(5000)
        assert pct.loc[rebal_date, "A"] == pytest.approx(0.5)
        assert pct.loc[rebal_date, "B"] == pytest.approx(0.5)
        assert dates[-1] not in result["seg_starts"]

    def test_only_helper_is_cached(self):
        assert hasattr(_get_cached_portfolio_allocation_data, "__wrapped__")
        assert not hasattr(render_portfolio_allocation, "__wrapped__")

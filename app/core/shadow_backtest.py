import contextlib
import io
import logging
import os
from collections import deque
from dataclasses import dataclass
from datetime import date, timedelta

import numpy as np
import pandas as pd

from app.common.constants import Freq
from app.common.special_tickers import (
    provider_fallback_ticker,
    zero_return_series,
)

log = logging.getLogger("shadow_backtest")

@dataclass
class TaxLot:
    ticker: str
    date_acquired: pd.Timestamp
    quantity: float
    cost_basis_per_share: float
    total_cost_basis: float

    def sell_shares(self, shares_to_sell):
        """
        Returns (shares_sold, cost_basis_of_sold_shares, remaining_lot)
        """
        if shares_to_sell >= self.quantity:
            return self.quantity, self.total_cost_basis, None
        
        fraction = shares_to_sell / self.quantity
        cost_basis_sold = self.total_cost_basis * fraction
        
        remaining_qty = self.quantity - shares_to_sell
        remaining_cost = self.total_cost_basis - cost_basis_sold
        
        remaining_lot = TaxLot(
            ticker=self.ticker,
            date_acquired=self.date_acquired,
            quantity=remaining_qty,
            cost_basis_per_share=self.cost_basis_per_share,
            total_cost_basis=remaining_cost
        )
        
        return shares_to_sell, cost_basis_sold, remaining_lot

def parse_ticker(ticker):
    """
    Parses ticker string to extract base symbol and modifiers.
    Returns (base, params_dict)
    """
    if "?" in ticker:
        parts = ticker.split("?")
        base = parts[0]
        query = parts[1]
        
        # Simple manual parsing to avoid urllib overhead/issues
        params = {}
        pairs = query.split("&")
        for p in pairs:
            if "=" in p:
                k, v = p.split("=", 1)
                params[k] = v
    else:
        base = ticker
        params = {}
        
    mapped_base = provider_fallback_ticker(base)
    if mapped_base != base.upper():
        base = mapped_base
    
    return base, params

# ... (rest of imports/helpers) ...



def get_tax_treatment(ticker):
    """
    Determines the tax treatment of an asset based on its ticker.
    Returns: 'Equity', 'Collectible', or 'Section1256'
    """
    base, _ = parse_ticker(ticker)
    
    # Collectibles (Physical Metal ETFs)
    collectibles = {
        "GLD", "IAU", "SLV", "SGOL", "PPLT", "PALL", "BAR", "AAAU", "PHYS", "PSLV",
        "OUNZ"
    }
    
    # Section 1256 (Futures-based ETFs, Currency ETFs, VIX)
    # Note: DBMF is a managed futures ETF, treated as 1256? 
    # DBMF issues a 1099, but gains are often from Cayman sub (ordinary) or 1256.
    # For simplicity/conservatism in this "Stresser", let's treat known 1256 names.
    # GSG (GSCI) is a partnership (K-1), usually 60/40.
    # PDBC is 1099 but uses Cayman sub (Ordinary/Capital mix).
    # Let's stick to the clear 1256 ones or K-1 commodities.
    section1256 = {
        "GSG", "DBC", "UUP", "CYA", "DBMF", "KMLM", "CTA", # Managed Futures often 60/40
        "VIX", "^VIX", "VXX", "UVXY", "SVXY", "UVIX", "SVIX", "ZVOL" # VIX Futures
    }
    
    if base in collectibles:
        return "Collectible"
    elif base in section1256:
        return "Section1256"
    else:
        return "Equity"

def fetch_prices(tickers, start_date, end_date, invest_dividends=True):
    """Fetches adjusted close prices for unique base tickers via provider chain."""
    from app.services.price_providers import get_price_provider

    unique_bases = set()
    for t in tickers:
        base, _ = parse_ticker(t)
        unique_bases.add(base)

    f = io.StringIO()
    print(f"DEBUG: fetch_prices via provider chain", file=f)
    print(f"  Tickers: {list(unique_bases)}", file=f)
    print(f"  Start: {start_date}", file=f)
    print(f"  End: {end_date}", file=f)
    print("-" * 20, file=f)

    local_prices = {}
    if "ZEROX" in unique_bases:
        local_prices["ZEROX"] = zero_return_series(start_date, end_date)

    provider_bases = sorted(unique_bases - set(local_prices))
    if provider_bases:
        provider = get_price_provider()
        prices = provider.fetch_prices(
            provider_bases, str(start_date), str(end_date), adjusted=invest_dividends,
        )
    else:
        prices = pd.DataFrame()

    output = f.getvalue()

    if isinstance(prices, pd.Series):
        prices = prices.to_frame()

    if local_prices:
        prices = pd.concat([prices, pd.DataFrame(local_prices)], axis=1)

    return prices, output

def _check_drift(positions, allocation, threshold_pct):
    """Return (triggered, max_drift_pct, worst_ticker)."""
    total_val = sum(positions.values())
    if total_val <= 0:
        return False, 0.0, ""
    max_drift, drifter = 0.0, ""
    total_target = sum(allocation.values())
    for ticker, target_w in allocation.items():
        current_w = (positions.get(ticker, 0.0) / total_val) * 100.0
        normalized_target = (target_w / total_target) * 100.0
        drift = abs(current_w - normalized_target)
        if drift > max_drift:
            max_drift, drifter = drift, ticker
    return max_drift > threshold_pct, max_drift, drifter


def _check_drift_fast(position_values, target_weights_pct, threshold_pct, tickers):
    """Array-based drift check used by the hot rebalance path."""
    total_val = float(position_values.sum())
    if total_val <= 0:
        return False, 0.0, ""

    current_weights_pct = (position_values / total_val) * 100.0
    drifts = np.abs(current_weights_pct - target_weights_pct)
    drift_idx = int(np.argmax(drifts))
    max_drift = float(drifts[drift_idx])
    return max_drift > threshold_pct, max_drift, tickers[drift_idx]


def run_shadow_backtest(allocation, start_val, start_date, end_date, api_port_series=None, rebalance_freq="Yearly", cashflow=0.0, cashflow_freq="Monthly", prices_df=None, rebalance_month=1, rebalance_day=1, custom_freq="Yearly", invest_dividends=True, pay_down_margin=False, tax_config=None, custom_rebal_config=None, threshold_pct=0.0, pm_buy_block=False, pm_buy_block_threshold=100000.0, starting_loan=0.0, margin_rate_annual=8.0, draw_monthly=0.0, draw_start_date=None, draw_monthly_retirement=0.0, retirement_date=None, dca_in_retirement=True, loan_repayment=0.0, loan_repayment_freq="Monthly"):
    """
    Runs a local backtest using Tax Lots (FIFO) to calculate ST/LT capital gains.
    Supports periodic cashflow injections (DCA).
    
    Args:
        prices_df (pd.DataFrame): Optional pre-fetched daily prices (index=Date, columns=Ticker).
                                  If provided, yfinance is skipped.
        rebalance_month (int): Month to rebalance (1-12) if rebalance_freq="Custom" and custom_freq="Yearly".
        rebalance_day (int): Day to rebalance (1-28) for custom rebalancing.
        custom_freq (str): For "Custom" rebalance mode - "Yearly", "Quarterly", or "Monthly".
        invest_dividends (bool): Whether to use Total Return (Adj Close) or Price Return (Close).
        pay_down_margin (bool): Whether cashflows are used to pay down margin loan.
        tax_config (dict): Global tax configuration (optional).
        custom_rebal_config (dict): Configuration for custom rebalancing logic (optional).
    """

    tickers = list(allocation.keys())
    total_alloc = sum(allocation.values())
    target_weight_fractions = np.array([allocation[ticker] / total_alloc for ticker in tickers], dtype=float)
    target_weight_pcts = target_weight_fractions * 100.0
    tax_treatments = {ticker: get_tax_treatment(ticker) for ticker in tickers}

    # Initialize logs
    logs = []
    log.info("Shadow Backtest started: %d tickers, %s to %s", len(tickers), start_date, end_date)
    log.info("Config: loan=$%.0f rate=%.2f%% draw=$%.0f/mo ret_draw=$%.0f/mo draw_start=%s ret_date=%s dca_in_ret=%s",
             starting_loan, margin_rate_annual, draw_monthly, draw_monthly_retirement,
             draw_start_date, retirement_date, dca_in_retirement)
    logs.append(f"Starting Shadow Backtest for {len(tickers)} tickers.")
    logs.append(f"Timeframe: {start_date} to {end_date}")
    logs.append(f"Rebalance Frequency: {rebalance_freq}")
    if rebalance_freq == "Custom":
        logs.append(f"Custom Frequency: {custom_freq}, Day: {rebalance_day}" + (f", Month: {rebalance_month}" if custom_freq == "Yearly" else ""))
    if rebalance_freq in ("Threshold", "Threshold+Calendar"):
        logs.append(f"Drift Threshold: {threshold_pct}%")

    # PM Buy Block tracking
    pm_blocked_dates = []
    loan_balance = starting_loan
    daily_rate = (1 + margin_rate_annual / 100.0) ** (1.0 / 252.0) - 1
    _prev_month = None  # for monthly draw detection (set after dates is established)
    if draw_monthly > 0 or draw_monthly_retirement > 0:
        _ds_label = str(draw_start_date) if draw_start_date is not None else "backtest start"
        logs.append(f"Monthly Draw: ${draw_monthly:,.0f}/mo starting {_ds_label}")
        if draw_monthly_retirement > 0 and retirement_date is not None:
            logs.append(f"Retirement Draw: ${draw_monthly_retirement:,.0f}/mo starting {retirement_date}")
    if pm_buy_block:
        logs.append(f"PM Buy Block: Enabled (threshold ${pm_buy_block_threshold:,.0f}, loan ${starting_loan:,.0f}, rate {margin_rate_annual:.1f}%, draw ${draw_monthly:,.0f}/mo)")

    if cashflow > 0:
        logs.append(f"DCA: ${cashflow:,.2f} {cashflow_freq}")
    
    if prices_df is not None and not prices_df.empty:
        logs.append("Using pre-fetched price data (Hybrid Simulation).")
        prices_base = prices_df
        # FORCE SLICE to respect start/end dates
        # Ensure index is datetime
        if not isinstance(prices_base.index, pd.DatetimeIndex):
             prices_base.index = pd.to_datetime(prices_base.index)
             
        # Normalize Timezone to Naive to prevent mismatch during slicing
        if prices_base.index.tz is not None:
             prices_base.index = prices_base.index.tz_localize(None)
        
        # Slicing
        prices_base = prices_base.loc[start_date:end_date]
        
        yf_output = None
    else:
        prices_base, yf_output = fetch_prices(tickers, start_date, end_date, invest_dividends=invest_dividends)
    
    if yf_output:
        logs.append("\n--- yfinance Output & Payload ---")
        logs.append(yf_output)
        logs.append("---------------------------------\n")
    
    # Calculate daily returns for base assets. Missing prices must remain NaN:
    # a stale sleeve should not become a fake 0% return on a newer trading day.
    returns_base = prices_base.pct_change(fill_method=None)
    
    # Construct leveraged returns
    returns_port = pd.DataFrame(index=returns_base.index)
    
    # Log Data Availability
    logs.append("\n[Data Check]")
    for col in returns_base.columns:
        valid_idx = returns_base[col].dropna().index
        if not valid_idx.empty:
            logs.append(f"  {col}: {valid_idx[0].date()} to {valid_idx[-1].date()}")
        else:
            logs.append(f"  {col}: NO DATA")
    logs.append("-" * 20)
    
    missing_tickers = []
    
    # Try to find FFR (Risk Free Rate) for leverage cost
    # Look for 'BIL' or '^IRX' in prices_base
    ffr_series = None
    if 'BIL' in returns_base.columns:
        ffr_series = returns_base['BIL'] # BIL Return ~ FFR (minus ER)
    elif 'CASHX' in returns_base.columns:
         ffr_series = returns_base['CASHX']
    elif 'SHV' in returns_base.columns:
         ffr_series = returns_base['SHV']
    
    # Defaults
    DEFAULT_FFR_ANNUAL = 0.04 # 4% assumption if no data
    
    for ticker in tickers:
        base, params = parse_ticker(ticker)
        
        # Fetch Base Returns
        if base in returns_base.columns:
            r_s = returns_base[base].copy()
            
            # 2. Try Unmapped Base (Base check above covers mapped, now check raw)
        else:
             raw_base = ticker.split("?")[0]
             if raw_base in returns_base.columns:
                 r_s = returns_base[raw_base].copy()
             else:
                 missing_tickers.append(ticker)
                 returns_port[ticker] = np.nan
                 continue
                 
        # --- Apply Testfol Modifiers ---
        
        # 1. Underlying Expense (UE)
        # Adds UE% annually to underlying return. (Adjust for drag)
        ue = float(params.get('UE', 0.0))
        if ue != 0:
            r_s = r_s + (ue / 100.0) / 252.0
            
        # 2. Underlying Return/Vol overrides (UR, UV) - SKIP (Complex)
        
        # 3. Correlation (UC) - SKIP (Complex)
        
        # 4. Leverage (L, SW, SP)
        L = float(params.get('L', 1.0))
        SW = float(params.get('SW', 1.0))
        # Default SP = 0.0%
        SP = float(params.get('SP', 0.0))
        
        if L != 1.0:
            # Leverage Formula: R_lev = L * R_u - Cost
            # Cost = SW * (L - 1) * (FFR + SP)
            
            # Calculate FFR term
            if ffr_series is not None:
                # FFR series is daily return. We need Annualized rate?
                # Usually: Cost is calculated on Daily basis.
                # (FFR_daily * 252) ~ FFR_annual.
                # Currently ffr_series is daily return.
                # So Cost_daily = SW * (L - 1) * (R_ffr + SP_daily)
                sp_daily = (SP / 100.0) / 252.0
                cost_daily = SW * (L - 1) * (ffr_series.reindex(r_s.index).fillna(DEFAULT_FFR_ANNUAL/252) + sp_daily)
            else:
                 # Constant assumption
                 ffr_daily = DEFAULT_FFR_ANNUAL / 252.0
                 sp_daily = (SP / 100.0) / 252.0
                 cost_daily = SW * (L - 1) * (ffr_daily + sp_daily)
                 
            r_s = (r_s * L) - cost_daily
            
        # 5. Expense Ratio (E)
        # Subtracts E% annually
        # Default E logic: 0% nominal.
        # "adds an extra 0.333% for every point of negative leverage... or 0.5% for >1"
        # Since params.get('E') returns the explicit value if set, we use that.
        # If user didn't set E, do we auto-calc? The user said "By default E is 0%, but..."
        # Implies the tool *should* auto-calc if not specified? 
        # "The default value for E assumes..." -> implies if E is omitted, we might want defaults.
        # But let's stick to 0 unless specified for simplicity, or implement the rule?
        # Let's implement the rule if E is missing.
        
        if 'E' in params:
            e_val = float(params['E'])
        else:
            e_val = 0.0

        if e_val != 0:
            r_s = r_s - (e_val / 100.0) / 252.0
            
        # Assign to portfolio
        returns_port[ticker] = r_s
            
    if missing_tickers:
        logs.append(f"CRITICAL ERROR: Missing price data for: {', '.join(missing_tickers)}")
        # We cannot simulate if assets are missing.
        empty_series = pd.Series(dtype=float)
        empty_series.index = pd.DatetimeIndex([], dtype='datetime64[ns]')
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), logs, empty_series, pd.Series(dtype=float), []
            
    # Determine Simulation Start Date (Hybrid Logic)
    # 2024-12-17: Fix - Enforce user start_date if provided
    if start_date:
        start_ts = pd.to_datetime(start_date)
        returns_port = returns_port[returns_port.index >= start_ts]

    valid_returns = returns_port.dropna()
    
    if valid_returns.empty:
        logs.append(f"Error: No valid data found after start date {start_date}.")
        empty_series = pd.Series(dtype=float)
        empty_series.index = pd.DatetimeIndex([], dtype='datetime64[ns]')
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), logs, empty_series, pd.Series(dtype=float), []
        
    sim_start_date = valid_returns.index[0]
    logs.append(f"First valid data found at: {sim_start_date.date()}")

    # Initialize Portfolio
    if sim_start_date > pd.to_datetime(start_date) and api_port_series is not None:
        logs.append(f"Hybrid Mode Active: Real data starts late ({sim_start_date.date()}).")
        try:
            current_val = api_port_series.asof(sim_start_date)
            logs.append(f"Handover from API: Initializing portfolio at ${current_val:,.2f} on {sim_start_date.date()}")
        except (KeyError, IndexError, TypeError):
            current_val = start_val
            logs.append(f"Handover Failed: Defaulting to start_val ${start_val:,.2f}")
        initial_lot_date = sim_start_date
        logs.append("Initializing Positions (Hybrid):")
    else:
        logs.append(f"Standard Mode: Starting simulation from beginning ({start_date}).")
        current_val = start_val
        initial_lot_date = pd.to_datetime(start_date)
        logs.append("Initializing Positions (Standard):")

    position_values = target_weight_fractions * current_val
    lot_quantities = position_values.copy()
    lot_cost_basis = position_values.copy()
    tax_lots = {}
    for ticker, value in zip(tickers, position_values):
        tax_lots[ticker] = deque([
            TaxLot(
                ticker=ticker,
                date_acquired=initial_lot_date,
                quantity=value,
                cost_basis_per_share=1.0,
                total_cost_basis=value,
            )
        ])
        logs.append(f"  {ticker}: ${value:,.2f}")

    trades = []
    composition = []
    unrealized_pl_by_year = []
    portfolio_history_vals = []
    portfolio_history_dates = []

    returns_port = valid_returns.reindex(columns=tickers)
    dates = returns_port.index
    return_matrix = returns_port.to_numpy(dtype=float, copy=False)
    date_months = dates.month.to_numpy()
    date_quarters = dates.quarter.to_numpy()
    date_years = dates.year.to_numpy()
    date_days = dates.day.to_numpy()
    days_in_month = dates.days_in_month.to_numpy()
    month_change = np.zeros(len(dates), dtype=bool)
    quarter_change = np.zeros(len(dates), dtype=bool)
    year_change = np.zeros(len(dates), dtype=bool)
    if len(dates) > 1:
        month_change[:-1] = date_months[1:] != date_months[:-1]
        quarter_change[:-1] = date_quarters[1:] != date_quarters[:-1]
        year_change[:-1] = date_years[1:] != date_years[:-1]
    month_end_or_last = month_change.copy()
    month_end_or_last[-1] = True

    # Initialize History with Start Date
    portfolio_history_dates.append(dates[0])
    portfolio_history_vals.append(current_val)

    # Initialize TWR Tracking
    twr_history_dates = [dates[0]]
    twr_history_vals = [1.0] # Start at 1.0 (indexed to 100)
    curr_twr = 1.0
    prev_post_flow_val = current_val # Tracks value after flows, for next day's return calc

    last_rebal_month = (0, 0)
    last_rebal_quarter = (0, 0)
    last_rebal_year = -1
    _prev_month = int(date_months[0])  # Initialize for monthly draw detection

    for i in range(1, len(dates)):
        date = dates[i]
        is_last_trading_day = i == len(dates) - 1
        is_month_boundary = bool(month_change[i])
        is_quarter_boundary = bool(quarter_change[i])
        is_year_boundary = bool(year_change[i])
        is_month_end = bool(month_end_or_last[i])

        # 1. Update Position Values
        position_values *= (1.0 + return_matrix[i])
        day_port_val = float(position_values.sum())

        # Record daily value (may be updated below after DCA injection)
        portfolio_history_dates.append(date)
        portfolio_history_vals.append(day_port_val)

        # Loan balance tracking (interest, draws, repayments)
        if pm_buy_block or draw_monthly > 0 or draw_monthly_retirement > 0 or loan_repayment > 0:
            if loan_balance > 0:
                loan_balance *= (1 + daily_rate)

            cur_month = int(date_months[i])
            if (draw_monthly > 0 or draw_monthly_retirement > 0) and _prev_month is not None and is_month_boundary:
                if draw_start_date is None or date.date() >= draw_start_date:
                    _cur_date = date.date()
                    if draw_monthly_retirement > 0 and retirement_date is not None and _cur_date >= retirement_date:
                        _draw_amt = draw_monthly_retirement
                    else:
                        _draw_amt = draw_monthly
                    if _draw_amt > 0:
                        loan_balance += _draw_amt
                        _draw_label = "RetDraw" if (retirement_date is not None and _cur_date >= retirement_date) else "Draw"
                        logs.append(f"  💸 {_draw_label} {_cur_date}: +${_draw_amt:,.0f} → loan ${loan_balance:,.0f}")

            if loan_repayment > 0 and loan_balance > 0 and not is_last_trading_day:
                _repay = False
                if loan_repayment_freq == "Monthly" and is_month_boundary:
                    _repay = True
                elif loan_repayment_freq == "Quarterly" and is_quarter_boundary:
                    _repay = True
                elif loan_repayment_freq == "Yearly" and is_year_boundary:
                    _repay = True
                if _repay:
                    _pre_repay = loan_balance
                    loan_balance = max(0, loan_balance - loan_repayment)
                    logs.append(f"  💰 Repay {date.date()}: -${_pre_repay - loan_balance:,.0f} → loan ${loan_balance:,.0f}")

            _prev_month = cur_month

        # Calculate TWR
        if prev_post_flow_val > 0:
            daily_ret = day_port_val / prev_post_flow_val
        else:
            daily_ret = 1.0

        curr_twr *= daily_ret
        twr_history_dates.append(date)
        twr_history_vals.append(curr_twr)

        # 2. Check for Cashflow Injection (DCA)
        should_inject = False
        if cashflow > 0:
            _dca_cutoff_date = draw_start_date or retirement_date
            if not dca_in_retirement and _dca_cutoff_date is not None and date.date() >= _dca_cutoff_date:
                should_inject = False
            elif cashflow_freq == Freq.YEARLY:
                should_inject = is_year_boundary
            elif cashflow_freq == Freq.QUARTERLY:
                should_inject = is_quarter_boundary
            elif cashflow_freq == Freq.MONTHLY:
                should_inject = is_month_boundary

        if should_inject:
            logs.append(f"\n[DCA] {date.date()} | Injecting ${cashflow:,.2f}")

            for idx, ticker in enumerate(tickers):
                amount = cashflow * target_weight_fractions[idx]
                current_pos = float(position_values[idx])
                total_qty = float(lot_quantities[idx])
                current_price = current_pos / total_qty if total_qty > 0 else 1.0
                shares_bought = amount / current_price if current_price != 0 else 0.0

                position_values[idx] = current_pos + amount
                lot_quantities[idx] += shares_bought
                lot_cost_basis[idx] += amount
                day_port_val += amount

                tax_lots[ticker].append(
                    TaxLot(
                        ticker=ticker,
                        date_acquired=date,
                        quantity=shares_bought,
                        cost_basis_per_share=current_price,
                        total_cost_basis=amount,
                    )
                )

                logs.append(f"  {ticker}: Bought ${amount:,.2f} ({shares_bought:.4f} units)")
                trades.append({
                    "Date": date,
                    "Ticker": ticker,
                    "Trade Amount": amount,
                    "Realized ST P&L": 0,
                    "Realized LT P&L": 0,
                    "Realized LT (Collectible)": 0,
                    "Realized P&L": 0,
                    "Price (Est)": current_price,
                })

            portfolio_history_vals[-1] = day_port_val

        prev_post_flow_val = day_port_val

        # 3. Check for Rebalance
        should_rebal = False

        if rebalance_freq == Freq.YEARLY:
            should_rebal = is_year_boundary
        elif rebalance_freq == Freq.QUARTERLY:
            should_rebal = is_quarter_boundary
        elif rebalance_freq == Freq.MONTHLY:
            should_rebal = is_month_boundary
        elif rebalance_freq == "Custom":
            effective_day = min(rebalance_day, int(days_in_month[i]))
            current_day = int(date_days[i])
            current_month = int(date_months[i])

            if custom_freq == "Monthly":
                if current_day >= effective_day or (is_month_end and effective_day > current_day):
                    month_key = (date.year, current_month)
                    if last_rebal_month != month_key:
                        should_rebal = True
                        last_rebal_month = month_key
                        if current_day != effective_day:
                            logs.append(f"ℹ️ Rebalance shifted: Target day {rebalance_day} → Actual {date.date()} {'(Month End)' if is_month_end else '(Next Open)'}")

            elif custom_freq == "Quarterly":
                if current_month in (1, 4, 7, 10) and (current_day >= effective_day or (is_month_end and effective_day > current_day)):
                    quarter_key = (date.year, current_month)
                    if last_rebal_quarter != quarter_key:
                        should_rebal = True
                        last_rebal_quarter = quarter_key
                        if current_day != effective_day:
                            logs.append(f"ℹ️ Rebalance shifted: Target day {rebalance_day} → Actual {date.date()} {'(Month End)' if is_month_end else '(Next Open)'}")

            else:  # Yearly (default)
                if current_month == rebalance_month and (current_day >= effective_day or (is_month_end and effective_day > current_day)):
                    if last_rebal_year != date.year:
                        should_rebal = True
                        last_rebal_year = date.year
                        if current_day != effective_day:
                            logs.append(f"ℹ️ Rebalance shifted: Target day {rebalance_day} → Actual {date.date()} {'(Month End)' if is_month_end else '(Next Open)'}")

        elif rebalance_freq == "Threshold":
            triggered, max_drift, drifter = _check_drift_fast(position_values, target_weight_pcts, threshold_pct, tickers)
            if triggered:
                should_rebal = True
                logs.append(f"  ⚡ Drift trigger: {drifter} at {max_drift:.1f}pp (threshold {threshold_pct:.1f}pp)")

        elif rebalance_freq == "Threshold+Calendar":
            is_check_date = False
            if custom_freq == Freq.YEARLY:
                is_check_date = is_year_boundary
            elif custom_freq == Freq.QUARTERLY:
                is_check_date = is_quarter_boundary
            elif custom_freq == Freq.MONTHLY:
                is_check_date = is_month_boundary
            if is_check_date:
                triggered, max_drift, drifter = _check_drift_fast(position_values, target_weight_pcts, threshold_pct, tickers)
                if triggered:
                    should_rebal = True
                    logs.append(f"  ⚡ Drift trigger at {custom_freq} check: {drifter} at {max_drift:.1f}pp")
                else:
                    logs.append(f"  ✓ {custom_freq} check: max drift {max_drift:.1f}pp < {threshold_pct:.1f}pp — skip")

        # 4. PM Buy Block — skip entire rebalance when equity < threshold
        if should_rebal and pm_buy_block:
            estimated_equity = day_port_val - max(loan_balance, 0)
            if estimated_equity < pm_buy_block_threshold:
                should_rebal = False
                pm_blocked_dates.append(date)
                logs.append(f"\n  ⛔ PM Buy Block: Equity ${estimated_equity:,.0f} < ${pm_buy_block_threshold:,.0f} (loan ${loan_balance:,.0f}) — rebalance skipped")

        if should_rebal:
            logs.append(f"\n[REBALANCE] {date.date()} | Portfolio Value: ${day_port_val:,.2f}")
            logs.append(f"{'Ticker':<10} {'Action':<6} {'Amount':<12} {'ST Gain':<12} {'LT Gain':<12}")
            logs.append("-" * 75)

            target_values = day_port_val * target_weight_fractions
            trade_amounts = target_values - position_values

            for idx, ticker in enumerate(tickers):
                current_pos = float(position_values[idx])
                trade_amt = float(trade_amounts[idx])
                st_gain = 0.0
                lt_gain = 0.0
                lt_gain_collectible = 0.0

                if abs(trade_amt) < 0.01:
                    continue

                tax_treatment = tax_treatments[ticker]

                if trade_amt > 0:
                    total_qty = float(lot_quantities[idx])
                    current_price = current_pos / total_qty if total_qty > 0 else 1.0
                    shares_bought = trade_amt / current_price if current_price != 0 else 0.0

                    tax_lots[ticker].append(
                        TaxLot(
                            ticker=ticker,
                            date_acquired=date,
                            quantity=shares_bought,
                            cost_basis_per_share=current_price,
                            total_cost_basis=trade_amt,
                        )
                    )
                    position_values[idx] = current_pos + trade_amt
                    lot_quantities[idx] += shares_bought
                    lot_cost_basis[idx] += trade_amt

                    logs.append(f"{ticker:<10} {'BUY':<6} ${trade_amt:,.2f}      -             -")
                    trades.append({
                        "Date": date,
                        "Ticker": ticker,
                        "Trade Amount": trade_amt,
                        "Realized ST P&L": 0,
                        "Realized LT P&L": 0,
                        "Realized LT (Collectible)": 0,
                        "Realized P&L": 0,
                        "Price (Est)": current_price,
                    })

                elif trade_amt < 0:
                    sell_val = -trade_amt
                    total_qty = float(lot_quantities[idx])
                    current_price = current_pos / total_qty if total_qty > 0 else 1.0
                    shares_remaining_to_sell = sell_val / current_price if current_price != 0 else 0.0

                    while shares_remaining_to_sell > 1e-12 and tax_lots[ticker]:
                        current_lot = tax_lots[ticker][0]
                        sold_qty, cost_basis_sold, remaining_lot = current_lot.sell_shares(shares_remaining_to_sell)
                        proceeds = sold_qty * current_price
                        gain_loss = proceeds - cost_basis_sold
                        is_long_term = (date - current_lot.date_acquired).days > 365

                        if tax_treatment == "Section1256":
                            lt_gain += gain_loss * 0.60
                            st_gain += gain_loss * 0.40
                        elif tax_treatment == "Collectible":
                            if is_long_term:
                                lt_gain_collectible += gain_loss
                            else:
                                st_gain += gain_loss
                        else:
                            if is_long_term:
                                lt_gain += gain_loss
                            else:
                                st_gain += gain_loss

                        lot_quantities[idx] -= sold_qty
                        lot_cost_basis[idx] -= cost_basis_sold

                        if remaining_lot:
                            tax_lots[ticker][0] = remaining_lot
                            shares_remaining_to_sell = 0.0
                        else:
                            tax_lots[ticker].popleft()
                            shares_remaining_to_sell -= sold_qty

                    if abs(lot_quantities[idx]) < 1e-12:
                        lot_quantities[idx] = 0.0
                    if abs(lot_cost_basis[idx]) < 1e-8:
                        lot_cost_basis[idx] = 0.0

                    position_values[idx] = current_pos + trade_amt
                    if abs(position_values[idx]) < 1e-8:
                        position_values[idx] = 0.0

                    logs.append(f"{ticker:<10} {'SELL':<6} ${sell_val:,.2f}      ${st_gain:,.2f}      ${lt_gain:,.2f}")
                    trades.append({
                        "Date": date,
                        "Ticker": ticker,
                        "Trade Amount": trade_amt,
                        "Realized ST P&L": st_gain,
                        "Realized LT P&L": lt_gain,
                        "Realized LT (Collectible)": lt_gain_collectible,
                        "Realized P&L": st_gain + lt_gain + lt_gain_collectible,
                        "Price (Est)": current_price,
                    })

            logs.append("-" * 75)

            for idx, ticker in enumerate(tickers):
                composition.append({
                    "Date": date,
                    "Ticker": ticker,
                    "Value": float(position_values[idx]),
                })

        # 5. Record Month-End Unrealized P&L
        if is_month_end:
            unrealized_pl_by_year.append({
                "Date": date,
                "Year": date.year,
                "Unrealized P&L": float(position_values.sum() - lot_cost_basis.sum()),
            })
                
    # Record final composition snapshot
    if dates.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), logs, pd.Series(dtype=float), pd.Series(dtype=float), []
        
    last_date = dates[-1]
    
    already_recorded = False
    if composition:
        last_comp_date = composition[-1]["Date"]
        if last_comp_date == last_date:
            already_recorded = True
            
    if not already_recorded:
        logs.append(f"\n[SNAPSHOT] Final Composition at {last_date.date()}")
        for idx, ticker in enumerate(tickers):
            composition.append({
                "Date": last_date,
                "Ticker": ticker,
                "Value": float(position_values[idx])
            })
            logs.append(f"  {ticker}: ${position_values[idx]:,.2f}")
                
    trades_df = pd.DataFrame(trades)
    composition_df = pd.DataFrame(composition)
    
    if not trades_df.empty:
        trades_df["Year"] = trades_df["Date"].dt.year
        # Aggregate P&L by Year (Summing ST and LT)
        agg_cols = ["Realized ST P&L", "Realized LT P&L"]
        if "Realized LT (Collectible)" in trades_df.columns:
            agg_cols.append("Realized LT (Collectible)")
        pl_by_year = trades_df.groupby("Year")[agg_cols].sum().sort_index()
        pl_by_year["Realized P&L"] = pl_by_year["Realized ST P&L"] + pl_by_year["Realized LT P&L"]
        if "Realized LT (Collectible)" in pl_by_year.columns:
            pl_by_year["Realized P&L"] += pl_by_year["Realized LT (Collectible)"]
    else:
        pl_by_year = pd.DataFrame()
        
    if not composition_df.empty:
        composition_df["Year"] = composition_df["Date"].dt.year
        
    # Calculate Unrealized P&L (Monthly)
    unrealized_pl_df = pd.DataFrame(unrealized_pl_by_year)
    if not unrealized_pl_df.empty:
        unrealized_pl_df = unrealized_pl_df.set_index("Date").sort_index()
    else:
        unrealized_pl_df = pd.DataFrame()

    logs.append("Shadow Backtest Completed Successfully.")
    
    # Generate Portfolio Series (Daily Total Value)
    if portfolio_history_vals:
        portfolio_series = pd.Series(portfolio_history_vals, index=pd.DatetimeIndex(portfolio_history_dates), name="Portfolio")
        # Ensure it's sorted and unique
        portfolio_series = portfolio_series[~portfolio_series.index.duplicated(keep='last')].sort_index()
    else:
        # Crucial: Must have DatetimeIndex for resample() to work downstream
        portfolio_series = pd.Series(dtype=float)
        portfolio_series.index = pd.DatetimeIndex([], dtype='datetime64[ns]')

    # Generate TWR Series
    if twr_history_vals:
        twr_series = pd.Series(twr_history_vals, index=pd.DatetimeIndex(twr_history_dates), name="TWR")
        twr_series = twr_series[~twr_series.index.duplicated(keep='last')].sort_index()
    else:
        twr_series = pd.Series(dtype=float)

    # Write logs to file (only when TESTFOL_DEBUG is set)
    if os.environ.get("TESTFOL_DEBUG"):
        try:
            debug_dir = os.path.join(os.path.dirname(__file__), "../../debug_tools")
            os.makedirs(debug_dir, exist_ok=True)
            with open(os.path.join(debug_dir, "shadow_backtest.log"), "w") as f:
                f.write("\n".join(logs))
            logs.append(f"Logs written to debug_tools/shadow_backtest.log")
        except Exception as e:
            logs.append(f"Failed to write log file: {e}")
        
    return trades_df, pl_by_year, composition_df, unrealized_pl_df, logs, portfolio_series, twr_series, pm_blocked_dates

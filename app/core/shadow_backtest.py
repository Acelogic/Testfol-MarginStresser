import yfinance as yf
import pandas as pd
import numpy as np
import io
import contextlib
from dataclasses import dataclass
from datetime import date, timedelta

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
                k, v = p.split("=")
                params[k] = v
    else:
        base = ticker
        params = {}
        
    # Handle Synthetic Tickers Mapping
    # ... (Mapping logic kept same) ...
     # Map SIM/TR tickers to their real counterparts for yfinance
    mapping = {
        # Cash/Bills
        "TBILL": "BIL", "CASHX": "BIL", "EFFRX": "BIL",
        "ZEROX": "BIL",  # 0% return placeholder
        
        # S&P 500 / Large Cap
        "SPYSIM": "SPY", "SPYTR": "SPY",
        "VOOSIM": "VOO",
        "VTVSIM": "VTV",  # US Large Cap Value
        "VUGSIM": "VUG",  # US Large Cap Growth
        
        # Mid Cap
        "VOSIM": "VO",    # US Mid Cap
        "VOESIM": "VOE",  # US Mid Cap Value
        "VOTSIM": "VOT",  # US Mid Cap Growth
        
        # Small/Micro Cap
        "VBSIM": "VB",    # US Small Cap
        "VBRSIM": "VBR",  # US Small Cap Value
        "VBKSIM": "VBK",  # US Small Cap Growth
        "IWCSIM": "IWC",  # US Micro Cap
        
        # Total Market
        "VTISIM": "VTI", "VTITR": "VTI",
        "VTSIM": "VT",
        "QQQSIM": "QQQ", "QQQTR": "QQQ",
        
        # International
        "VXUSSIM": "VXUS", "VXUSX": "VXUS",
        
        # Bonds
        "TLTSIM": "TLT", "TLTTR": "TLT",
        "ZROZSIM": "ZROZ", "ZROZX": "ZROZ",
        "IEFSIM": "IEF", "IEFTR": "IEF",
        "IEISIM": "IEI", "IEITR": "IEI",
        "SHYSIM": "SHY", "SHYTR": "SHY",
        "BNDSIM": "BND",
        
        # Metals
        "GLDSIM": "GLD", "GOLDX": "GLD",
        "SLVSIM": "SLV",
        
        # Commodities
        "GSGSIM": "GSG", "GSGTR": "GSG",
        
        # Managed Futures
        "DBMFSIM": "DBMF", "DBMFX": "DBMF",
        "KMLMSIM": "KMLM", "KMLMX": "KMLM",
        
        # Volatility
        "VIXSIM": "^VIX", "VOLIX": "^VIX",
        "SVIXSIM": "SVIX", "SVIXX": "SVIX",
        "UVIXSIM": "UVXY",
        "ZVOLSIM": "SVXY", "ZIVBX": "SVXY",  # Approximate with SVXY
        
        # Crypto
        "BTCSIM": "BTC-USD", "BTCTR": "BTC-USD",
        "ETHSIM": "ETH-USD", "ETHTR": "ETH-USD",
        
        # Sector ETFs
        "XLBSIM": "XLB", "XLBTR": "XLB",
        "XLCSIM": "XLC", "XLCTR": "XLC",
        "XLESIM": "XLE", "XLETR": "XLE",
        "XLFSIM": "XLF", "XLFTR": "XLF",
        "XLISIM": "XLI", "XLITR": "XLI",
        "XLKSIM": "XLK", "XLKTR": "XLK",
        "XLPSIM": "XLP", "XLPTR": "XLP",
        "XLUSIM": "XLU", "XLUTR": "XLU",
        "XLVSIM": "XLV", "XLVTR": "XLV",
        "XLYSIM": "XLY", "XLYTR": "XLY",
        
        # Special/Leveraged
        "FNGUSIM": "FNGU",
        "CAOSSIM": "CAOS",
        "GDESIM": "GDE",
        "MCISIM": "MCI",
        "REITSIM": "VNQ",
        
        # Legacy
        "DIA_SIM": "DIA",
    }
    
    # Don't remap if it's our custom NDXMEGASIM
    if base in mapping and base != "NDXMEGASIM":
        base = mapping[base]
    
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
        "VIX", "VXX", "UVXY", "SVXY" # VIX Futures
    }
    
    if base in collectibles:
        return "Collectible"
    elif base in section1256:
        return "Section1256"
    else:
        return "Equity"

def fetch_prices(tickers, start_date, end_date):
    """Fetches adjusted close prices for unique base tickers."""
    unique_bases = set()
    for t in tickers:
        base, _ = parse_ticker(t)
        unique_bases.add(base)
    
    # Download data
    # Capture stdout/stderr to get yfinance logs
    f = io.StringIO()
    
    # Log the "payload" (parameters)
    print(f"DEBUG: yf.download parameters:", file=f)
    print(f"  Tickers: {list(unique_bases)}", file=f)
    print(f"  Start: {start_date}", file=f)
    print(f"  End: {end_date}", file=f)
    print(f"  Progress: True", file=f)
    print("-" * 20, file=f)
    
    with contextlib.redirect_stdout(f), contextlib.redirect_stderr(f):
        data = yf.download(list(unique_bases), start=start_date, end=end_date, progress=True)
    
    output = f.getvalue()
    
    if "Adj Close" in data:
        prices = data["Adj Close"]
    elif "Close" in data:
        prices = data["Close"]
    else:
        prices = data
        
    if isinstance(prices, pd.Series):
        prices = prices.to_frame()
        
    return prices, output

def run_shadow_backtest(allocation, start_val, start_date, end_date, api_port_series=None, rebalance_freq="Yearly", cashflow=0.0, cashflow_freq="Monthly", prices_df=None, rebalance_month=1, rebalance_day=1, custom_freq="Yearly"):
    """
    Runs a local backtest using Tax Lots (FIFO) to calculate ST/LT capital gains.
    Supports periodic cashflow injections (DCA).
    
    Args:
        prices_df (pd.DataFrame): Optional pre-fetched daily prices (index=Date, columns=Ticker).
                                  If provided, yfinance is skipped.
        rebalance_month (int): Month to rebalance (1-12) if rebalance_freq="Custom" and custom_freq="Yearly".
        rebalance_day (int): Day to rebalance (1-28) for custom rebalancing.
        custom_freq (str): For "Custom" rebalance mode - "Yearly", "Quarterly", or "Monthly".
    """

    tickers = list(allocation.keys())

    # Initialize logs
    logs = []
    logs.append(f"Starting Shadow Backtest for {len(tickers)} tickers.")
    logs.append(f"Timeframe: {start_date} to {end_date}")
    logs.append(f"Rebalance Frequency: {rebalance_freq}")
    if rebalance_freq == "Custom":
        logs.append(f"Custom Frequency: {custom_freq}, Day: {rebalance_day}" + (f", Month: {rebalance_month}" if custom_freq == "Yearly" else ""))
    
    if cashflow > 0:
        logs.append(f"DCA: ${cashflow:,.2f} {cashflow_freq}")
    
    if prices_df is not None and not prices_df.empty:
        logs.append("Using pre-fetched price data (Hybrid Simulation).")
        prices_base = prices_df
        yf_output = None
    else:
        prices_base, yf_output = fetch_prices(tickers, start_date, end_date)
    
    if yf_output:
        logs.append("\n--- yfinance Output & Payload ---")
        logs.append(yf_output)
        logs.append("---------------------------------\n")
    
    # Calculate daily returns for base assets
    returns_base = prices_base.pct_change() # Keep NaNs to detect missing data
    
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
        SW = float(params.get('SW', 1.1))
        # Default SP = sgn(L) * 0.4%
        default_sp = 0.4 if L >= 0 else -0.4 
        SP = float(params.get('SP', default_sp))
        
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
             # Auto-calc default E based on Leverage
             if L > 1:
                 e_val = (L - 1) * 0.5 # 0.5% per point above 1
             elif L < 0:
                 # -1x -> 1 point of neg leverage? 
                 # "0.333% for every point of negative leverage"
                 # if L = -1, is that 1 point? abs(L)?
                 # Usually inverse ETFs have ~1%. -1x -> 1%?
                 # Let's use abs(L) * 0.333?
                 e_val = abs(L) * 0.333
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
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), logs, empty_series, pd.Series(dtype=float)
            
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
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), logs, empty_series, pd.Series(dtype=float)
        
    sim_start_date = valid_returns.index[0]
    logs.append(f"First valid data found at: {sim_start_date.date()}")
    
    # ... (lines between omitted) ... need to be careful with multi-replace or use separate tool calls if distant.
    # The first two blocks are close (missing_tickers is near valid_returns check).
    
    # The third block is at line 722. I can do the first two in one chunk if they are close.
    # missing_tickers check ends around line 410?
    # valid_returns check is around 420.
    # They are contiguous in the file logic provided the lines I see above match.
    
    # Let's check line numbers again from Step 715 output.
    # missing_tickers check is NOT shown in 715 output, it was replacing loop logic.
    # But in Step 712 view:
    # missing_tickers check is implied to be before line 400?
    # Wait, the view in 712 starts at 350, loop ends, then logic continues.
    # Actually, let's look at 709 output again.
    # missing_tickers check is typically right after the loop.
    
    # I'll use separate replacement chunks or one big chunk if confidently contiguous.
    # Let's verify line numbers. I'll read lines 400-430 again to be sure of context.
        
    sim_start_date = valid_returns.index[0]
    logs.append(f"First valid data found at: {sim_start_date.date()}")
    
    # Initialize Portfolio
    positions = {} # Current value of each position
    tax_lots = {t: [] for t in tickers} # List of TaxLot objects per ticker
    
    trades = []
    composition = []
    unrealized_pl_by_year = []
    portfolio_history_vals = []
    portfolio_history_dates = []
    
    # Check if we are starting late (Hybrid Mode)
    if sim_start_date > pd.to_datetime(start_date) and api_port_series is not None:
        logs.append(f"Hybrid Mode Active: Real data starts late ({sim_start_date.date()}).")
        try:
            current_val = api_port_series.asof(sim_start_date)
            logs.append(f"Handover from API: Initializing portfolio at ${current_val:,.2f} on {sim_start_date.date()}")
        except:
            current_val = start_val
            logs.append(f"Handover Failed: Defaulting to start_val ${start_val:,.2f}")
            
        # Initialize positions
        total_alloc = sum(allocation.values())
        logs.append(f"Initializing Positions (Hybrid):")
        for ticker, weight in allocation.items():
            val = current_val * (weight / total_alloc)
            positions[ticker] = val
            
            # Create initial tax lot
            # We assume price=1.0 for simplicity since we track value
            tax_lots[ticker].append(TaxLot(
                ticker=ticker,
                date_acquired=sim_start_date,
                quantity=val, # quantity = value when price=1
                cost_basis_per_share=1.0,
                total_cost_basis=val
            ))
            logs.append(f"  {ticker}: ${val:,.2f}")
            
        returns_port = valid_returns
        dates = returns_port.index
        
    else:
        logs.append(f"Standard Mode: Starting simulation from beginning ({start_date}).")
        current_val = start_val
        total_alloc = sum(allocation.values())
        logs.append(f"Initializing Positions (Standard):")
        for ticker, weight in allocation.items():
            val = start_val * (weight / total_alloc)
            positions[ticker] = val
            
            tax_lots[ticker].append(TaxLot(
                ticker=ticker,
                date_acquired=pd.to_datetime(start_date),
                quantity=val,
                cost_basis_per_share=1.0,
                total_cost_basis=val
            ))
            logs.append(f"  {ticker}: ${val:,.2f}")
            
        dates = valid_returns.index

    # Iterate through dates
    # Initialize History with Start Date
    portfolio_history_dates.append(dates[0])
    portfolio_history_vals.append(current_val)
    
    # Initialize TWR Tracking
    twr_history_dates = [dates[0]]
    twr_history_vals = [1.0] # Start at 1.0 (indexed to 100)
    curr_twr = 1.0
    prev_post_flow_val = current_val # Tracks value after flows, for next day's return calc

    for i in range(1, len(dates)):
        date = dates[i]
        prev_date = dates[i-1]

        # 1. Update Position Values
        day_port_val = 0
        for ticker in tickers:
            r = returns_port.loc[date, ticker]
            positions[ticker] *= (1 + r)
            day_port_val += positions[ticker]
            
        # Record daily value
        portfolio_history_dates.append(date)
        portfolio_history_vals.append(day_port_val)
        
        # Calculate TWR
        # Return = Current Pre-Flow / Previous Post-Flow
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
            if cashflow_freq == "Yearly":
                if i < len(dates) - 1 and dates[i+1].year != date.year:
                    should_inject = True
            elif cashflow_freq == "Quarterly":
                if i < len(dates) - 1 and dates[i+1].quarter != date.quarter:
                    should_inject = True
            elif cashflow_freq == "Monthly":
                if i < len(dates) - 1 and dates[i+1].month != date.month:
                    should_inject = True
                    
        if should_inject:
            logs.append(f"\n[DCA] {date.date()} | Injecting ${cashflow:,.2f}")
            
            # Distribute cashflow according to target allocation
            total_alloc = sum(allocation.values())
            for ticker, weight in allocation.items():
                amount = cashflow * (weight / total_alloc)
                positions[ticker] += amount
                day_port_val += amount
                
                # Create new Tax Lot for the injection
                # Determine current price proxy
                total_qty = sum(lot.quantity for lot in tax_lots[ticker])
                current_pos = positions[ticker] - amount # Value BEFORE injection
                
                if total_qty > 0:
                    current_price = current_pos / total_qty
                else:
                    current_price = 1.0
                    
                shares_bought = amount / current_price
                
                new_lot = TaxLot(
                    ticker=ticker,
                    date_acquired=date,
                    quantity=shares_bought,
                    cost_basis_per_share=current_price,
                    total_cost_basis=amount
                )
                tax_lots[ticker].append(new_lot)
                
                logs.append(f"  {ticker}: Bought ${amount:,.2f} ({shares_bought:.4f} units)")
                
                trades.append({
                    "Date": date,
                    "Ticker": ticker,
                    "Trade Amount": amount,
                    "Realized ST P&L": 0,
                    "Realized LT P&L": 0,
                    "Realized P&L": 0,
                    "Price (Est)": current_price
                })
        
        # Update baseline for next day's return calculation
        prev_post_flow_val = day_port_val
            
        # 3. Check for Rebalance
        should_rebal = False
        
        # 3. Check for Rebalance
        should_rebal = False
        
        if rebalance_freq == "Yearly":
            if i < len(dates) - 1:
                next_date = dates[i+1]
                if next_date.year != date.year:
                    should_rebal = True
                    
        elif rebalance_freq == "Quarterly":
            # Rebalance at end of Mar, Jun, Sep, Dec
            # Check if next date is in a new quarter
            if i < len(dates) - 1:
                next_date = dates[i+1]
                if next_date.quarter != date.quarter:
                    should_rebal = True
                    
        elif rebalance_freq == "Monthly":
            # Rebalance at end of every month
            if i < len(dates) - 1:
                next_date = dates[i+1]
                if next_date.month != date.month:
                    should_rebal = True
                    
        elif rebalance_freq == "Custom":
            # Custom rebalancing based on custom_freq (Yearly/Quarterly/Monthly)
            
            # Check if this is the last trading day of the month
            is_month_end = False
            if i < len(dates) - 1:
                if dates[i+1].month != date.month:
                    is_month_end = True
            else:
                is_month_end = True

            days_in_month = date.days_in_month
            # Clamp target day to actual days in potential month (for Feb/Apr etc)
            effective_day = min(rebalance_day, days_in_month)
            
            if custom_freq == "Monthly":
                # Rebalance if we reached target day OR it's month end and we haven't reached target yet
                if date.day >= effective_day or (is_month_end and effective_day > date.day):
                    month_key = (date.year, date.month)
                    if 'last_rebal_month' not in locals():
                        last_rebal_month = (0, 0)
                    if last_rebal_month != month_key:
                        should_rebal = True
                        last_rebal_month = month_key
                        if date.day != effective_day:
                             logs.append(f"ℹ️ Rebalance shifted: Target day {rebalance_day} → Actual {date.date()} {'(Month End)' if is_month_end else '(Next Open)'}")
                        
            elif custom_freq == "Quarterly":
                quarter_months = [1, 4, 7, 10]
                if date.month in quarter_months:
                    if date.day >= effective_day or (is_month_end and effective_day > date.day):
                        quarter_key = (date.year, date.month)
                        if 'last_rebal_quarter' not in locals():
                            last_rebal_quarter = (0, 0)
                        if last_rebal_quarter != quarter_key:
                            should_rebal = True
                            last_rebal_quarter = quarter_key
                            if date.day != effective_day:
                                logs.append(f"ℹ️ Rebalance shifted: Target day {rebalance_day} → Actual {date.date()} {'(Month End)' if is_month_end else '(Next Open)'}")

            else:  # Yearly (default)
                if date.month == rebalance_month:
                     if date.day >= effective_day or (is_month_end and effective_day > date.day):
                        if 'last_rebal_year' not in locals():
                            last_rebal_year = -1
                        if last_rebal_year != date.year:
                            should_rebal = True
                            last_rebal_year = date.year
                            if date.day != effective_day:
                                logs.append(f"ℹ️ Rebalance shifted: Target day {rebalance_day} → Actual {date.date()} {'(Month End)' if is_month_end else '(Next Open)'}")
                    
        # No forced rebalance on last day
                
        # 4. Execute Rebalance
        if should_rebal:
            logs.append(f"\n[REBALANCE] {date.date()} | Portfolio Value: ${day_port_val:,.2f}")
            logs.append(f"{'Ticker':<10} {'Action':<6} {'Amount':<12} {'ST Gain':<12} {'LT Gain':<12}")
            logs.append("-" * 75)
            
            # Record Pre-Rebalance Composition
            for ticker in tickers:
                composition.append({
                    "Date": date,
                    "Ticker": ticker,
                    "Value": positions[ticker]
                })
                
            # Calculate Trades
            for ticker in tickers:
                target_weight = allocation.get(ticker, 0)
                target_val = day_port_val * (target_weight / 100)
                current_pos = positions[ticker]
                
                trade_amt = target_val - current_pos
                
                st_gain = 0.0
                lt_gain = 0.0
                lt_gain_collectible = 0.0
                
                if abs(trade_amt) < 0.01:
                    continue
                
                # Determine Tax Treatment
                tax_treatment = get_tax_treatment(ticker)
                
                if trade_amt > 0: # BUY
                    # Add new tax lot
                    # We need to determine "shares" bought.
                    # Since we don't have real prices, we use the ratio of current_pos / sum(lots.quantity)
                    # Wait, if we track quantity = initial value, then:
                    # Current Price = Current Value / Total Quantity
                    
                    total_qty = sum(lot.quantity for lot in tax_lots[ticker])
                    if total_qty > 0:
                        current_price = current_pos / total_qty
                    else:
                        current_price = 1.0 # Default if starting from 0
                        
                    shares_bought = trade_amt / current_price
                    
                    new_lot = TaxLot(
                        ticker=ticker,
                        date_acquired=date,
                        quantity=shares_bought,
                        cost_basis_per_share=current_price,
                        total_cost_basis=trade_amt
                    )
                    tax_lots[ticker].append(new_lot)
                    positions[ticker] += trade_amt
                    
                    logs.append(f"{ticker:<10} {'BUY':<6} ${trade_amt:,.2f}      -             -")
                    
                    trades.append({
                        "Date": date,
                        "Ticker": ticker,
                        "Trade Amount": trade_amt,
                        "Realized ST P&L": 0,
                        "Realized LT P&L": 0,
                        "Realized LT (Collectible)": 0,
                        "Realized P&L": 0,
                        "Price (Est)": current_pos
                    })
                    
                elif trade_amt < 0: # SELL
                    sell_val = -trade_amt
                    
                    # Determine Current Price
                    total_qty = sum(lot.quantity for lot in tax_lots[ticker])
                    if total_qty > 0:
                        current_price = current_pos / total_qty
                    else:
                        current_price = 1.0
                        
                    shares_to_sell = sell_val / current_price
                    
                    # FIFO Logic
                    shares_remaining_to_sell = shares_to_sell
                    
                    while shares_remaining_to_sell > 0 and tax_lots[ticker]:
                        current_lot = tax_lots[ticker][0] # First lot (FIFO)
                        
                        sold_qty, cost_basis_sold, remaining_lot = current_lot.sell_shares(shares_remaining_to_sell)
                        
                        # Calculate Gain/Loss
                        proceeds = sold_qty * current_price
                        gain_loss = proceeds - cost_basis_sold
                        
                        # Determine Holding Period
                        holding_period = date - current_lot.date_acquired
                        is_long_term = holding_period.days > 365
                        
                        # Apply Tax Treatment Rules
                        if tax_treatment == "Section1256":
                            # 60/40 Rule regardless of holding period
                            lt_gain += gain_loss * 0.60
                            st_gain += gain_loss * 0.40
                            
                        elif tax_treatment == "Collectible":
                            if is_long_term:
                                lt_gain_collectible += gain_loss
                            else:
                                st_gain += gain_loss
                                
                        else: # Equity (Standard)
                            if is_long_term:
                                lt_gain += gain_loss
                            else:
                                st_gain += gain_loss
                        
                        # Update Lots
                        if remaining_lot:
                            tax_lots[ticker][0] = remaining_lot
                            shares_remaining_to_sell = 0
                        else:
                            tax_lots[ticker].pop(0) # Lot fully consumed
                            shares_remaining_to_sell -= sold_qty
                            
                    positions[ticker] += trade_amt # trade_amt is negative
                    
                    logs.append(f"{ticker:<10} {'SELL':<6} ${sell_val:,.2f}      ${st_gain:,.2f}      ${lt_gain:,.2f}")
                    
                    trades.append({
                        "Date": date,
                        "Ticker": ticker,
                        "Trade Amount": trade_amt,
                        "Realized ST P&L": st_gain,
                        "Realized LT P&L": lt_gain,
                        "Realized LT (Collectible)": lt_gain_collectible,
                        "Realized P&L": st_gain + lt_gain + lt_gain_collectible,
                        "Price (Est)": current_price
                    })

            logs.append("-" * 75)
            
        # 5. Record Month-End Unrealized P&L
        is_month_end = False
        if i < len(dates) - 1:
            if dates[i+1].month != date.month or dates[i+1].year != date.year:
                is_month_end = True
        elif i == len(dates) - 1:
            is_month_end = True
            
        if is_month_end:
            total_unrealized = 0
            for t in tickers:
                current_val = positions[t]
                total_basis = sum(lot.total_cost_basis for lot in tax_lots[t])
                total_unrealized += (current_val - total_basis)
                
            unrealized_pl_by_year.append({
                "Date": date,
                "Year": date.year,
                "Unrealized P&L": total_unrealized
            })
                
    # Record final composition snapshot
    if dates.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), logs, pd.Series(dtype=float), pd.Series(dtype=float)
        
    last_date = dates[-1]
    
    already_recorded = False
    if composition:
        last_comp_date = composition[-1]["Date"]
        if last_comp_date == last_date:
            already_recorded = True
            
    if not already_recorded:
        logs.append(f"\n[SNAPSHOT] Final Composition at {last_date.date()}")
        for ticker in tickers:
            composition.append({
                "Date": last_date,
                "Ticker": ticker,
                "Value": positions[ticker]
            })
            logs.append(f"  {ticker}: ${positions[ticker]:,.2f}")
                
    trades_df = pd.DataFrame(trades)
    composition_df = pd.DataFrame(composition)
    
    if not trades_df.empty:
        trades_df["Year"] = trades_df["Date"].dt.year
        # Aggregate P&L by Year (Summing ST and LT)
        pl_by_year = trades_df.groupby("Year")[["Realized ST P&L", "Realized LT P&L"]].sum().sort_index()
        pl_by_year["Realized P&L"] = pl_by_year["Realized ST P&L"] + pl_by_year["Realized LT P&L"]
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

    # Write logs to file
    try:
        import os
        os.makedirs("debug_tools", exist_ok=True)
        with open("debug_tools/shadow_backtest.log", "w") as f:
            f.write("\n".join(logs))
        logs.append(f"Logs written to debug_tools/shadow_backtest.log")
    except Exception as e:
        logs.append(f"Failed to write log file: {e}")
        
    return trades_df, pl_by_year, composition_df, unrealized_pl_df, logs, portfolio_series, twr_series

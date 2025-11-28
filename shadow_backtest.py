import yfinance as yf
import pandas as pd
import numpy as np

import io
import contextlib

def parse_ticker(ticker):
    """Parses ticker string to extract base symbol and leverage."""
    # Format: Ticker?L=X
    if "?L=" in ticker:
        parts = ticker.split("?L=")
        base = parts[0]
        leverage = float(parts[1])
    else:
        base = ticker
        leverage = 1.0
        
    # Handle Synthetic Tickers Mapping
    # Map SIM tickers to their real counterparts for yfinance
    mapping = {
        "QQQSIM": "QQQ", "QQQTR": "QQQ",
        "SPYSIM": "SPY", "SPYTR": "SPY",
        "DIA_SIM": "DIA",
        "TBILL": "BIL", "CASHX": "BIL",
        "GLDSIM": "GLD", "GOLDX": "GLD",
        "TLTSIM": "TLT", "TLTTR": "TLT",
        "ZROZSIM": "ZROZ", "ZROZX": "ZROZ",
        "VXUSSIM": "VXUS", "VXUSX": "VXUS",
        "VTISIM": "VTI", "VTITR": "VTI",
        "VTSIM": "VT",
        "DBMFSIM": "DBMF", "DBMFX": "DBMF",
        "GSGSIM": "GSG", "GSGTR": "GSG",
        "IEFSIM": "IEF", "IEFTR": "IEF",
        "IEISIM": "IEI", "IEITR": "IEI",
        "SHYSIM": "SHY", "SHYTR": "SHY",
        "BTCSIM": "BTC-USD", "BTCTR": "BTC-USD",
        "ETHSIM": "ETH-USD", "ETHTR": "ETH-USD",
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
        "VIXSIM": "^VIX", "VOLIX": "^VIX"
    }
    
    if base in mapping:
        base = mapping[base]
    
    return base, leverage

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

def run_shadow_backtest(allocation, start_val, start_date, end_date, api_port_series=None, rebalance_freq="Yearly"):
    """
    Runs a local backtest. If real data is missing for early years (e.g. QQQ starts 1999),
    it waits until valid data exists and initializes using the API's portfolio value at that point.
    """

    tickers = list(allocation.keys())

    # Initialize logs
    logs = []
    logs.append(f"Starting Shadow Backtest for {len(tickers)} tickers.")
    logs.append(f"Timeframe: {start_date} to {end_date}")
    
    prices_base, yf_output = fetch_prices(tickers, start_date, end_date)
    
    if yf_output:
        logs.append("\n--- yfinance Output & Payload ---")
        logs.append(yf_output)
        logs.append("---------------------------------\n")
    
    # Calculate daily returns for base assets
    returns_base = prices_base.pct_change() # Keep NaNs to detect missing data
    
    # Construct leveraged returns
    returns_port = pd.DataFrame(index=returns_base.index)
    
    for ticker in tickers:
        base, leverage = parse_ticker(ticker)
        if base in returns_base.columns:
            returns_port[ticker] = returns_base[base] * leverage
        else:
            returns_port[ticker] = np.nan # Mark as missing
            
    # Determine Simulation Start Date (Hybrid Logic)
    # We can only simulate when we have data for ALL assets.
    # Drop rows where any asset is NaN
    valid_returns = returns_port.dropna()
    
    if valid_returns.empty:
        logs.append("Error: No valid data found for any common date range.")
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), logs
        
    sim_start_date = valid_returns.index[0]
    logs.append(f"First valid data found at: {sim_start_date.date()}")
    
    # Initialize Portfolio
    positions = {}
    cost_basis = {}
    trades = []
    composition = []
    unrealized_pl_by_year = []
    
    # Check if we are starting late (Hybrid Mode)
    if sim_start_date > pd.to_datetime(start_date) and api_port_series is not None:
        logs.append(f"Hybrid Mode Active: Real data starts late ({sim_start_date.date()}).")
        # We are starting late. Use Testfol's value as the truth up to this point.
        try:
            # Get value at sim_start_date (or closest before)
            current_val = api_port_series.asof(sim_start_date)
            logs.append(f"Handover from API: Initializing portfolio at ${current_val:,.2f} on {sim_start_date.date()}")
        except:
            current_val = start_val # Fallback
            logs.append(f"Handover Failed: Defaulting to start_val ${start_val:,.2f}")
            
        # Initialize positions assuming we rebalance into target at this start point
        total_alloc = sum(allocation.values())
        logs.append(f"Initializing Positions (Hybrid):")
        for ticker, weight in allocation.items():
            val = current_val * (weight / total_alloc)
            positions[ticker] = val
            cost_basis[ticker] = val
            logs.append(f"  {ticker}: ${val:,.2f} (Basis: ${val:,.2f})")
            
        # Filter returns to start from sim_start_date
        returns_port = valid_returns
        dates = returns_port.index
        
    else:
        logs.append(f"Standard Mode: Starting simulation from beginning ({start_date}).")
        # Normal Start
        current_val = start_val
        total_alloc = sum(allocation.values())
        logs.append(f"Initializing Positions (Standard):")
        for ticker, weight in allocation.items():
            val = start_val * (weight / total_alloc)
            positions[ticker] = val
            cost_basis[ticker] = val
            logs.append(f"  {ticker}: ${val:,.2f} (Basis: ${val:,.2f})")
            
        dates = valid_returns.index

    # Iterate through dates
    for i in range(1, len(dates)):
        date = dates[i]
        
        # 1. Update Position Values
        day_port_val = 0
        for ticker in tickers:
            r = returns_port.loc[date, ticker]
            positions[ticker] *= (1 + r)
            day_port_val += positions[ticker]
            
        # 2. Check for Rebalance
        should_rebal = False
        if rebalance_freq == "Yearly":
            if i < len(dates) - 1:
                next_date = dates[i+1]
                if next_date.year != date.year:
                    should_rebal = True
            # No forced rebalance on last day
                
        # 3. Execute Rebalance
        if should_rebal:
            logs.append(f"\n[REBALANCE] {date.date()} | Portfolio Value: ${day_port_val:,.2f}")
            logs.append(f"{'Ticker':<10} {'Current Val':<15} {'Target Val':<15} {'Trade Amt':<15} {'P&L':<15}")
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
                
                # Execute Trade
                positions[ticker] += trade_amt
                
                # Calculate P&L
                realized_pl = 0
                if trade_amt > 0: # BUY
                    cost_basis[ticker] += trade_amt
                elif trade_amt < 0: # SELL
                    sell_amt = -trade_amt
                    if current_pos > 0:
                        fraction_sold = sell_amt / current_pos
                        fraction_sold = min(fraction_sold, 1.0)
                        
                        basis_reduction = cost_basis[ticker] * fraction_sold
                        realized_pl = sell_amt - basis_reduction
                        cost_basis[ticker] -= basis_reduction
                    else:
                        realized_pl = sell_amt
                
                logs.append(f"{ticker:<10} ${current_pos:,.2f}      ${target_val:,.2f}      ${trade_amt:,.2f}      ${realized_pl:,.2f}")
                
                trades.append({
                    "Date": date,
                    "Ticker": ticker,
                    "Trade Amount": trade_amt,
                    "Realized P&L": realized_pl,
                    "Price (Est)": current_pos
                })
            logs.append("-" * 75)
            
        # 4. Record Year-End Unrealized P&L
        # We capture this at the end of the year (after any rebalancing)
        is_year_end = False
        if i < len(dates) - 1:
            if dates[i+1].year != date.year:
                is_year_end = True
        elif i == len(dates) - 1:
            is_year_end = True
            
        if is_year_end:
            total_unrealized = sum(positions[t] - cost_basis[t] for t in tickers)
            unrealized_pl_by_year.append({
                "Year": date.year,
                "Unrealized P&L": total_unrealized
            })
                
    # Record final composition snapshot
    if dates.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), logs
        
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
        pl_by_year = trades_df.groupby("Year")["Realized P&L"].sum().sort_index().to_frame()
    else:
        pl_by_year = pd.DataFrame()
        
    if not composition_df.empty:
        composition_df["Year"] = composition_df["Date"].dt.year
        
    # Calculate Unrealized P&L by Year
    unrealized_pl_df = pd.DataFrame(unrealized_pl_by_year)
    if not unrealized_pl_df.empty:
        unrealized_pl_df = unrealized_pl_df.set_index("Year").sort_index()
    else:
        unrealized_pl_df = pd.DataFrame()

    logs.append("Shadow Backtest Completed Successfully.")
    
    # Write logs to file
    try:
        import os
        os.makedirs("debug_tools", exist_ok=True)
        with open("debug_tools/shadow_backtest.log", "w") as f:
            f.write("\n".join(logs))
        logs.append(f"Logs written to debug_tools/shadow_backtest.log")
    except Exception as e:
        logs.append(f"Failed to write log file: {e}")
        
    return trades_df, pl_by_year, composition_df, unrealized_pl_df, logs

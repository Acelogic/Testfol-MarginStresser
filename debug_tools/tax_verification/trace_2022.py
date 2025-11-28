import sys
sys.path.append("..")
import testfol_api as api
import pandas as pd
import datetime as dt

# Configuration
start_date = dt.date(2012, 1, 1)
end_date = dt.date.today()
start_val = 10000.0
cashflow = 0.0
cashfreq = "Monthly"
invest_div = True
rebalance = "Yearly"

alloc_list = [
    {"Ticker":"AAPL?L=2","Weight %":7.5,"Maint %":50},
    {"Ticker":"MSFT?L=2","Weight %":7.5,"Maint %":50},
    {"Ticker":"AVGO?L=2","Weight %":7.5,"Maint %":50},
    {"Ticker":"AMZN?L=2","Weight %":7.5,"Maint %":50},
    {"Ticker":"META?L=2","Weight %":7.5,"Maint %":50},
    {"Ticker":"NVDA?L=2","Weight %":7.5,"Maint %":50},
    {"Ticker":"GOOGL?L=2","Weight %":7.5,"Maint %":50},
    {"Ticker":"TSLA?L=2","Weight %":7.5,"Maint %":50},
    {"Ticker":"GLD","Weight %":20,"Maint %":25},
    {"Ticker":"VXUS","Weight %":15,"Maint %":25},
    {"Ticker":"TQQQ","Weight %":5,"Maint %":75},
]

alloc_preview = {item["Ticker"]: item["Weight %"] for item in alloc_list}

print("Fetching backtest...")
try:
    port_series, stats, extra_data = api.fetch_backtest(
        start_date, end_date, start_val,
        cashflow, cashfreq, 60,
        invest_div, rebalance, alloc_preview
    )
    
    rebal_events = extra_data.get("rebalancing_events", [])
    
    # Initialize cost basis
    initial_val = port_series.iloc[0]
    cost_basis = {}
    total_alloc = sum(alloc_preview.values())
    for ticker, weight in alloc_preview.items():
        cost_basis[ticker] = initial_val * (weight / total_alloc)
        
    print(f"Initial Cost Basis: {cost_basis}")
    
    for group in rebal_events:
        tickers = group.get("tickers", [])
        events = group.get("events", [])
        n_tickers = len(tickers)
        
        for event in events:
            date_str = event[0]
            date = pd.to_datetime(date_str)
            
            # Only care about 2022
            if date.year != 2022:
                # We still need to process updates to keep cost basis current
                # But we can skip printing
                pass
            
            try:
                port_val = port_series.asof(date)
            except:
                continue
                
            if date.year == 2022:
                print(f"\n--- Rebalance Event: {date_str} (Port Val: {port_val:,.2f}) ---")
            
            for i, ticker in enumerate(tickers):
                trade_idx = 1 + 2*n_tickers + i
                if trade_idx >= len(event): break
                
                trade_pct = event[trade_idx]
                trade_amt = port_val * (trade_pct / 100)
                
                # Logic from process_rebalancing_data
                target_weight = alloc_preview.get(ticker, 0)
                curr_val = port_val * (target_weight / 100) # Using Target Weight as proxy for value? 
                # Wait, in the code we used `curr_val` for fraction_sold calculation.
                # If we reverted to Target Weight for `curr_val`, we are using that for P&L too!
                # Let's verify what the code actually does.
                
                realized_pl = 0
                
                if ticker not in cost_basis: cost_basis[ticker] = 0
                
                old_basis = cost_basis[ticker]
                
                if trade_amt > 0: # BUY
                    cost_basis[ticker] += trade_amt
                    if date.year == 2022:
                        print(f"BUY  {ticker}: Amt={trade_amt:,.2f} (Pct={trade_pct}%) | New Basis={cost_basis[ticker]:,.2f}")
                elif trade_amt < 0: # SELL
                    sell_amt = -trade_amt
                    
                    # CRITICAL: What is curr_val used here?
                    # In the current code (reverted), curr_val = port_val * (target_weight / 100)
                    # This assumes we are selling from a position that is EXACTLY at target weight?
                    # No, that's wrong. But that's what the code does now.
                    
                    fraction_sold = 0
                    if curr_val > 0:
                        fraction_sold = sell_amt / curr_val
                        fraction_sold = min(fraction_sold, 1.0)
                        
                        basis_reduction = cost_basis[ticker] * fraction_sold
                        realized_pl = sell_amt - basis_reduction
                        cost_basis[ticker] -= basis_reduction
                    else:
                        realized_pl = sell_amt
                        
                    if date.year == 2022:
                        print(f"SELL {ticker}: Amt={sell_amt:,.2f} (Pct={trade_pct}%) | Basis={old_basis:,.2f} | Val(Est)={curr_val:,.2f} | Frac={fraction_sold:.4f} | Reduct={basis_reduction:,.2f} | P&L={realized_pl:,.2f}")

except Exception as e:
    print(f"Error: {e}")

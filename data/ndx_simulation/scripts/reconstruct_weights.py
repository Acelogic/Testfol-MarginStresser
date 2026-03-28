import pandas as pd
import json
import numpy as np
import datetime
import os
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), "../src"))
import config
import calendar
import ndx_parser
import price_manager
from official_index_data import get_official_constituents

# Configuration
INPUT_CSV = config.COMPONENTS_FILE
MAPPING_FILE = os.path.join(config.ASSETS_DIR, "name_mapping.json")
OUTPUT_FILE = config.WEIGHTS_FILE
OFFICIAL_ALIGNMENT_FILE = os.path.join(config.RESULTS_DIR, "ndx_official_membership_alignment.csv")
PROXY_TICKER = "QQQ"
APPLY_OFFICIAL_NDX_FILTER = False
MAPPING_OVERRIDES = {
    "Xcel Energy, Inc.": "XEL",
}


def apply_mapping_overrides(mapping):
    """Patch known stale mappings in the legacy name map."""
    mapping = dict(mapping)
    mapping.update(MAPPING_OVERRIDES)
    return mapping

def load_data(): # Load Components
    print(f"Loading {config.COMPONENTS_FILE}...")
    df = pd.read_csv(config.COMPONENTS_FILE)
    df['Date'] = pd.to_datetime(df['Date'])
    
    # FILTERING: Remove corrupt rows (e.g. where Company name is a date)
    # Valid company names don't start with digits (usually)
    # This matches the fix in ndx_scanner.py
    df = df[~df['Company'].str.match(r'^\d{4}-\d{2}-\d{2}', na=False)]
    
    # Ensure Value is positive
    df['Value'] = pd.to_numeric(df['Value'], errors='coerce')
    df = df[df['Value'] > 0]

    # Filter future dates (to avoid speculative/forward-looking data)
    today = pd.Timestamp.now().normalize()
    df = df[df['Date'] <= today]

    # Load Name Mapping
    with open(MAPPING_FILE, "r") as f:
        mapping = json.load(f)

    mapping = apply_mapping_overrides(mapping)
    return df, mapping

def get_unique_tickers(mapping):
    tickers = list(set(mapping.values()))
    if PROXY_TICKER not in tickers:
        tickers.append(PROXY_TICKER)
    return tickers



def _third_friday(year, month):
    """Return the 3rd Friday of the given month/year."""
    c = calendar.Calendar(firstweekday=calendar.SUNDAY)
    monthcal = c.monthdatescalendar(year, month)
    fridays = [day for week in monthcal for day in week
               if day.weekday() == calendar.FRIDAY and day.month == month]
    return fridays[2] if len(fridays) >= 3 else fridays[-1]


def get_rebalance_dates(start_year=2000, end_year=2025):
    """
    Returns (effective_date, reference_date) pairs for each quarterly rebalance.

    Per Nasdaq methodology:
      - Reference date: Last trading day of Feb/May/Aug/Nov (weights determined here)
      - Effective date: First trading day following 3rd Friday of Mar/Jun/Sep/Dec
    """
    pairs = []
    # Reference months map to effective months: Feb→Mar, May→Jun, Aug→Sep, Nov→Dec
    ref_eff_months = [(2, 3), (5, 6), (8, 9), (11, 12)]

    current_year = start_year
    while current_year <= end_year + 1:
        for ref_month, eff_month in ref_eff_months:
            if current_year > end_year and eff_month > 3:
                break

            # Effective date: first business day after 3rd Friday of eff_month
            date_3rd_fri = _third_friday(current_year, eff_month)
            effective_date = pd.Timestamp(date_3rd_fri) + pd.offsets.BDay(1)

            # Reference date: last business day of ref_month
            ref_year = current_year
            # BMonthEnd(0) from day 1 gives the last business day of that month
            reference_date = pd.Timestamp(year=ref_year, month=ref_month, day=1) + pd.offsets.BMonthEnd(0)

            pairs.append((effective_date, reference_date))
        current_year += 1

    # Filter to not exceed now + 90 days
    limit = pd.Timestamp.now() + pd.Timedelta(days=90)
    pairs = [(e, r) for e, r in pairs if e <= limit]

    # Edge case: data starts 2000-06-30, inject bootstrap date
    start_override_eff = pd.Timestamp("2000-06-30")
    start_override_ref = pd.Timestamp("2000-05-31") - pd.offsets.BDay(0)
    if not any(e == start_override_eff for e, _ in pairs):
        pairs.append((start_override_eff, start_override_ref))
        pairs.sort(key=lambda x: x[0])

    return pairs

def reconstruct():
    df, mapping = load_data()
    tickers = get_unique_tickers(mapping)
    
    # Use Centralized Price Manager (handles caching)
    prices = price_manager.get_price_data(tickers, start_date="1999-01-01")
    
    if prices is None or prices.empty:
        print("No price data. Aborting.")
        return

    # Ensure index is Datetime
    prices.index = pd.to_datetime(prices.index)
    
    # Resample prices to daily if needed (fill fwd) to handle weekend quarters
    prices = prices.ffill()
    ndx_parser.process_files()
    
    rebalance_pairs = get_rebalance_dates()
    final_rows = []

    # Proxy Returns
    if PROXY_TICKER in prices.columns:
        proxy_prices = prices[PROXY_TICKER]
    else:
        proxy_prices = pd.Series(1.0, index=prices.index)

    print(f"Reconstructing weights for {len(rebalance_pairs)} quarters...")

    for effective_date, reference_date in rebalance_pairs:
        # Use effective_date for filing lookback and weight projection.
        # Theoretically, Nasdaq uses reference_date (end of prior month) for
        # weight determination, but since our filings are quarterly SEC data
        # (not live index weights), projecting to the effective date gives us
        # better weight approximation by using more recent filing data.
        q_date = effective_date

        # 1. Find latest preceding filing
        # Must be within reasonable window (e.g. 400 days) to avoid zombie filings
        lookback_window = pd.Timedelta(days=400)
        valid_filings = df[(df['Date'] <= q_date) & (df['Date'] >= q_date - lookback_window)]

        if valid_filings.empty:
            continue

        # Take latest filing for EACH company
        filing_data = valid_filings.sort_values('Date', ascending=False).drop_duplicates('Company', keep='first').copy()

        # 2. Project Value from filing date to effective date
        idx_quarter = prices.index.get_indexer([q_date], method='pad')[0]
        if idx_quarter == -1:
            continue
        date_q = prices.index[idx_quarter]
        
        # Pre-calculate Proxy Return for the Quarter Date? 
        # No, Proxy Return depends on the start date (filing date), which varies.
        # But efficiently, most filings are on the same few dates.
        # We can handle it per row.

        for _, row in filing_data.iterrows():
            name = row['Company']
            fil_date = row['Date'] # The specific filing date for this company
            
            # Skip invalid rows
            try:
                val_f = float(str(row['Value']).replace(',',''))
            except:
                continue

            ticker = mapping.get(name)
            
            # Find Price Index for this specific filing date
            if fil_date not in prices.index:
                # Approximate lookup
                idx_f = prices.index.get_indexer([fil_date], method='pad')[0]
                if idx_f == -1:
                    date_f = fil_date # Fallback, probably won't have price
                else:
                    date_f = prices.index[idx_f]
            else:
                date_f = fil_date
            
            val_q = val_f 
            
            if ticker and ticker in prices.columns:
                try:
                    p_f = prices.at[date_f, ticker]
                    p_q = prices.at[date_q, ticker]
                    
                    if pd.notna(p_f) and pd.notna(p_q) and p_f > 0:
                        ret = p_q / p_f
                        val_q = val_f * ret
                    else:
                        # Fallback to proxy
                        # Calculate proxy ret for this specific period
                        try:
                            px_f = proxy_prices.at[date_f]
                            px_q = proxy_prices.at[date_q]
                            if px_f > 0:
                                val_q = val_f * (px_q / px_f)
                        except:
                            pass
                except:
                     pass
            else:
                 # Unmapped: Proxy
                 try:
                    px_f = proxy_prices.at[date_f]
                    px_q = proxy_prices.at[date_q]
                    if px_f > 0:
                        val_q = val_f * (px_q / px_f)
                 except:
                    pass
            
            final_rows.append({
                "Date": q_date.date(),
                "Ticker": ticker if ticker else name, # Use Name if no ticker
                "Name": name,
                "Value": val_q,
                "IsMapped": bool(ticker)
            })

    # Convert to DataFrame
    res_df = pd.DataFrame(final_rows)
    
    # Aggregate duplicate tickers (e.g. merger variations or same ticker mappings)
    # Sum 'Value' for same ('Date', 'Ticker')
    # Keep metadata from first occurrence
    if not res_df.empty:
        res_df['Date'] = pd.to_datetime(res_df['Date'])
        res_df = res_df.groupby(['Date', 'Ticker'], as_index=False).agg({
            'Value': 'sum',
            'Name': 'first',
            'IsMapped': 'first'
        })

        print("Generating official NDX membership alignment audit...")
        alignment_rows = []

        for dt, grp in res_df.groupby('Date', sort=True):
            grp = grp.copy()
            mapped_tickers = set(grp.loc[grp['IsMapped'] == True, 'Ticker'])
            official_tickers = get_official_constituents("NDX", dt)

            if not official_tickers:
                alignment_rows.append({
                    "Date": dt.date(),
                    "OfficialCount": 0,
                    "KeptMappedCount": len(mapped_tickers),
                    "RemovedExtraCount": 0,
                    "MissingOfficialCount": 0,
                    "RemovedExtras": "",
                    "MissingOfficial": "",
                })
                continue

            official_set = set(official_tickers)
            kept_grp = grp[(grp['IsMapped'] == True) & (grp['Ticker'].isin(official_set))].copy()

            removed_extras = sorted(mapped_tickers - official_set)
            missing_official = sorted(official_set - set(kept_grp['Ticker']))

            alignment_rows.append({
                "Date": dt.date(),
                "OfficialCount": len(official_set),
                "KeptMappedCount": len(kept_grp),
                "RemovedExtraCount": len(removed_extras),
                "MissingOfficialCount": len(missing_official),
                "RemovedExtras": "|".join(removed_extras),
                "MissingOfficial": "|".join(missing_official),
            })

        alignment_df = pd.DataFrame(alignment_rows)
        alignment_df.to_csv(OFFICIAL_ALIGNMENT_FILE, index=False)
        print(f"Saved official membership alignment audit to {OFFICIAL_ALIGNMENT_FILE}")

        if APPLY_OFFICIAL_NDX_FILTER:
            print("Official NDX membership filter is enabled.")
            filtered_groups = []

            for dt, grp in res_df.groupby('Date', sort=True):
                official_tickers = get_official_constituents("NDX", dt)
                if not official_tickers:
                    filtered_groups.append(grp)
                    continue

                official_set = set(official_tickers)
                kept_grp = grp[(grp['IsMapped'] == True) & (grp['Ticker'].isin(official_set))].copy()

                if kept_grp.empty:
                    print(f"Warning: {dt.date()} official NDX filter yielded no mapped members; keeping reconstructed universe.")
                    filtered_groups.append(grp)
                else:
                    filtered_groups.append(kept_grp)

            res_df = pd.concat(filtered_groups, ignore_index=True)
    
    # Calculate Weights per Date
    # GroupBy Date sum
    print("Calculating final weights...")
    sums = res_df.groupby('Date')['Value'].transform('sum')
    res_df['Weight'] = res_df['Value'] / sums
    
    # ---------------------------------------------------------
    # APPLY NDX CAPPING RULES (Methodology_NDX.pdf)
    # To ensure the "Reconstructed" weights represent the Index at Rebalance.
    #
    # Quarterly: Trigger at 24%/48%, cap at 20%/40% (iterative)
    # Annual (Dec): Trigger at 15%/top5>40%, cap at 14%/38.5%
    #               + outside-top-5 capped at min(4.4%, 5th largest)
    # ---------------------------------------------------------
    
    def apply_ndx_capping(group, is_annual=False):
        """Apply NDX capping rules per Methodology_NDX.pdf.

        Quarterly constraints (checked FIRST — if both satisfied, no adjustment):
          - No company weight may exceed 24%
          - Aggregate weight of companies > 4.5% may not exceed 48%

        Quarterly Stage 1: If any weight > 24%, cap all at 20%
        Quarterly Stage 2: If aggregate(>4.5%) > 48%, set aggregate to 40%

        Annual Stage 1: If any weight > 15%, cap all at 14%
        Annual Stage 2: If top-5 aggregate > 40%, set to 38.5%; cap outside-top-5
                        at min(4.4%, weight of 5th largest)
        """
        w = group['Weight'].values.copy()
        tickers = group['Ticker'].values.copy()
        w_series = pd.Series(w, index=tickers)

        # Normalize
        if w_series.sum() > 0:
            w_series = w_series / w_series.sum()

        if not is_annual:
            # --- QUARTERLY RULES ---
            # "If neither constraint is violated, no further adjustments are made"
            if w_series.max() <= 0.24 and w_series[w_series > 0.045].sum() <= 0.48:
                return pd.DataFrame({'Ticker': w_series.index, 'CappedWeight': w_series.values})

            for _ in range(20):
                w_series = w_series / w_series.sum()

                # Stage 1: Trigger at 24%, cap at 20%
                if w_series.max() > 0.24:
                    over = w_series[w_series > 0.20]
                    if not over.empty:
                        surplus = (over - 0.20).sum()
                        w_series[w_series > 0.20] = 0.20
                        others = w_series[w_series < 0.20]
                        if not others.empty:
                            w_series[others.index] = others + (surplus * others / others.sum())

                # Stage 2: Aggregate(>4.5%) must not exceed 48%, target 40%
                above = w_series[w_series > 0.045]
                if above.sum() > 0.48:
                    target_agg = 0.40
                    scale = target_agg / above.sum()
                    scaled_above = above * scale
                    surplus = above.sum() - target_agg

                    # Some stocks may have dropped below 4.5% after scaling
                    w_series[above.index] = scaled_above
                    below = w_series[w_series <= 0.045]
                    if not below.empty:
                        w_series[below.index] = below + (surplus * below / below.sum())

                # Check convergence
                if w_series.max() <= 0.2401 and w_series[w_series > 0.045].sum() <= 0.4801:
                    break

        else:
            # --- ANNUAL RULES (December) ---
            # Check if adjustments needed at all
            w_sorted = w_series.sort_values(ascending=False)
            needs_stage1 = w_series.max() > 0.15
            needs_stage2 = w_sorted.iloc[:5].sum() > 0.40 if len(w_sorted) >= 5 else False

            if not needs_stage1 and not needs_stage2:
                return pd.DataFrame({'Ticker': w_series.index, 'CappedWeight': w_series.values})

            for _ in range(20):
                w_series = w_series / w_series.sum()

                # Stage 1: Trigger at 15%, cap at 14%
                if w_series.max() > 0.15:
                    over = w_series[w_series > 0.14]
                    if not over.empty:
                        surplus = (over - 0.14).sum()
                        w_series[w_series > 0.14] = 0.14
                        others = w_series[w_series <= 0.14]
                        if not others.empty:
                            w_series[others.index] = others + (surplus * others / others.sum())

                # Stage 2: Top-5 aggregate > 40% → set to 38.5%
                w_sorted = w_series.sort_values(ascending=False)
                if len(w_sorted) >= 5:
                    top5_tickers = w_sorted.iloc[:5].index
                    top5_sum = w_series[top5_tickers].sum()

                    if top5_sum > 0.40:
                        scale = 0.385 / top5_sum
                        w_series[top5_tickers] = w_series[top5_tickers] * scale
                        surplus = top5_sum - 0.385

                        # Distribute surplus to non-top-5
                        others_idx = w_series.index.difference(top5_tickers)
                        others = w_series[others_idx]
                        if not others.empty:
                            w_series[others_idx] = others + (surplus * others / others.sum())

                        # Cap outside-top-5 at min(4.4%, weight of 5th largest)
                        # Use the ORIGINAL top5_tickers (pre-redistribution sort)
                        # to determine "outside" — redistribution can push others
                        # above original top-5 members, but that doesn't reclassify them.
                        fifth_val = w_series[top5_tickers].min()  # 5th largest of original top-5
                        cap_val = min(0.044, fifth_val)

                        outside_idx = w_series.index.difference(top5_tickers)
                        outside_over = w_series[outside_idx][w_series[outside_idx] > cap_val]

                        if not outside_over.empty:
                            surplus2 = (outside_over - cap_val).sum()
                            w_series[outside_over.index] = cap_val
                            # Redistribute to uncapped outside-top-5 stocks
                            uncapped = w_series[outside_idx][w_series[outside_idx] < cap_val]
                            if not uncapped.empty:
                                w_series[uncapped.index] = uncapped + (surplus2 * uncapped / uncapped.sum())

                # Check convergence
                w_check = w_series.sort_values(ascending=False)
                stage1_ok = w_check.iloc[0] <= 0.1501
                stage2_ok = (len(w_check) < 5) or (w_check.iloc[:5].sum() <= 0.4001)
                # Outside-top-5 cap: use actual min(4.4%, 5th-largest) + tolerance
                if len(w_check) >= 5:
                    fifth = w_check.iloc[4]
                    outside_cap = min(0.044, fifth) + 0.001
                    outside_ok = (len(w_check) < 6) or (w_check.iloc[5:] <= outside_cap).all()
                else:
                    outside_ok = True
                if stage1_ok and stage2_ok and outside_ok:
                    break

        return pd.DataFrame({'Ticker': w_series.index, 'CappedWeight': w_series.values})

    print("Applying NDX capping rules (Quarterly: 24%/4.5% | Annual: 14%/38.5%)...")
    
    # Apply per Date
    # Create temp df to merge back
    capped_list = []
    
    for dt, grp in res_df.groupby('Date'):
        # Determine if Annual Reconstitution (December)
        # dt is Timestamp
        is_annual = (dt.month == 12)
        
        capped_grp = apply_ndx_capping(grp, is_annual=is_annual)
        capped_grp['Date'] = dt
        capped_list.append(capped_grp)
        
    capped_df = pd.concat(capped_list)
    
    # Merge back to update weights
    # Note: 'res_df' has multiple rows, 'capped_df' has adjusted weights
    res_df = res_df.merge(capped_df, on=['Date', 'Ticker'], how='left')
    
    # Overwrite Weight
    res_df['Weight'] = res_df['CappedWeight']
    res_df.drop(columns=['CappedWeight'], inplace=True)
    
    # Save
    res_df.to_csv(OUTPUT_FILE, index=False)
    print(f"Saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    reconstruct()

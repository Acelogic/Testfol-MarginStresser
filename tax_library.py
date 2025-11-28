import pandas as pd
import os

# Global cache for tax tables
_TAX_TABLES = {}

def load_tax_tables(csv_path):
    """
    Parses the Historical Income Tax Rates CSV and populates _TAX_TABLES.
    Structure: _TAX_TABLES[year][status] = [(threshold, rate), ...]
    """
    global _TAX_TABLES
    if _TAX_TABLES:
        return

    try:
        df = pd.read_csv(csv_path)
        
        # Normalize columns
        # The CSV structure is a bit complex with merged headers in the raw view, 
        # but pandas read_csv might handle it if we are careful.
        # Let's assume standard columns based on inspection:
        # 0: Year
        # 1: MFJ Rate, 3: MFJ Bracket
        # 4: MFS Rate, 6: MFS Bracket
        # 7: Single Rate, 9: Single Bracket
        # 10: HoH Rate, 12: HoH Bracket
        
        # We'll iterate through rows
        current_year = None
        
        for index, row in df.iterrows():
            try:
                year_str = str(row.iloc[0]).strip()
                if not year_str or year_str == "nan":
                    continue
                    
                year = int(year_str)
                
                if year not in _TAX_TABLES:
                    _TAX_TABLES[year] = {
                        "Single": [],
                        "Married Filing Jointly": [],
                        "Married Filing Separately": [],
                        "Head of Household": []
                    }
                
                # Helper to parse rate/bracket
                def parse_bracket(rate_str, bracket_str):
                    if pd.isna(rate_str) or pd.isna(bracket_str):
                        return None
                    try:
                        rate = float(str(rate_str).replace("%", "")) / 100.0
                        bracket = float(str(bracket_str).replace("$", "").replace(",", ""))
                        return (bracket, rate)
                    except:
                        return None

                # Single
                b_single = parse_bracket(row.iloc[7], row.iloc[9])
                if b_single: _TAX_TABLES[year]["Single"].append(b_single)
                
                # MFJ
                b_mfj = parse_bracket(row.iloc[1], row.iloc[3])
                if b_mfj: _TAX_TABLES[year]["Married Filing Jointly"].append(b_mfj)
                
                # MFS
                b_mfs = parse_bracket(row.iloc[4], row.iloc[6])
                if b_mfs: _TAX_TABLES[year]["Married Filing Separately"].append(b_mfs)
                
                # HoH
                b_hoh = parse_bracket(row.iloc[10], row.iloc[12])
                if b_hoh: _TAX_TABLES[year]["Head of Household"].append(b_hoh)
                
            except Exception:
                continue
                
        # Sort tables by threshold
        for year in _TAX_TABLES:
            for status in _TAX_TABLES[year]:
                _TAX_TABLES[year][status].sort(key=lambda x: x[0])
                
    except Exception as e:
        print(f"Error loading tax tables: {e}")

def calculate_historical_tax(year, taxable_income, filing_status="Single", csv_path="Historical Income Tax Rates and Brackets, 1862-2025.csv"):
    """
    Calculates tax based on historical ordinary income brackets.
    """
    load_tax_tables(csv_path)
    
    if year not in _TAX_TABLES:
        # Fallback to closest year or 2024
        if _TAX_TABLES:
            year = max(_TAX_TABLES.keys())
        else:
            return 0.0
            
    brackets = _TAX_TABLES.get(year, {}).get(filing_status, [])
    if not brackets:
        return 0.0
        
    # Calculate Progressive Tax
    tax = 0.0
    previous_bracket_limit = 0.0
    
    # Brackets are (Threshold, Rate). 
    # The threshold is usually the *start* of the bracket or the *end*?
    # In the CSV: "10% > $0", "12% > $11,000". 
    # This means 10% applies from 0 to 11,000. 12% applies above 11,000.
    # So the threshold is the LOWER bound of the bracket.
    
    for i in range(len(brackets)):
        lower_bound, rate = brackets[i]
        
        # Determine upper bound
        if i < len(brackets) - 1:
            upper_bound = brackets[i+1][0]
        else:
            upper_bound = float("inf")
            
        # Check overlap with income
        # We are taxing the chunk of income between lower_bound and upper_bound
        
        # Actually, the logic is:
        # Income in [lower_bound, upper_bound) is taxed at rate.
        
        if taxable_income > lower_bound:
            taxable_amount = min(taxable_income, upper_bound) - lower_bound
            tax += taxable_amount * rate
            
    return tax

# Global cache for capital gains rates
_CAP_GAINS_RATES = {}

def load_capital_gains_rates(excel_path):
    """
    Parses the Federal Capital Gains Tax Rates Excel and populates _CAP_GAINS_RATES.
    Structure: _CAP_GAINS_RATES[year] = rate (float, e.g. 0.20)
    """
    global _CAP_GAINS_RATES
    if _CAP_GAINS_RATES:
        return

    try:
        # Read Sheet1, no header, cols 0 (Year) and 5 (Rate)
        df = pd.read_excel(excel_path, sheet_name="Sheet1", header=None, usecols=[0, 5])
        df.columns = ["Year", "Rate"]
        
        for index, row in df.iterrows():
            try:
                year = int(row["Year"])
                rate_val = row["Rate"]
                
                # Handle potential non-numeric data or percentages
                if pd.isna(rate_val):
                    continue
                    
                # Rate in Excel seems to be in percent (e.g. 20 for 20%)
                # But let's check if it's < 1 (e.g. 0.20) just in case
                rate = float(rate_val)
                if rate > 1.0:
                    rate /= 100.0
                    
                _CAP_GAINS_RATES[year] = rate
                
            except Exception:
                continue
                
    except Exception as e:
        print(f"Error loading capital gains rates: {e}")

def get_capital_gains_inclusion_rate(year):
    """
    Returns the percentage of long-term capital gains that was historically included in taxable income.
    Based on historical tax law changes.
    """
    if year < 1934:
        return 1.0 # 100% included (but 12.5% cap applied 1922-1933)
    elif 1934 <= year <= 1937:
        return 0.40 # Sliding scale. Using 40% (assets held 5-10 years) as a representative proxy.
    elif 1938 <= year <= 1941:
        return 0.50 # 50% for assets held > 24 months
    elif 1942 <= year <= 1978:
        return 0.50 # 50% deduction (so 50% included)
    elif 1979 <= year <= 1986:
        return 0.40 # 60% deduction (so 40% included)
    else:
        return 1.0 # 100% included (1987-Present)

# Standard Deduction Data (1970-2024)
# Source: Tax Policy Center, IRS
_STANDARD_DEDUCTIONS = {
    1970: {"Single": 1100, "Married Filing Jointly": 1100, "Head of Household": 1100},
    1971: {"Single": 1050, "Married Filing Jointly": 1050, "Head of Household": 1050},
    1972: {"Single": 1300, "Married Filing Jointly": 1300, "Head of Household": 1300},
    1973: {"Single": 1300, "Married Filing Jointly": 1300, "Head of Household": 1300},
    1974: {"Single": 1300, "Married Filing Jointly": 1300, "Head of Household": 1300},
    1975: {"Single": 1600, "Married Filing Jointly": 1900, "Head of Household": 1900},
    1976: {"Single": 1700, "Married Filing Jointly": 2100, "Head of Household": 2100},
    1977: {"Single": 2200, "Married Filing Jointly": 3200, "Head of Household": 3200},
    1978: {"Single": 2200, "Married Filing Jointly": 3200, "Head of Household": 3200},
    1979: {"Single": 2300, "Married Filing Jointly": 3400, "Head of Household": 3400},
    1980: {"Single": 2300, "Married Filing Jointly": 3400, "Head of Household": 3400},
    1981: {"Single": 2300, "Married Filing Jointly": 3400, "Head of Household": 3400},
    1982: {"Single": 2300, "Married Filing Jointly": 3400, "Head of Household": 3400},
    1983: {"Single": 2300, "Married Filing Jointly": 3400, "Head of Household": 3400},
    1984: {"Single": 2300, "Married Filing Jointly": 3400, "Head of Household": 3400},
    1985: {"Single": 2400, "Married Filing Jointly": 3550, "Head of Household": 3550},
    1986: {"Single": 2480, "Married Filing Jointly": 3670, "Head of Household": 3670},
    1987: {"Single": 2540, "Married Filing Jointly": 3760, "Head of Household": 3760},
    1988: {"Single": 3000, "Married Filing Jointly": 5000, "Head of Household": 4400},
    1989: {"Single": 3100, "Married Filing Jointly": 5200, "Head of Household": 4550},
    1990: {"Single": 3250, "Married Filing Jointly": 5450, "Head of Household": 4750},
    1991: {"Single": 3400, "Married Filing Jointly": 5700, "Head of Household": 5000},
    1992: {"Single": 3600, "Married Filing Jointly": 6000, "Head of Household": 5250},
    1993: {"Single": 3700, "Married Filing Jointly": 6200, "Head of Household": 5450},
    1994: {"Single": 3800, "Married Filing Jointly": 6350, "Head of Household": 5600},
    1995: {"Single": 3900, "Married Filing Jointly": 6550, "Head of Household": 5750},
    1996: {"Single": 4000, "Married Filing Jointly": 6700, "Head of Household": 5900},
    1997: {"Single": 4150, "Married Filing Jointly": 6900, "Head of Household": 6050},
    1998: {"Single": 4250, "Married Filing Jointly": 7100, "Head of Household": 6250},
    1999: {"Single": 4300, "Married Filing Jointly": 7200, "Head of Household": 6350},
    2000: {"Single": 4400, "Married Filing Jointly": 7350, "Head of Household": 6450},
    2001: {"Single": 4550, "Married Filing Jointly": 7600, "Head of Household": 6650},
    2002: {"Single": 4700, "Married Filing Jointly": 7850, "Head of Household": 6900},
    2003: {"Single": 4750, "Married Filing Jointly": 9500, "Head of Household": 7000},
    2004: {"Single": 4850, "Married Filing Jointly": 9700, "Head of Household": 7150},
    2005: {"Single": 5000, "Married Filing Jointly": 10000, "Head of Household": 7300},
    2006: {"Single": 5150, "Married Filing Jointly": 10300, "Head of Household": 7550},
    2007: {"Single": 5350, "Married Filing Jointly": 10700, "Head of Household": 7850},
    2008: {"Single": 5450, "Married Filing Jointly": 10900, "Head of Household": 8000},
    2009: {"Single": 5700, "Married Filing Jointly": 11400, "Head of Household": 8350},
    2010: {"Single": 5700, "Married Filing Jointly": 11400, "Head of Household": 8400},
    2011: {"Single": 5800, "Married Filing Jointly": 11600, "Head of Household": 8500},
    2012: {"Single": 5950, "Married Filing Jointly": 11900, "Head of Household": 8700},
    2013: {"Single": 6100, "Married Filing Jointly": 12200, "Head of Household": 8950},
    2014: {"Single": 6200, "Married Filing Jointly": 12400, "Head of Household": 9100},
    2015: {"Single": 6300, "Married Filing Jointly": 12600, "Head of Household": 9250},
    2016: {"Single": 6300, "Married Filing Jointly": 12600, "Head of Household": 9300},
    2017: {"Single": 6350, "Married Filing Jointly": 12700, "Head of Household": 9350},
    2018: {"Single": 12000, "Married Filing Jointly": 24000, "Head of Household": 18000},
    2019: {"Single": 12200, "Married Filing Jointly": 24400, "Head of Household": 18350},
    2020: {"Single": 12400, "Married Filing Jointly": 24800, "Head of Household": 18650},
    2021: {"Single": 12550, "Married Filing Jointly": 25100, "Head of Household": 18800},
    2022: {"Single": 12950, "Married Filing Jointly": 25900, "Head of Household": 19400},
    2023: {"Single": 13850, "Married Filing Jointly": 27700, "Head of Household": 20800},
    2024: {"Single": 14600, "Married Filing Jointly": 29200, "Head of Household": 21900},
    2025: {"Single": 15000, "Married Filing Jointly": 30000, "Head of Household": 23625}, # Projected/Estimated
}

def get_standard_deduction(year, filing_status, income=0):
    """
    Returns the standard deduction for the given year and filing status.
    Handles pre-1970 logic (10% of income capped at $1,000).
    """
    # Normalize filing status
    if filing_status not in ["Single", "Married Filing Jointly", "Head of Household"]:
        filing_status = "Single" # Fallback
        
    if year in _STANDARD_DEDUCTIONS:
        return _STANDARD_DEDUCTIONS[year].get(filing_status, 0)
    elif year < 1970:
        # Pre-1970 Rule: 10% of AGI, capped at $1,000
        # (Simplified, actual history is complex but this is a reasonable proxy)
        deduction = income * 0.10
        return min(deduction, 1000.0)
    else:
        # Future years: Use latest known
        latest_year = max(_STANDARD_DEDUCTIONS.keys())
        return _STANDARD_DEDUCTIONS[latest_year].get(filing_status, 0)

def calculate_tax_on_realized_gains(realized_gain=0.0, other_income=0.0, year=2024, filing_status="Single", method="2024_fixed", excel_path="Federal-Capital-Gains-Tax-Rates-Collections-1913-2025_fv.xlsx", short_term_gain=0.0, long_term_gain=0.0, long_term_gain_collectible=0.0, use_standard_deduction=True):
    """
    Calculates tax on realized gains using the specified method.
    
    Parameters:
    - realized_gain: Total realized gain (legacy support, treated as LT if st/lt not provided)
    - short_term_gain: Short-term capital gains (taxed as ordinary income)
    - long_term_gain: Long-term capital gains (preferential rates)
    - long_term_gain_collectible: Long-term gains on collectibles (taxed as ordinary, capped at 28%)
    - use_standard_deduction: If True, subtracts standard deduction from ordinary income.
    """
    # Legacy support: if st/lt not specified, assume all is realized_gain (treated as LT)
    if short_term_gain == 0 and long_term_gain == 0 and long_term_gain_collectible == 0:
        long_term_gain = realized_gain
        
    total_gain = short_term_gain + long_term_gain + long_term_gain_collectible
    if total_gain <= 0:
        return 0.0
        
    # --- Apply Standard Deduction ---
    std_deduction = 0.0
    if use_standard_deduction:
        # 1. Get Deduction Amount
        std_deduction = get_standard_deduction(year, filing_status, income=other_income)
    
    # 2. Apply to Ordinary Income (other_income) first
    effective_other_income = max(0, other_income - std_deduction)
    unused_deduction = max(0, std_deduction - other_income)
    
    # 3. Apply unused deduction to Short-Term Gains (Ordinary)
    effective_st_gain = max(0, short_term_gain - unused_deduction)
    unused_deduction = max(0, unused_deduction - short_term_gain)
    
    # 4. Apply unused deduction to Collectible Gains (28% Group)
    effective_collectible_gain = max(0, long_term_gain_collectible - unused_deduction)
    unused_deduction = max(0, unused_deduction - long_term_gain_collectible)
    
    # 5. Apply remaining unused deduction to Long-Term Gains (0/15/20% Group)
    effective_lt_gain = max(0, long_term_gain - unused_deduction)
    
    # --- Tax Calculation ---
    
    # 1. Tax on Short-Term Gains (Ordinary Income)
    # Stacked on top of effective_other_income
    st_tax = 0.0
    current_stack = effective_other_income
    
    if effective_st_gain > 0:
        base_tax = calculate_historical_tax(year, current_stack, filing_status, csv_path="Historical Income Tax Rates and Brackets, 1862-2025.csv")
        current_stack += effective_st_gain
        total_ordinary_tax = calculate_historical_tax(year, current_stack, filing_status, csv_path="Historical Income Tax Rates and Brackets, 1862-2025.csv")
        st_tax = total_ordinary_tax - base_tax
        
    # 2. Tax on Collectible Gains (Ordinary Rate, Capped at 28%)
    # Stacked on top of (Other Income + ST Gain)
    collectible_tax = 0.0
    if effective_collectible_gain > 0:
        base_tax = calculate_historical_tax(year, current_stack, filing_status, csv_path="Historical Income Tax Rates and Brackets, 1862-2025.csv")
        
        # Calculate tax as if it were ordinary
        temp_stack = current_stack + effective_collectible_gain
        full_ordinary_tax = calculate_historical_tax(year, temp_stack, filing_status, csv_path="Historical Income Tax Rates and Brackets, 1862-2025.csv")
        marginal_tax = full_ordinary_tax - base_tax
        
        # Calculate tax with 28% cap
        # We need to check if the marginal rate exceeds 28%.
        # Since calculate_historical_tax is progressive, we can't just multiply.
        # But we can compare the total tax amount.
        # Max tax is 28% of the gain.
        max_tax = effective_collectible_gain * 0.28
        
        # Wait, the rule is: Tax is calculated at ordinary rates, but the rate applied to this income cannot exceed 28%.
        # So if you are in 10%, 12%, 22%, 24% brackets, you pay that rate.
        # If you are in 32%, 35%, 37%, you pay 28%.
        # My logic `min(marginal_tax, max_tax)` is roughly correct for the aggregate, 
        # assuming the "marginal_tax" calculation reflects the progressive rates.
        # However, if the gain spans across the 24% and 32% brackets, 
        # the part in 24% is taxed at 24%, the part in 32% is taxed at 28%.
        # `min(total_marginal, total_max)` might under-tax if the average rate is < 28% but some part is > 28%?
        # No, `max_tax` is flat 28%. If average ordinary rate is < 28%, `marginal_tax` is lower.
        # If average ordinary rate is > 28%, `max_tax` is lower.
        # This simplification holds: You pay the lesser of (Ordinary Tax) or (28% Flat).
        # Actually, strictly speaking, you pay ordinary rates until they exceed 28%.
        # So `min` works.
        
        collectible_tax = min(marginal_tax, max_tax)
        current_stack += effective_collectible_gain

    # 3. Tax on Long-Term Gains (Preferential Rates)
    # Stacked on top of (Other + ST + Collectible)
    lt_tax = 0.0
    stacking_income = current_stack
    
    if effective_lt_gain > 0:
        long_term_gain = effective_lt_gain
        if method == "historical_max_rate":
            load_capital_gains_rates(excel_path)
            # Fallback to closest year
            if year not in _CAP_GAINS_RATES:
                 if _CAP_GAINS_RATES:
                     year = max(_CAP_GAINS_RATES.keys())
                 else:
                     return 0.0
            rate = _CAP_GAINS_RATES.get(year, 0.0)
            lt_tax = long_term_gain * rate
            
        elif method == "historical_smart":
            # For modern years (2024+), use the specific fixed brackets
            if year >= 2024:
                lt_tax = calculate_federal_tax(long_term_gain, stacking_income, filing_status, year=year)
            else:
                # Historical Logic (Pre-2024)
                # 1. Calculate Regular Tax with Inclusion Rate
                inclusion_rate = get_capital_gains_inclusion_rate(year)
                taxable_lt_gain = long_term_gain * inclusion_rate
                
                # Marginal tax on the included gain
                # Stacked on top of stacking_income
                total_income_regular = stacking_income + taxable_lt_gain
                tax_total_regular = calculate_historical_tax(year, total_income_regular, filing_status)
                tax_base_regular = calculate_historical_tax(year, stacking_income, filing_status)
                regular_tax_liability = max(0, tax_total_regular - tax_base_regular)
                
                # 2. Calculate Alternative Tax (Max Rate Cap)
                load_capital_gains_rates(excel_path)
                max_rate = _CAP_GAINS_RATES.get(year, 0.35) 
                alternative_tax_liability = long_term_gain * max_rate
                
                lt_tax = min(regular_tax_liability, alternative_tax_liability)
    
        else:
            # Default: 2024 Fixed Brackets
            lt_tax = calculate_federal_tax(long_term_gain, stacking_income, filing_status)

    # 3. NIIT (2013-Present)
    # NIIT applies on top of everything else for high earners
    niit = 0.0
    if year >= 2013:
        niit_threshold = 250000 if filing_status == "Married Filing Jointly" else 200000
        magi = other_income + short_term_gain + long_term_gain
        if magi > niit_threshold:
            excess = magi - niit_threshold
            # NIIT applies to the lesser of (Net Investment Income) or (Excess MAGI)
            # NII = ST Gain + LT Gain
            investment_income = short_term_gain + long_term_gain
            subject_to_niit = min(investment_income, excess)
            niit = subject_to_niit * 0.038
            
    return st_tax + lt_tax + niit

def calculate_federal_tax(realized_gain, other_income, filing_status="Single", year=2025):
    """
    Calculates estimated federal tax on long-term capital gains using specific year brackets.
    Includes Net Investment Income Tax (NIIT).
    """
    if realized_gain <= 0:
        return 0.0
        
    # Define Brackets
    # Format: (Threshold, Rate)
    # 0% up to Threshold 1
    # 15% up to Threshold 2
    # 20% above Threshold 2
    
    brackets = []
    niit_threshold = 200000
    
    if year == 2024:
        if filing_status == "Married Filing Jointly":
            brackets = [
                (94050, 0.00),
                (583750, 0.15),
                (float("inf"), 0.20)
            ]
            niit_threshold = 250000
        elif filing_status == "Head of Household":
            brackets = [
                (63000, 0.00),
                (551350, 0.15),
                (float("inf"), 0.20)
            ]
            niit_threshold = 200000
        else: # Single
            brackets = [
                (47025, 0.00),
                (518900, 0.15),
                (float("inf"), 0.20)
            ]
            niit_threshold = 200000
            
    else: # Default to 2025 (and future)
        if filing_status == "Married Filing Jointly":
            brackets = [
                (98900, 0.00),
                (613700, 0.15),
                (float("inf"), 0.20)
            ]
            niit_threshold = 250000
        elif filing_status == "Head of Household":
            brackets = [
                (64750, 0.00),
                (566700, 0.15),
                (float("inf"), 0.20)
            ]
            niit_threshold = 200000
        else: # Single
            brackets = [
                (49450, 0.00),
                (545500, 0.15),
                (float("inf"), 0.20)
            ]
            niit_threshold = 200000
        
    # Calculate Capital Gains Tax
    # The capital gains "stack" on top of other income.
    # So we start filling brackets from `other_income`.
    
    current_income_level = other_income
    gains_remaining = realized_gain
    total_tax = 0.0
    
    # 0% Bracket
    limit_0 = brackets[0][0]
    if current_income_level < limit_0:
        room = limit_0 - current_income_level
        taxable_at_0 = min(gains_remaining, room)
        # Tax is 0
        gains_remaining -= taxable_at_0
        current_income_level += taxable_at_0
        
    if gains_remaining <= 0:
        return total_tax
        
    # 15% Bracket
    limit_15 = brackets[1][0]
    if current_income_level < limit_15:
        room = limit_15 - current_income_level
        taxable_at_15 = min(gains_remaining, room)
        total_tax += taxable_at_15 * 0.15
        gains_remaining -= taxable_at_15
        current_income_level += taxable_at_15
        
    if gains_remaining <= 0:
        pass
    else:
        # 20% Bracket (Remaining)
        total_tax += gains_remaining * 0.20
        current_income_level += gains_remaining
        
    # Net Investment Income Tax (NIIT)
    magi = other_income + realized_gain
    if magi > niit_threshold:
        excess = magi - niit_threshold
        subject_to_niit = min(realized_gain, excess)
        total_tax += subject_to_niit * 0.038
        
    return total_tax

def calculate_tax_series_with_carryforward(pl_series, other_income, filing_status="Single", method="2024_fixed", excel_path="Federal-Capital-Gains-Tax-Rates-Collections-1913-2025_fv.xlsx", use_standard_deduction=True):
    """
    Calculates tax for a series of P&L (indexed by Year), handling loss carryforwards.
    Accepts either a Series (Total P&L) or a DataFrame (with 'Realized ST P&L' and 'Realized LT P&L').
    Returns a Series of Tax Owed.
    """
    tax_owed_series = pd.Series(index=pl_series.index, dtype=float)
    
    # Carryforwards
    st_loss_carryforward = 0.0
    lt_loss_carryforward = 0.0
    
    # Ensure sorted by year
    sorted_pl = pl_series.sort_index()
    
    for year, row in sorted_pl.iterrows() if isinstance(pl_series, pd.DataFrame) else sorted_pl.items():
        
        # Extract ST and LT gains/losses
        if isinstance(pl_series, pd.DataFrame):
            st_pl = row.get("Realized ST P&L", 0.0)
            lt_pl = row.get("Realized LT P&L", 0.0) # Standard LT
            lt_col_pl = row.get("Realized LT (Collectible)", 0.0) # Collectible LT
            
            # If columns missing, assume all LT (legacy)
            if "Realized ST P&L" not in row and "Realized LT P&L" not in row:
                 pass 
        else:
            # Series input: Assume all Long-Term (Legacy behavior)
            st_pl = 0.0
            lt_pl = row
            lt_col_pl = 0.0
            
        # --- Netting Rules ---
        
        # 1. Net Current Year ST
        net_st = st_pl - st_loss_carryforward
        if net_st < 0:
            st_loss_carryforward = abs(net_st)
            net_st = 0.0
        else:
            st_loss_carryforward = 0.0
            
        # 2. Net Current Year LT (Buckets)
        # Apply LT Loss Carryforward to Collectibles (28%) FIRST (Taxpayer favorable)
        net_lt_col = lt_col_pl
        if lt_loss_carryforward > 0:
            if net_lt_col > 0:
                used = min(net_lt_col, lt_loss_carryforward)
                net_lt_col -= used
                lt_loss_carryforward -= used
        
        # Apply remaining LT Loss Carryforward to Standard LT (15%/20%)
        net_lt_std = lt_pl
        if lt_loss_carryforward > 0:
            # Note: net_lt_std can be negative (current year loss). 
            # If negative, we just add to the loss? No, carryforward is separate.
            # We net CF against GAINS.
            if net_lt_std > 0:
                used = min(net_lt_std, lt_loss_carryforward)
                net_lt_std -= used
                lt_loss_carryforward -= used
                
        # 3. Internal LT Netting (Current Year + Remaining CF)
        # We now have net_lt_col and net_lt_std. One or both could be negative (current year loss).
        # Also we might still have lt_loss_carryforward if it wiped out all gains.
        
        # If we have remaining CF, it means both net_lt_col and net_lt_std are <= 0 (or fully offset).
        # So we just treat them as losses to be added to CF?
        # Let's simplify: Combine all LT components to find Net LT Position.
        # But we need to preserve character if positive.
        
        final_lt_col = 0.0
        final_lt_std = 0.0
        net_lt_loss = 0.0
        
        # Case A: Both Positive (or 0)
        if net_lt_col >= 0 and net_lt_std >= 0:
            final_lt_col = net_lt_col
            final_lt_std = net_lt_std
            # lt_loss_carryforward remains if any
            
        # Case B: Col Negative, Std Positive
        elif net_lt_col < 0 and net_lt_std >= 0:
            # Offset Std Gain with Col Loss
            combined = net_lt_std + net_lt_col
            if combined >= 0:
                final_lt_std = combined
                final_lt_col = 0.0
            else:
                final_lt_std = 0.0
                final_lt_col = 0.0
                net_lt_loss = abs(combined)
                
        # Case C: Std Negative, Col Positive
        elif net_lt_std < 0 and net_lt_col >= 0:
            # Offset Col Gain with Std Loss
            combined = net_lt_col + net_lt_std
            if combined >= 0:
                final_lt_col = combined
                final_lt_std = 0.0
            else:
                final_lt_col = 0.0
                final_lt_std = 0.0
                net_lt_loss = abs(combined)
                
        # Case D: Both Negative
        else:
            net_lt_loss = abs(net_lt_col) + abs(net_lt_std)
            
        # Add any remaining carryforward to the net loss
        net_lt_loss += lt_loss_carryforward
        lt_loss_carryforward = 0.0 # Reset, will be rebuilt from net_lt_loss
        
        # 4. Cross-Netting (ST vs LT)
        final_st = net_st # Initialize final_st
        
        # If Net ST is Loss
        if st_loss_carryforward > 0: # This means net_st was < 0
            # Offset LT Gains (High Tax First? No, usually 28% then 15%)
            # Offset Collectible First
            if final_lt_col > 0:
                used = min(final_lt_col, st_loss_carryforward)
                final_lt_col -= used
                st_loss_carryforward -= used
                
            # Offset Standard Next
            if final_lt_std > 0 and st_loss_carryforward > 0:
                used = min(final_lt_std, st_loss_carryforward)
                final_lt_std -= used
                st_loss_carryforward -= used
                
        # If Net LT is Loss
        if net_lt_loss > 0:
            # Offset ST Gain
            if final_st > 0:
                used = min(final_st, net_lt_loss)
                final_st -= used
                net_lt_loss -= used
                
            # Remaining becomes LT Carryforward
            lt_loss_carryforward = net_lt_loss
            
        # 5. Calculate Tax
        
        # Check for Net Capital Loss to deduct from Ordinary Income (Max $3,000)
        deduction_amount = 0.0
        tax_savings = 0.0
        
        # If we have remaining losses after cross-netting
        if st_loss_carryforward > 0 or lt_loss_carryforward > 0:
            total_loss = st_loss_carryforward + lt_loss_carryforward
            deduction_amount = min(total_loss, 3000.0)
            
            # Reduce carryforwards by the used deduction
            # ST losses are used first, then LT
            remaining_deduction = deduction_amount
            
            if st_loss_carryforward > 0:
                used_st = min(st_loss_carryforward, remaining_deduction)
                st_loss_carryforward -= used_st
                remaining_deduction -= used_st
                
            if remaining_deduction > 0 and lt_loss_carryforward > 0:
                used_lt = min(lt_loss_carryforward, remaining_deduction)
                lt_loss_carryforward -= used_lt
                remaining_deduction -= used_lt
            
            # Calculate Tax Savings from this deduction
            if deduction_amount > 0:
                std_ded = 0.0
                if use_standard_deduction:
                    std_ded = get_standard_deduction(year, filing_status, income=other_income)
                
                base_taxable = max(0, other_income - std_ded)
                reduced_taxable = max(0, other_income - deduction_amount - std_ded)
                
                base_ordinary_tax = calculate_historical_tax(year, base_taxable, filing_status)
                reduced_ordinary_tax = calculate_historical_tax(year, reduced_taxable, filing_status)
                tax_savings = base_ordinary_tax - reduced_ordinary_tax
        
        tax = calculate_tax_on_realized_gains(
            realized_gain=0, # Not used when st/lt provided
            other_income=max(0, other_income - deduction_amount), 
            year=year, 
            filing_status=filing_status, 
            method=method, 
            excel_path=excel_path,
            short_term_gain=final_st,
            long_term_gain=final_lt_std,
            long_term_gain_collectible=final_lt_col,
            use_standard_deduction=use_standard_deduction
        )
        
        # Apply tax savings (refund)
        tax_owed_series[year] = tax - tax_savings
            
    return tax_owed_series

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

def calculate_tax_on_realized_gains(realized_gain, other_income, year, filing_status="Single", method="2024_fixed", excel_path="Federal-Capital-Gains-Tax-Rates-Collections-1913-2025_fv.xlsx"):
    """
    Calculates tax on realized gains using the specified method.
    
    Methods:
    - "2024_fixed": Uses 2024 Long-Term Capital Gains brackets (0%, 15%, 20% + NIIT).
    - "historical_max_rate": Uses the flat maximum rate from the Excel file.
    - "historical_smart": Calculates tax using historical inclusion rates and ordinary brackets, 
                          capped by the historical maximum rate (Alternative Tax).
    """
    if realized_gain <= 0:
        return 0.0
        
    if method == "historical_max_rate":
        load_capital_gains_rates(excel_path)
        # Fallback to closest year
        if year not in _CAP_GAINS_RATES:
             if _CAP_GAINS_RATES:
                 year = max(_CAP_GAINS_RATES.keys())
             else:
                 return 0.0
        rate = _CAP_GAINS_RATES.get(year, 0.0)
        return realized_gain * rate
        
    elif method == "historical_smart":
        # 1. Calculate Regular Tax with Inclusion Rate
        inclusion_rate = get_capital_gains_inclusion_rate(year)
        taxable_gain = realized_gain * inclusion_rate
        
        # Marginal tax on the included gain
        total_income_regular = other_income + taxable_gain
        tax_total_regular = calculate_historical_tax(year, total_income_regular, filing_status)
        tax_base_regular = calculate_historical_tax(year, other_income, filing_status)
        regular_tax_liability = max(0, tax_total_regular - tax_base_regular)
        
        # 2. Calculate Alternative Tax (Max Rate Cap)
        # This acts as a ceiling. If the regular calculation exceeds the max cap rate, pay the max cap.
        # Note: The Excel file contains the "Maximum Effective Rate".
        load_capital_gains_rates(excel_path)
        max_rate = _CAP_GAINS_RATES.get(year, 0.35) # Default to 35% if missing, though load should handle it
        alternative_tax_liability = realized_gain * max_rate
        
        # 3. NIIT (2013-Present)
        # NIIT applies on top of everything else for high earners
        niit = 0.0
        if year >= 2013:
            niit_threshold = 250000 if filing_status == "Married Filing Jointly" else 200000
            magi = other_income + realized_gain # NIIT uses full gain
            if magi > niit_threshold:
                excess = magi - niit_threshold
                subject_to_niit = min(realized_gain, excess)
                niit = subject_to_niit * 0.038
        
        # The tax is the lesser of Regular (with exclusion) or Alternative (Max Rate), plus NIIT
        # Note: For 1988-1990, there was no exclusion and no special max rate (taxed as ordinary), 
        # so regular_tax_liability will naturally be the result.
        # For 2003+, the "Max Rate" in Excel (15%) is the preferential rate. 
        # Our "Regular Tax" calculation (100% inclusion) would be high, so "Alternative" (15%) wins.
        
        return min(regular_tax_liability, alternative_tax_liability) + niit

    else:
        # Default: 2024 Fixed Brackets
        return calculate_federal_tax(realized_gain, other_income, filing_status)

def calculate_federal_tax(realized_gain, other_income, filing_status="Single"):
    """
    Calculates estimated federal tax on long-term capital gains using 2024 brackets.
    Includes Net Investment Income Tax (NIIT).
    """
    if realized_gain <= 0:
        return 0.0
        
    # 2024 Long-Term Capital Gains Brackets
    # Format: (Threshold, Rate)
    # 0% up to Threshold 1
    # 15% up to Threshold 2
    # 20% above Threshold 2
    
    if filing_status == "Married Filing Jointly":
        brackets = [
            (94050, 0.00),
            (583750, 0.15),
            (float("inf"), 0.20)
        ]
        niit_threshold = 250000
    else: # Single (and others mapped to Single for simplicity)
        brackets = [
            (47025, 0.00),
            (518900, 0.15),
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

def calculate_tax_series_with_carryforward(pl_series, other_income, filing_status="Single", method="2024_fixed", excel_path="Federal-Capital-Gains-Tax-Rates-Collections-1913-2025_fv.xlsx"):
    """
    Calculates tax for a series of P&L (indexed by Year), handling loss carryforwards.
    Returns a Series of Tax Owed.
    """
    tax_owed_series = pd.Series(index=pl_series.index, dtype=float)
    loss_carryforward = 0.0
    
    # Ensure sorted by year
    sorted_pl = pl_series.sort_index()
    
    for year, realized_pl in sorted_pl.items():
        # Net against carryforward
        net_pl = realized_pl - loss_carryforward
        
        if net_pl > 0:
            # We have a taxable gain after using up carryforward
            taxable_gain = net_pl
            loss_carryforward = 0.0 # Used up
            
            # Calculate Tax
            tax = calculate_tax_on_realized_gains(
                taxable_gain, 
                other_income, 
                year, 
                filing_status, 
                method=method, 
                excel_path=excel_path
            )
            tax_owed_series[year] = tax
            
        else:
            # We have a loss (or 0) after netting
            # net_pl is negative or zero. This becomes the new carryforward.
            loss_carryforward = abs(net_pl)
            tax_owed_series[year] = 0.0
            
    return tax_owed_series

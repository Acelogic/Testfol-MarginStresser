
import pandas as pd
import numpy as np

def calculate_cagr(start_val, end_val, years):
    if years <= 0: return 0.0
    return ((end_val / start_val) ** (1 / years) - 1) * 100

# Parameters
start_val = 10000
growth_rate = 0.10
years = 5
annual_contribution = 1000

# Scenario 1: No Cashflow
# End Value = Start * (1+r)^n
end_val_no_cf = start_val * ((1 + growth_rate) ** years)
cagr_no_cf = calculate_cagr(start_val, end_val_no_cf, years)

# Scenario 2: With Cashflow (Buying Assets)
# End Value = Start*(1+r)^n + Contrib*(1+r)^(n-1) + ...
future_val_contribs = 0
for i in range(1, years + 1):
    # Contribution at end of year i, grows for (years - i) years? 
    # Let's assume contribution at START of year for simplicity or END.
    # Usually DCA is monthly. Let's say End of Year.
    # Contrib at end of year 1 grows for 4 years.
    future_val_contribs += annual_contribution * ((1 + growth_rate) ** (years - i))

end_val_with_cf = end_val_no_cf + future_val_contribs
cagr_with_cf = calculate_cagr(start_val, end_val_with_cf, years)

print(f"Scenario 1 (No Cashflow): Start=${start_val}, End=${end_val_no_cf:.2f}, CAGR={cagr_no_cf:.2f}%")
print(f"Scenario 2 (With Cashflow): Start=${start_val}, End=${end_val_with_cf:.2f}, CAGR={cagr_with_cf:.2f}%")

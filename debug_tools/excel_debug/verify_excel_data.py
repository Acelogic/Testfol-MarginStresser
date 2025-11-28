import pandas as pd

try:
    file_path = "../Federal-Capital-Gains-Tax-Rates-Collections-1913-2025_fv.xlsx"
    sheet_name = "Sheet1"
    
    # Read without header
    df = pd.read_excel(file_path, sheet_name=sheet_name, header=None)
    
    # Rename columns for clarity based on hypothesis
    # 0: Year, 5: Rate
    df_subset = df[[0, 5]].copy()
    df_subset.columns = ["Year", "Rate"]
    
    # Filter for specific years
    years_to_check = [1913, 1980, 2000, 2024, 2025]
    
    print("Checking specific years:")
    for year in years_to_check:
        row = df_subset[df_subset["Year"] == year]
        if not row.empty:
            print(row.to_string(index=False))
        else:
            print(f"Year {year} not found")
            
except Exception as e:
    print(f"Error: {e}")

import pandas as pd

try:
    file_path = "../Federal-Capital-Gains-Tax-Rates-Collections-1913-2025_fv.xlsx"
    sheet_name = "Sheet1"
    
    # Read first 20 rows without header
    df = pd.read_excel(file_path, sheet_name=sheet_name, header=None, nrows=20)
    print(df.to_string())
    
except Exception as e:
    print(f"Error: {e}")

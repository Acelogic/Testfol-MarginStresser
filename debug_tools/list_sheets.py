import pandas as pd

try:
    file_path = "../Federal-Capital-Gains-Tax-Rates-Collections-1913-2025_fv.xlsx"
    xl = pd.ExcelFile(file_path)
    print("Sheet names:", xl.sheet_names)
    
except Exception as e:
    print(f"Error: {e}")

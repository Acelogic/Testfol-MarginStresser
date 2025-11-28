import pandas as pd
import sys

try:
    file_path = "../Federal-Capital-Gains-Tax-Rates-Collections-1913-2025_fv.xlsx"
    xl = pd.ExcelFile(file_path)
    print("Sheet names:", xl.sheet_names)
    
    for sheet in xl.sheet_names:
        print(f"\n--- Sheet: {sheet} ---")
        df = pd.read_excel(file_path, sheet_name=sheet, nrows=10)
        print(df.to_string())
        print("-" * 20)
        
except Exception as e:
    print(f"Error: {e}")

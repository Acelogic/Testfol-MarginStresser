import pandas as pd

try:
    file_path = "../Federal-Capital-Gains-Tax-Rates-Collections-1913-2025_fv.xlsx"
    sheet_name = "Sheet1"
    
    # Read with header=6 (row 7)
    df = pd.read_excel(file_path, sheet_name=sheet_name, header=6, nrows=5)
    print("Columns:", df.columns.tolist())
    print(df.to_string())
    
except Exception as e:
    print(f"Error: {e}")

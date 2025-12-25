import pandas as pd

df = pd.read_csv("output/ndx_mega2_constituents.csv")
df['Date'] = pd.to_datetime(df['Date'])

# Get Dec 2024 row
row = df[df['Date'] == '2024-12-31'].iloc[0]

print(f"Date: {row['Date']}")
print(f"Type: {row['Type']}")
print(f"Count: {row['Count']}")

tickers = row['Tickers'].split('|')
weights = [float(w) for w in row['Weights'].split('|')]

wd = pd.DataFrame({'Ticker': tickers, 'Weight': weights}).sort_values('Weight', ascending=False)
print(wd)

print(f"Total Weight: {wd['Weight'].sum()}")

# Check Fillers vs Standards
# Assuming fill-to-9 logic, the top ones are Standards.
# If count is 9, usually top 6-7 are standards, bottom 2-3 are fillers.
# Let's see the weight distribution.

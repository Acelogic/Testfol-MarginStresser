# FAQ & Troubleshooting

## Common Issues

### "Weights must sum to 100%"
**Cause**: The total percentage in your allocation table does not equal exactly 100.00%.
**Fix**: Adjust the "Weight %" column values. Ensure there are no empty rows with 0% weight that might be confusing the validator.

### API Request Failed
**Cause**: The application could not connect to the Testfol API.
**Fix**:
1. Check your internet connection.
2. Verify that the ticker symbols are correct and valid on Testfol.io.
3. Try a shorter date range or fewer tickers to reduce request size.
4. Wait a few moments and try again (the API might be temporarily busy).

### Charts Not Displaying
**Cause**: The backtest didn't run or returned empty data.
**Fix**:
1. Ensure you clicked the **"Run back-test"** button.
2. Check if the date range is valid for the selected tickers (e.g., requesting data for a stock before it IPO'd).

### "NameError: name 'px' is not defined"
**Cause**: This was a known bug in older versions where a library import was missing.
**Fix**: This has been resolved in the current version. If you see this, please report it as a regression.

## Frequently Asked Questions

### Can I simulate short selling?
No, the current version only supports long positions with margin leverage. Short selling involves different margin mechanics not yet implemented.

### How accurate is the tax simulation?
It is an *estimation*. While it uses historical tax brackets and sophisticated lot tracking (HIFO/LIFO), it cannot account for your specific personal financial situation, deductions, or complex tax events like wash sales perfectly. **Do not use this for actual tax filing.**

### Why is my "Net Equity" lower than I expected?
If you enabled "Pay from Cash" for taxes, the simulation withdraws money from your portfolio to pay taxes annually. This reduces the compounding effect significantly over long periods compared to a tax-deferred or tax-free account.

### What does "Maintenance %" mean?
It is the minimum amount of equity (as a percentage of total market value) you must hold.
- **25%**: Standard Reg T requirement for most stocks.
- **30-50%**: Often required for volatile stocks or concentrated positions.
- **100%**: Cash account (no margin allowed).
If your equity drops below this percentage, a margin call occurs.

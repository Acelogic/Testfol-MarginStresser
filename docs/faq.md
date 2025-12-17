# FAQ & Troubleshooting

## Common Issues

### "Why is my Start Date capped?"
**Cause**: The simulation start date is strictly limited by the **earliest common data availability** for the assets in your portfolio.
- **Explanation**: The local tax engine requires real daily price data to track tax lots. It cannot use synthetic data for this purpose. Use the inception date of your *youngest* asset as a guide.
- **Example**: If your portfolio contains an ETF that launched in 2002 (e.g., TLT), the entire simulation cannot start before that date.

### "Weights must sum to 100%"
**Cause**: The total percentage in your allocation table does not equal exactly 100.00%.
**Fix**: Adjust the "Weight %" column. Ensure no hidden rows have residual values.

### API Request Failed
**Cause**: Connection issue or invalid ticker.
**Fix**: Verify your internet connection and ticker symbols. Testfol.io must be reachable.

---

## Technical Questions

### How accurate is the tax simulation?
**Very High Fidelity (Mechanically)**.
-   The app tracks every single share purchase and sale (FIFO Tax Lots).
-   It distinguishes Short-Term vs. Long-Term gains daily.
-   It applies historical tax brackets appropriate for each year.

**However**, it is **NOT** a replacement for a CPA. It does not account for:
-   Wash Sale rules (30-day window).
-   Your personal deductions or credits.
-   State-specific tax brackets (it uses a flat state rate).

### Can I simulate Short Selling?
**No**. The simulator currently supports "Long Only" strategies with margin leverage. Short selling involves borrowing shares (not cash) and has different margin maintenance rules not yet modeled here.

### Why is there a difference between "Pay with Margin" and "Pay from Cash"?
-   **Pay with Margin**: You keep your assets compounding but accumulate a compounding debt liability. Good in bull markets.
-   **Pay from Cash**: You stay debt-free (regarding taxes) but interrupt the compound growth of your assets by selling them. Often safer but lower total return.

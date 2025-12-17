# Methodology & Technical Details

This document outlines the technical implementation details of the Testfol Margin Stresser, including margin calculations, detailed tax logic, and data sources.

## Data Source
The application is powered by **[testfol.io](https://testfol.io)**.
- **Historical Data**: Price data is sourced from Testfol's extensive database, covering stocks and ETFs back to 1885.
- **Backtesting Engine**: The core portfolio performance (total return, rebalancing) is calculated by the Testfol API.

## Margin Simulation Logic
The margin simulation is applied *on top* of the unleveraged portfolio performance returned by the API.

### 1. Daily Interest Calculation
Interest is compounded daily based on the annual interest rate provided.

$$ \text{Daily Rate} = (1 + \frac{\text{Annual Rate}}{100})^{1/365.25} - 1 $$
$$ \text{Loan}_{t} = \text{Loan}_{t-1} \times (1 + \text{Daily Rate}) + \text{Cashflows} $$

*(Note: The simulation uses a geometric daily rate conversion to be precise).*

### 2. Monthly Adjustments
If a "Monthly Margin Draw" is configured, it is added to the loan balance on the 1st of each month.

### 3. Margin Call Detection
A margin call is triggered when the **Margin Usage** reaches or exceeds 100%.

$$ \text{Usage \%} = \frac{\text{Loan}}{\text{Portfolio Value} \times (1 - \text{Maintenance \%})} $$

---

## Shadow Backtest Engine (Tax Lots)

To calculate taxes accurately, the application runs a local "Shadow Backtest" that reconstructs the portfolio using a **Tax Lot System**.

### FIFO Accounting
The simulation strictly uses **First-In, First-Out (FIFO)** accounting.
- **Sell Logic**: When shares are sold (for rebalancing, paying taxes, or draws), the oldest shares are sold first.
- **Tracking**: Each purchase creates a `TaxLot` with a specific date and cost basis.

### Proportional Cost Basis Recovery
When a `TaxLot` is partially sold, the Cost Basis is recovered proportionally. This ensures that you only pay tax on the *gain* portion of the specific shares sold, effectively "returning principal" tax-free.

**Formula:**
$$ \text{Fraction Sold} = \frac{\text{Shares to Sell}}{\text{Total Lot Quantity}} $$
$$ \text{Basis Used} = \text{Total Lot Cost Basis} \times \text{Fraction Sold} $$
$$ \text{Realized Gain} = \text{Proceeds} - \text{Basis Used} $$

*Example*:
- You bought $20k worth of stock (100 shares).
- It grows to $30k (now $300/share).
- You sell $15k (50 shares).
- **Fraction**: 50%, so **Basis Used**: $10k.
- **Gain**: $15k - $10k = $5k.

---

## Tax Simulation Methods

The application includes a sophisticated tax simulation engine (`tax_library.py`) that estimates Federal and State taxes.

### Method 1: Historical Smart Calculation (Default)
This method reconstructs the actual historical tax burden by simulating two eras:

1.  **Deduction/Exclusion Era (Pre-1987)**:
    -   Calculates the **Inclusion Rate** (e.g., 50% of nominal gain).
    -   Apply Ordinary Income Tax brackets to the included portion.
    -   **Alternative Tax Cap**: Checks if the gain exceeds the historical "Alternative Tax" cap (e.g., 25%). You pay the lower amount.

2.  **Modern Bracket Era (1987–Present)**:
    -   Applies 0%, 15%, 20% preferential brackets based on the year's specific inflation-adjusted thresholds.
    -   Adds **Net Investment Income Tax (NIIT)** of 3.8% if applicable (post-2013).

### Method 2: Historical Max Rate
Simplified. Applies the top marginal capital gains tax rate for that specific year to all long-term gains.

### Method 3: 2025 Fixed
Applies current (2025) tax brackets to all historical years. Useful for "anachronistic" comparisons.

## Limitations
- **State Tax**: Applied as a flat rate on net gains.
- **Carryforward**: Capital losses are carried forward indefinitely to offset future gains.
- **Cash Drag**: The API backtest may assume perfect execution; the shadow backtest validates this using daily data.

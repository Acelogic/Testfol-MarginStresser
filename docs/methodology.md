# Methodology & Technical Details

This document outlines the technical implementation details of the Testfol Margin Stresser, including margin calculations, tax logic, and data sources.

## Data Source
The application is powered by **[testfol.io](https://testfol.io)**.
- **Historical Data**: Price data is sourced from Testfol's extensive database, covering stocks and ETFs back to 1885.
- **Backtesting Engine**: The core portfolio performance (total return, rebalancing) is calculated by the Testfol API.

## Margin Simulation Logic

The margin simulation is applied *on top* of the unleveraged portfolio performance returned by the API.

### 1. Daily Interest Calculation
Interest is compounded daily based on the annual interest rate provided.

$$ \text{Daily Rate} = \frac{\text{Annual Rate}}{100} / 252 $$
$$ \text{Daily Loan}_{t} = \text{Loan}_{t-1} \times (1 + \text{Daily Rate}) $$

### 2. Monthly Adjustments
If a "Monthly Margin Draw" is configured, it is added to the loan balance on the 1st of each month.

$$ \text{Loan}_{t} = \text{Loan}_{t} + \text{Monthly Draw} $$

### 3. Margin Call Detection
A margin call is triggered when the **Margin Usage** reaches or exceeds 100%.

$$ \text{Usage \%} = \frac{\text{Loan}}{\text{Portfolio Value} \times (1 - \text{Maintenance \%})} $$

- **Portfolio Value**: Total market value of assets.
- **Maintenance \%**: Weighted average maintenance requirement of the portfolio.

## Tax Simulation

The application includes a sophisticated tax simulation engine that estimates Federal and State taxes on realized gains.

### Tax Calculation Methods
- **HIFO (Highest In, First Out)**: Sells the most expensive shares first to minimize taxable gains.
- **LIFO (Last In, First Out)**: Sells the most recently acquired shares first.
- **FIFO (First In, First Out)**: Sells the oldest shares first (standard default).

### Payment Methods
You can choose how estimated taxes are paid:
1. **Pay with Margin**: The tax amount is added to your margin loan. This preserves your asset base but increases your debt and interest costs.
2. **Pay from Cash**: The tax amount is withdrawn from the portfolio (simulating asset sales). This reduces your compounding base but keeps debt lower.

### Assumptions
- **Tax Rates**: Based on historical US Federal Capital Gains tax rates and brackets (1913-Present).
- **State Tax**: A flat rate can be applied (default 0%).
- **Carryforward**: Capital losses are carried forward to offset future gains.
- **Payment Timing**: Taxes for a given year are assumed to be paid on April 15th of the *following* year.

## Limitations
- **Transaction Costs**: Trading commissions and bid-ask spreads are not included.
- **Slippage**: Assumes trades are executed exactly at the closing price.
- **Margin Rates**: The simulation uses a constant interest rate, whereas real-world margin rates fluctuate.
- **Regulatory Changes**: Historical margin requirements (Reg T) have changed over time; this tool applies the user-defined maintenance requirement constantly.

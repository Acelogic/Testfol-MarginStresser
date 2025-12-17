# User Guide

Welcome to the Testfol Margin Stresser User Guide. This application allows you to simulate leveraged portfolio performance over historical periods, tracking margin debt, equity levels, and potential margin calls.

## Getting Started

### 1. Configure Global Settings
The sidebar contains all the configuration options for your backtest.

**Date Range**
- **Start/End Date**: Select the simulation period.
- **Range**: You can backtest as far back as **1884** (ticker data permitting).

**Portfolios**
- **Save/Load**: Use the "Saved Portfolios" section to persist your favorite strategies.
- **Allocation**: Enter your portfolio weights in the table. Must sum to 100%.

### 2. Strategy Configuration Tabs

#### 💼 Portfolio
- **Start Value**: Initial capital (e.g., $10,000).
- **Cashflow**: Periodic contributions (positive) or withdrawals (negative).
- **Tax Simulation**:
    -   **Filing Status**: Affects tax brackets.
    -   **Other Income**: Used to determine your base tax bracket.
    -   **Calculation Method**:
        -   *Historical Smart (Default)*: Uses actual historical inclusion rates and brackets.
        -   *Historical Max*: Applies the top capital gains rate for that year.
        -   *2025 Fixed*: Applies modern brackets to all years.

#### 🏦 Margin & Financing
- **Tax Payment Mode**: 
    -   *Pay from Cash*: Sell assets to pay taxes (reduces compounding).
    -   *Pay with Margin*: Borrow to pay taxes (increases leverage).
- **Loan Config**: Set starting loan or initial equity %.
- **Rates**: Annual interest rate and default maintenance requirement.

#### ⚙️ Settings
- **Chart Style**: Switch between "Classic", "Dashboard", and "Candlestick".
- **Log Scale**: Toggle logarithmic axis for long-term growth charts.

---

## Ticker Modifiers
You can use Testfol modifiers in the ticker symbol to simulate leverage or expense ratios:

| Modifier | Description | Example |
| :--- | :--- | :--- |
| `?L=X` | **Leverage**: Multiplies daily returns by X. | `SPY?L=2` (2x S&P 500) |
| `?D=X` | **Drag**: Applies annual drag (expense ratio). | `SPY?D=0.50` (0.50% fee) |
| `_SIM` | **Simulation**: Often used to extend history. | `UPRO_SIM` (Simulated 3x SPY) |

---

## Understanding the Results

### Summary Metrics
- **CAGR**: Compound Annual Growth Rate.
- **Sharpe Ratio**: Risk-adjusted return (using risk-free rate = 0).
- **Max Drawdown**: Largest peak-to-trough decline.
- **Post-Tax Net Equity**: The final liquidation value of your portfolio after paying all loan balances and taxes.

### Visualizations
- **Combined Chart**: Shows Portfolio Value (Gross Assets), Net Equity, and Loan Balance on one chart to visualize leverage dynamics.
- **Tax Analysis Tab**: Breakdown of annual tax liabilities and the difference between paying with cash vs. margin.

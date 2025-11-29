# User Guide

Welcome to the Testfol Margin Stresser User Guide. This application allows you to simulate leveraged portfolio performance over historical periods, tracking margin debt, equity levels, and potential margin calls.

## Getting Started

### 1. Configure Global Parameters
The sidebar contains all the configuration options for your backtest.

**Date Range**
- Select the start and end dates for your simulation.
- Data is available from 1885-01-01 to the present.

**Starting Conditions**
- **Starting Value**: The initial value of your portfolio (e.g., $10,000).
- **Rolling Window**: The period (in months) used for rolling analysis (default: 60 months).

**Cash Flows**
- **Cashflow**: Amount to contribute or withdraw periodically.
- **Frequency**: Choose between Yearly, Quarterly, or Monthly.

### 2. Set Rebalancing Options
- **Re-invest Dividends**: Check this box to automatically reinvest dividends.
- **Rebalance Frequency**: Choose how often the portfolio rebalances to its target allocation (Yearly, Quarterly, Monthly, or None).

### 3. Configure Margin Settings
This is where you define how the margin simulation behaves.

**Loan Parameters**
- **Starting Loan**: The initial amount borrowed.
- **Initial Equity %**: Alternatively, set the starting equity percentage.
    - 100% = No leverage (0 loan).
    - 50% = 2x leverage (Loan = Equity).
- **Interest Rate**: The annual interest rate charged on the margin debt.
- **Monthly Margin Draw**: Additional amount to borrow each month (increases debt).

### 4. Define Portfolio Allocation
Enter your portfolio allocation in the table.

| Column | Description | Example |
|--------|-------------|---------|
| **Ticker** | Stock/ETF symbol | `AAPL`, `SPY`, `VTI` |
| **Weight %** | Portfolio percentage | `50` |
| **Maint %** | Maintenance requirement | `30` (30% equity required) |

**Ticker Modifiers**:
You can use Testfol modifiers in the ticker symbol:
- `?L=X`: Leverage multiplier (e.g., `SPY?L=2` for 2x S&P 500).
- `?D=X`: Drag percentage (e.g., `SPY?D=0.5` for 0.5% drag).

**Important**: The total "Weight %" must sum to exactly 100%.

### 5. Run Backtest
Click the **"Run back-test"** button to start the simulation.

## Visualization Modes

### Combined Chart View
A single interactive chart showing:
- **Portfolio**: Total value of assets.
- **Equity**: Net value (Assets - Loan).
- **Loan**: Outstanding debt.
- **Margin Usage %**: How close you are to a margin call (100% = Call).

### Dashboard View
A comprehensive dashboard with separate charts for:
- **Portfolio Value**: Log/Linear scale options.
- **Leverage**: Track current vs. target leverage.
- **Margin Debt**: Visualize debt growth and interest costs.
- **Gauges**: Real-time risk indicators.

## Understanding Results

### Summary Metrics
- **CAGR**: Compound Annual Growth Rate.
- **Sharpe Ratio**: Risk-adjusted return.
- **Max Drawdown**: Largest drop from peak.
- **Final Leverage**: Ending leverage ratio.

### Margin Analysis
- **Margin Call**: Occurs if "Margin Usage %" reaches 100%.
- **Available Margin**: How much more you could borrow before hitting the limit.

### Tax Analysis
The "Tax Analysis" tab provides a breakdown of estimated taxes if you enabled tax simulation.
- **Pay with Margin**: Taxes are paid by increasing the loan.
- **Pay from Cash**: Taxes are paid by selling assets (reducing the portfolio).

## Presets
You can save and load portfolio configurations using the "Save preset" and "Load preset (JSON)" buttons in the sidebar. This is useful for comparing different strategies quickly.

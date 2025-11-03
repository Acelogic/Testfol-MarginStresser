# Testfol Margin Simulator

A powerful Streamlit-based GUI application for backtesting portfolio strategies with margin trading simulation using the Testfol API.

## Overview

This tool allows you to simulate leveraged portfolio performance over historical periods, tracking margin debt, equity levels, and potential margin calls. It provides both a combined chart view and an enhanced dashboard view with multiple visualizations.

**Powered by [testfol.io](https://testfol.io)** - This application uses testfol.io as the backend data provider for all historical price data, dividend information, and portfolio calculations. Testfol.io offers comprehensive market data dating back to 1885, enabling thorough backtesting across multiple market cycles.

## Features

### Core Functionality
- **Historical Backtesting**: Test portfolio strategies from 1885 to present
- **Margin Simulation**: Model margin loans with custom interest rates and maintenance requirements
- **Multiple Asset Support**: Allocate across stocks, ETFs, and other securities
- **Flexible Rebalancing**: Choose yearly, quarterly, or monthly rebalancing
- **Cash Flow Modeling**: Add periodic cash flows to simulate contributions
- **Dividend Reinvestment**: Option to reinvest dividends automatically

### Visualization Modes

#### Combined Chart View
Single interactive chart showing:
- Portfolio value over time
- Equity (net liquidating value)
- Outstanding loan balance
- Margin usage percentage
- Equity percentage

#### Dashboard View
Comprehensive multi-chart dashboard featuring:
- **Portfolio Value Chart**: Compare leveraged vs unleveraged performance
- **Leverage Over Time**: Track current, target, and maximum allowed leverage
- **Margin Debt Evolution**: Visualize debt growth, portfolio value, and monthly interest
- **Margin Status Gauges**: Real-time utilization and leverage indicators
- **Available Margin Display**: Shows additional borrowing power

### Advanced Features
- **Per-Ticker Maintenance Requirements**: Set individual maintenance percentages for each holding
- **Portfolio Presets**: Save and load allocation configurations
- **Log Scale Charts**: Optional logarithmic scaling for better visualization
- **Rolling Window Analysis**: Analyze performance over moving time windows
- **Margin Call Detection**: Automatic identification of maintenance breaches

## Installation

### Prerequisites
- Python 3.7 or higher
- pip package manager

### Dependencies
```bash
pip install -r requirements.txt 
```

### Quick Start
```bash
streamlit run Testfol_MarginStresser.py
```

The application will open in your default web browser at `http://localhost:8501`

## Usage Guide

### 1. Configure Global Parameters

**Date Range**
- Set start and end dates for your backtest
- Available data from 1885-01-01 to present

**Starting Conditions**
- **Starting Value**: Initial portfolio value (default: $10,000)
- **Rolling Window**: Analysis period in months (default: 60)

**Cash Flows**
- Add periodic contributions or withdrawals
- Choose frequency: Yearly, Quarterly, or Monthly

### 2. Set Rebalancing Options

- **Re-invest Dividends**: Toggle dividend reinvestment
- **Rebalance Frequency**: Choose how often to rebalance allocations

### 3. Configure Margin Settings

**Loan Parameters**
- **Starting Loan**: Initial borrowed amount
- **Initial Equity %**: Starting equity percentage (100 = no leverage)
- **Interest Rate**: Annual interest percentage on margin debt
- **Monthly Margin Draw**: Additional monthly borrowing (optional)

**How It Works**:
- If you start with $10,000 and 50% equity, you're borrowing $5,000
- Interest compounds daily based on annual rate
- Optional monthly draws increase debt on the 1st of each month

### 4. Define Portfolio Allocation

The allocation table requires three columns:

| Column | Description | Example |
|--------|-------------|---------|
| **Ticker** | Stock/ETF symbol | `AAPL` or `AAPL?L=2` for 2x leverage |
| **Weight %** | Portfolio percentage | `10.5` |
| **Maint %** | Maintenance requirement | `50` (50% equity required) |

**Ticker Modifiers**: Testfol supports special modifiers to customize ticker behavior
- `?L=X` - Apply leverage multiplier (e.g., `AAPL?L=2` = 2x leveraged Apple)
- `?D=X` - Apply drag percentage (e.g., `SPY?D=0.5` = 0.5% annual drag)
- Multiple modifiers can be combined (e.g., `AAPL?L=2?D=0.3`)

**Examples**:
- `AAPL?L=2` = 2x leveraged Apple position
- `SPY?L=3` = 3x leveraged S&P 500
- `QQQ?L=1.5?D=0.2` = 1.5x leverage with 0.2% drag

For a complete list of available ticker modifiers and advanced options, visit **[testfol.io/help](https://testfol.io/help)**

**Important**: Total weights must sum to exactly 100%

### 5. Set Maintenance Requirements

Each ticker can have a custom maintenance percentage:
- **Lower values** (25-35%): Conservative stocks, broad ETFs
- **Medium values** (40-50%): Individual stocks
- **Higher values** (60-75%): Volatile stocks, leveraged ETFs

The app calculates a weighted average maintenance requirement for your portfolio.

### 6. Choose Visualization Mode

**Combined Chart**
- Select which series to display
- Toggle log scale for left axis
- Unified view with synchronized hover

**Dashboard View**
- Separate charts for different metrics
- Independent log scale controls
- Gauge visualizations for current status
- Enhanced readability for complex data

### 7. Run Backtest

Click **"Run back-test"** to execute the simulation. The app will:

1. Fetch historical price data from Testfol API
2. Calculate portfolio performance with rebalancing
3. Simulate margin debt accumulation
4. Apply daily interest charges
5. Track maintenance requirements
6. Identify any margin calls

## Understanding the Results

### Summary Statistics

**Final Outcomes**
- **Final Portfolio**: Total portfolio value at end date
- **Final Equity**: Net liquidating value (portfolio - loan)
- **Final Loan**: Outstanding margin debt
- **Final Usage %**: Margin utilization vs. maintenance requirement

**Performance Metrics**
- **CAGR**: Compound Annual Growth Rate
- **Sharpe Ratio**: Risk-adjusted return measure
- **Max Drawdown**: Largest peak-to-trough decline

### Dashboard Gauges

**Margin Utilization**
- Green zone (0-50%): Low risk
- Yellow zone (50-80%): Moderate risk
- Red zone (80-100%): High risk
- **100% = Margin Call**

**Leverage**
- Current leverage ratio (Portfolio / Equity)
- Target leverage based on initial settings
- Maximum allowed before margin call

**Available Margin**
- Additional amount you could borrow
- Based on current portfolio value and maintenance requirements

### Maintenance Breaches

The app identifies days when margin usage exceeded 100%, indicating margin call scenarios:
- **Date**: When the breach occurred
- **Usage %**: How far over the maintenance requirement
- **Equity %**: Percentage of portfolio that was equity

## Example Scenarios

### Conservative Portfolio
```
Starting Value: $50,000
Starting Loan: $0 (100% equity)
Interest Rate: 8%
Allocation:
  - VOO (S&P 500): 60%, 25% maint
  - VEA (International): 30%, 25% maint
  - BND (Bonds): 10%, 15% maint
```

### Moderate Leverage
```
Starting Value: $100,000
Starting Loan: $50,000 (50% equity = 2x leverage)
Interest Rate: 8.5%
Allocation:
  - QQQ: 40%, 35% maint
  - VTI: 30%, 30% maint
  - VXUS: 20%, 30% maint
  - GLD: 10%, 25% maint
```

### Aggressive Strategy
```
Starting Value: $25,000
Starting Loan: $18,750 (25% equity = 4x leverage)
Interest Rate: 9.5%
Monthly Draw: $500
Allocation:
  - TQQQ (3x NASDAQ): 40%, 75% maint
  - UPRO (3x S&P): 30%, 75% maint
  - Individual tech: 30%, 50% maint
```

## Portfolio Presets

### Saving Presets
1. Configure your allocation table
2. Click **"Save preset"** in the sidebar
3. Download the JSON file

### Loading Presets
1. Click **"Load preset (JSON)"** in the sidebar
2. Select your saved JSON file
3. The allocation table updates automatically

Presets store ticker symbols, weights, and maintenance percentages.

## Tips & Best Practices

### Risk Management
- Start with higher equity percentages (60-80%) to learn the tool
- Gradually increase leverage as you understand the dynamics
- Monitor margin usage carefully in volatile markets
- Set realistic maintenance requirements based on asset volatility

### Backtesting Strategy
- Test multiple time periods, including bear markets
- Use rolling windows to see performance across different market regimes
- Compare leveraged vs unleveraged returns
- Pay attention to drawdown periods and recovery time

### Allocation Design
- Diversify across uncorrelated assets to reduce volatility
- Balance high-maintenance assets (leveraged ETFs) with lower ones
- Consider mixing growth stocks with stable dividend payers
- Use gold or bonds for portfolio stability

### Interpreting Results
- Higher leverage amplifies both gains and losses
- Interest costs compound over time
- Margin calls often occur during rapid drawdowns
- Recovery from margin calls is extremely difficult

## Technical Details

### Margin Calculation
```python
Daily Interest Rate = (Annual Rate / 100) / 252
Daily Loan = Previous Loan × (1 + Daily Interest Rate)
Monthly Loan = Daily Loan + Monthly Draw (on 1st of month)
```

### Margin Call Trigger
```python
Usage % = Loan / (Portfolio × (1 - Maintenance %))
Margin Call occurs when Usage % ≥ 100%
```

### Leverage Ratio
```python
Current Leverage = Portfolio Value / Equity
Max Leverage = 1 / (1 - Maintenance %)
```

## API Information

This application is **built on [testfol.io](https://testfol.io)**, a comprehensive portfolio backtesting platform that provides the backend infrastructure and data.

### Testfol API Endpoint
```
https://testfol.io/api/backtest
```

### What Testfol Provides
- **Historical price data** for thousands of stocks and ETFs dating back to 1885
- **Dividend information** with automatic reinvestment calculations
- **Portfolio rebalancing** calculations with multiple frequency options
- **Performance statistics** including CAGR, Sharpe ratio, and maximum drawdown
- **Rolling window analysis** for evaluating strategies across different time periods

All portfolio performance calculations, data retrieval, and backtesting logic are handled by testfol.io's robust API. This application adds margin simulation and visualization layers on top of Testfol's core backtesting capabilities.

### Ticker Modifiers

Testfol supports powerful ticker modifiers that customize how individual positions behave. Common modifiers include:

- **`?L=X`** - Leverage multiplier (e.g., `AAPL?L=2` for 2x leverage)
- **`?D=X`** - Annual drag percentage (e.g., `SPY?D=0.5` for 0.5% drag)
- **Multiple modifiers** can be combined (e.g., `AAPL?L=2?D=0.3`)

**Examples in practice**:
- `TQQQ?L=1` - Use TQQQ as-is (already 3x leveraged)
- `VOO?L=2?D=0.15` - 2x leverage on VOO with 0.15% annual drag
- `GLD?L=1.5` - 1.5x leverage on gold

For a complete reference of all available ticker modifiers and advanced features, see the official documentation at **[testfol.io/help](https://testfol.io/help)**

For more information about testfol.io and its capabilities, visit their website at **https://testfol.io**

## Limitations

- Data availability varies by ticker and date range
- Past performance does not guarantee future results
- Simplified margin model (doesn't include all real-world factors)
- Interest rates are constant (real margin rates vary)
- No transaction costs or taxes included
- Assumes perfect rebalancing execution

## Troubleshooting

**"Weights must sum to 100%"**
- Check that your Weight % column adds up to exactly 100.00
- Remove empty rows from the allocation table

**API Request Failed**
- Check your internet connection
- Verify ticker symbols are correct
- Try a shorter date range
- Wait a moment and retry

**Charts Not Displaying**
- Ensure you clicked "Run back-test"
- Check that portfolio weights sum to 100%
- Verify date range is valid

**Slow Performance**
- Reduce the date range
- Decrease rolling window size
- Use fewer allocation table rows

## License

This tool interfaces with the Testfol API. Please review Testfol's terms of service for usage guidelines.

## Disclaimer

**This tool is for educational and research purposes only.** 

- Not financial advice
- Margin trading carries significant risk
- You can lose more than your initial investment
- Past performance does not indicate future results
- Consult a financial advisor before trading on margin

## Support

For issues related to:
- **The GUI application**: Check the code comments and documentation
- **Testfol API**: Visit https://testfol.io
- **Streamlit framework**: Visit https://docs.streamlit.io

---

**Version**: 4.0  
**Last Updated**: 2025

Happy backtesting! 📈

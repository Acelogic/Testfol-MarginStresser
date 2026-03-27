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

### 2. Data Provider Setup (Optional)
The app fetches price data from multiple sources with automatic fallback:
-   **Default**: yfinance (free, no setup required)
-   **Polygon.io**: Better coverage for delisted tickers. Set `POLYGON_API_KEY` environment variable.
-   **Testfol.io**: Used for server-side backtesting. Falls back to local engine if unavailable.

If the Testfol API goes down, backtests automatically run locally using the Shadow Engine with prices from your configured providers. A warning banner will notify you when this happens.

### 3. Strategy Configuration Tabs

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
- **Margin Rate Model**: Select how interest is calculated:
    -   **Tiered (Blended)**: (Default) Uses IBKR Pro-style tiered rates, blending different spreads based on your loan size against the historical Fed Funds benchmark.
    -   **Variable (Fed + Spread)**: Uses the historical **Fed Funds Rate** + a fixed spread.
    -   **Fixed Annual %**: Uses a constant interest rate.
- **Maintenance**: Set the default maintenance requirement.

#### ⚙️ Settings
- **Chart Style**: Switch between "Classic", "Dashboard", and "Candlestick".
- **Log Scale**: Toggle logarithmic axis for long-term growth charts.

---

## Ticker Modifiers
You can use Testfol modifiers in the ticker symbol to simulate leverage or expense ratios:

| Modifier | Description | Example |
| :--- | :--- | :--- |
| `?L=X` | **Leverage**: Multiplies daily returns by X. | `SPY?L=2` (2x S&P 500) |
| `?E=X` | **Expense Ratio**: Applies annual expense ratio (%). | `QQQ?E=0.20` (0.20% annual fee) |
| `?D=X` | **Drag**: Applies annual drag (legacy alias for expense ratio). | `SPY?D=0.50` (0.50% fee) |
| `_SIM` | **Simulation**: Often used to extend history. | `UPRO_SIM` (Simulated 3x SPY) |

You can combine modifiers: `NDXMEGASIM?L=2&E=0.95` (2x leveraged with 0.95% expense ratio).

---

## Understanding the Results

### Summary Metrics
- **CAGR**: Compound Annual Growth Rate.
- **Sharpe Ratio**: Risk-adjusted return (using risk-free rate = 0).
- **Max Drawdown**: Largest peak-to-trough decline.
- **Post-Tax Net Equity**: The final liquidation value of your portfolio after paying all loan balances and taxes.

### 4. Chart & Returns Analysis

#### 📈 Chart Tab (Primary)
The **Chart** tab contains the main visualization tools, organized into sub-tabs:

*   **🧮 Margin Calcs**: (Default) Main portfolio chart (Gross vs Net), margin usage, and detailed safety metrics.
*   **📉 200DMA**: 200-Day Moving Average analysis.
*   **📉 150MA**: 150-Day Moving Average & **Stan Weinstein Stage Analysis**.
*   **📜 Cheat Sheet**: Technical Pivot Points and Support/Resistance levels.

#### 🔮 Monte Carlo Analysis (New)
A robust stochastic simulation to stress-test your strategy against thousands of alternative history timelines.
-   **Configuration**:
    -   **Scenarios**: Choose between 100 (fast) to 5,000 (deep) iterations.
    -   **Start Value / Monthly Add**: Override your backtest settings to answer "What if I started with $X?"
-   **Charts**:
    -   **Fan Chart**: Shows the cone of probability (P10 to P90) over 10 years. Toggle "Show Paths" to see individual random outcomes.
    -   **Distribution Histogram**: Visualizes the bell curve of final portfolio values. Uses explicit "B" (Billion) formatting and supports Log/Linear scales.
-   **Metrics**:
    -   **Median CAGR**: The most likely annual return.
    -   **Worst Case DD (P90)**: The 90th percentile worst drawdown.
    -   **Chance of Loss**: Probability of ending with less money than you started.

#### 🏗️ Margin Details (Margin Calcs Tab)
-   **Combined Chart**: Shows Portfolio Value (Gross Assets), Net Equity, and Loan Balance on one chart to visualize leverage dynamics.
-   **Tax Analysis**: Breakdown of annual tax liabilities and the difference between paying with cash vs. margin.

#### 📉 Technical Analysis (Chart Sub-tabs)
The application includes a comprehensive technical analysis suite to help you identify market trends and regimes.

-   **Moving Averages (150MA & 200MA)**: 
    -   Dedicated sub-tabs in the **Chart** section.
    -   **Metrics**: View "Time Under MA", "Longest Period Under", and "Max Depth" to understand historical drawdown behaviors.
-   **Stan Weinstein Stage Analysis (150MA Tab)**:
    -   Estimates the current market phase based on Price vs. 150MA and the Slope of the MA.
    -   **Stage 1 (Basing)** 🟡: Flat MA after a decline. Accumulation/bottoming phase.
    -   **Stage 2 (Advancing)** 🟢: Bull market trend (Price > Rising MA). Includes "Corrections" when price dips below rising MA.
    -   **Stage 3 (Topping)** 🟠: Flat MA after an advance. Distribution/exhaustion phase.
    -   **Stage 4 (Declining)** 🔴: Bear market trend (Price < Falling MA). Includes "Bear Rallies" when price pops above falling MA.
    -   **Methodology**: Uses adaptive volatility-based thresholds, distinguishes Stage 1 from 3 based on prior trend context, and applies 5-day smoothing to reduce daily noise.
-   **Trader's Cheat Sheet**:
    -   Located in the **Cheat Sheet** sub-tab.
    -   Displays standard deviation levels, Fibonacci retracements, and Pivot Points for quick reference.

#### 📊 Returns Analysis Tab
(Formerly "Analysis")
Focuses on periodic performance metrics:
-   **Seasonal Summary**: Monthly return heatmaps.
-   **Annual/Quarterly/Monthly**: Detailed return tables.

#### 🩻 X-Ray Analyzer
The X-Ray engine allows you to peer inside your ETFs to see the actual underlying exposures. It supports a wide range of assets including:

-   **Standard ETFs**: SPY, QQQ, VTI, BND, etc.
-   **Leveraged ETFs**: Automatically decomposes leveraged funds (e.g., `UPRO`) into their 3x underlying components (e.g., 300% SPY).
-   **Simulated Tickers**: Full support for extended history tickers from the `shadow_backtest` engine:
    -   **Markets**: `SPYSIM`, `QQQSIM`, `VTVSIM` (Value), `VUGSIM` (Growth).
    -   **Sectors**: `XLBSIM` (Materials), `XLKSIM` (Tech), `XLFSIM` (Financials), etc.
    -   **Bonds**: `IEFSIM` (7-10y), `SHYSIM` (1-3y), `TLTSIM` (20y+).
    -   **Modifiers**: Works seamlessly with leverage modifiers like `SPYSIM?L=2`.


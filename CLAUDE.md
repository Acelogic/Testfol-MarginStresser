# Project: Testfol-MarginStresser

## Quick Start
```bash
streamlit run testfol_charting.py --server.port 8501
```

## Entry Point
- **Main file:** `testfol_charting.py` (NOT `app/main.py`)

## Project Structure
- `app/core/backtest_orchestrator.py` - Backtest routing (API vs local engine with automatic failover)
- `app/core/shadow_backtest.py` - Local backtest engine (FIFO tax lots, rebalancing, margin)
- `app/core/calculations.py` - Core calculation functions (MA analysis, stats, etc.)
- `app/services/testfol_api.py` - API calls to Testfol backend
- `app/services/price_providers.py` - Multi-provider price data (Polygon.io → yfinance chain)
- `app/services/data_service.py` - Data sourcing (SIM tickers, CSV splicing, provider chain)
- `app/ui/charts.py` - Chart rendering functions (Plotly visualizations)
- `app/ui/results.py` - Results display and tab management
- `app/common/cache.py` - Disk-based caching with HMAC signing

## Key Features
- Margin stress testing and backtesting
- Multi-provider price data: Polygon.io → yfinance (automatic failover from Testfol API)
- Moving Average analysis: 200DMA, 150MA, Munger200WMA
- Stan Weinstein Stage Analysis
- Portfolio comparison and benchmarking

## Environment Variables
- `POLYGON_API_KEY` - (Optional) Polygon.io API key for price data. Falls back to yfinance if not set.
- `TESTFOL_EMAIL` / `TESTFOL_PASSWORD` - Testfol.io credentials for API backtesting
- `TESTFOL_API_KEY` - Alternative static API key for Testfol

## Moving Average Indicators
- **200DMA** - 200-day moving average (daily data)
- **150MA** - 150-day moving average with Weinstein Stage Analysis
- **Munger200WMA** - 200-week moving average (~4 years, weekly data)

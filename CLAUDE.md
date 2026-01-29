# Project: Testfol-MarginStresser

## Quick Start
```bash
streamlit run testfol_charting.py --server.port 8501
```

## Entry Point
- **Main file:** `testfol_charting.py` (NOT `app/main.py`)

## Project Structure
- `app/core/calculations.py` - Core calculation functions (MA analysis, stats, etc.)
- `app/ui/charts.py` - Chart rendering functions (Plotly visualizations)
- `app/ui/results.py` - Results display and tab management
- `app/services/testfol_api.py` - API calls to Testfol backend

## Key Features
- Margin stress testing and backtesting
- Moving Average analysis: 200DMA, 150MA, Munger200WMA
- Stan Weinstein Stage Analysis
- Portfolio comparison and benchmarking

## Moving Average Indicators
- **200DMA** - 200-day moving average (daily data)
- **150MA** - 150-day moving average with Weinstein Stage Analysis
- **Munger200WMA** - 200-week moving average (~4 years, weekly data)

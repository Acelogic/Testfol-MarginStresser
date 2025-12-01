# Testfol Margin Stresser

A powerful Streamlit-based GUI application for backtesting portfolio strategies with margin trading simulation using the [Testfol API](https://testfol.io).

## Overview

This tool allows you to simulate leveraged portfolio performance over historical periods, tracking margin debt, equity levels, and potential margin calls. It provides both a combined chart view and an enhanced dashboard view with multiple visualizations.

**Powered by [testfol.io](https://testfol.io)** - This application uses testfol.io as the backend data provider for all historical price data, dividend information, and portfolio calculations.

## Documentation

We have moved our detailed documentation to the app itself and the `docs/` folder:

- **[User Guide](docs/user_guide.md)**: Detailed instructions on how to configure and use the app.
- **[Methodology](docs/methodology.md)**: Explanation of margin calculations, tax logic, and assumptions.
- **[FAQ & Troubleshooting](docs/faq.md)**: Common questions and fixes for known issues.

## Quick Start

### Prerequisites
- Python 3.7 or higher
- pip package manager

### Installation

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Run the application:
   ```bash
   streamlit run testfol_charting.py
   ```

3. The app will open in your browser at `http://localhost:8501`.

## Features

- **Historical Backtesting**: Test portfolio strategies from 1885 to present.
- **Margin Simulation**: Model margin loans with custom interest rates and maintenance requirements.
- **Tax Simulation**: Estimate Federal and State taxes with HIFO/LIFO/FIFO lot tracking.
- **Interactive Charts**: TradingView-style candlestick charts and performance dashboards.
- **Portfolio Presets**: Save and load your favorite strategies.

## License

This project is licensed under the GPL V2 License - see the [LICENSE](LICENSE) file for details.

This tool interfaces with the Testfol API. Please review Testfol's terms of service for usage guidelines.

## Disclaimer

**This tool is for educational and research purposes only.** Not financial advice. Margin trading carries significant risk.

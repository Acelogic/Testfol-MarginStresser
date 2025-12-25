# Edgar Parser & NDX Mega Simulator

This project contains tools to parse SEC filings, reconstruct historical Nasdaq-100 weights, and simulate the "NDX Mega" strategies.

## Directory Structure

- **`src/`**: Python source code.
  - `reconstruct_weights.py`: Main script to generate quarterly weights from filings.
  - `backtest_ndx_mega2.py`: Simulator for NDX Mega 2.0 (Target 47%, Buffer 50%, Min 9).
  - `validate_ndx.py`: Validation suite (Sim vs QQQ, Mega vs QBIG).
  - `config.py`: Central configuration for paths and parameters.
  - `edgar_parser.py`: Parses raw HTML filings into CSV components.
  - `edgar_downloader.py`: Downloads 485BPOS filings from Edgar.
- **`assets/`**: Input data files.
  - `nasdaq_components.csv`: Parsed raw filing data (Date, Company, Shares, Value).
  - `nasdaq_changes.csv`: Historical index add/delete events.
  - `name_mapping.json`: Mapping of Company Names to Tickers.
- **`results/`**: Generated outputs.
  - `nasdaq_quarterly_weights.csv`: Reconstructed weights (The "Golden Source" for backtesting).
  - `charts/`: Validation charts.
  - `output/`: Constituent history CSVs.
- **`cache/`**: Data caches.
  - `prices_cache.pkl`: `yfinance` price data cache.
  - `downloads/`: Raw SEC filing HTML/TXT files.
- **`debug/`**: Debugging logs and intermediate reports.

## How to Run

1. **Reconstruct Weights** (if assets changed):
   ```bash
   python src/reconstruct_weights.py
   ```

2. **Run Backtest (Mega 2.0)**:
   ```bash
   python src/backtest_ndx_mega2.py
   ```
   *Outputs `NDXMEGA2SIM.csv` to the project root (../) for use by the main app.*

3. **Validate**:
   ```bash
   python src/validate_ndx.py
   ```

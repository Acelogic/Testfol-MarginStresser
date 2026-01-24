# Architecture

**Analysis Date:** 2026-01-23

## Pattern Overview

**Overall:** Modular MVC-inspired Streamlit application with separated concerns across services, core logic, UI rendering, and reporting layers. The application serves as a financial backtesting platform with multi-portfolio support, margin simulation, and tax-aware analysis.

**Key Characteristics:**
- Streamlit-based web UI with session-state management for interactivity
- Pluggable service layer for API integration and data fetching
- Core computational layer for backtesting, Monte Carlo, and tax calculations
- Separate UI rendering modules for different visualization concerns
- FastAPI backend for mobile integration
- Multi-platform support (web via Streamlit, mobile via Expo React Native)

## Layers

**UI/Presentation Layer:**
- Purpose: User interface rendering and interaction management
- Location: `app/ui/`
- Contains: Sidebar navigation, configuration forms, results visualization, charts, asset explorers
- Depends on: Services, core calculations, common utilities
- Used by: `testfol_charting.py` (main Streamlit entry point), FastAPI backend

**Services Layer:**
- Purpose: External data fetching, API integration, data transformation
- Location: `app/services/`
- Contains: `testfol_api.py` (Testfol API client with caching), `data_service.py` (market data fetching via yfinance/API), `xray_engine.py` (ETF expansion/holdings analysis), `backend.py` (FastAPI endpoints)
- Depends on: External APIs (testfol.io, yfinance), file system for caching
- Used by: Core layer, UI layer, main application

**Core/Business Logic Layer:**
- Purpose: Backtesting simulation, mathematical calculations, tax computations
- Location: `app/core/`
- Contains: `shadow_backtest.py` (leverage-aware simulation), `calculations.py` (CAGR, Sharpe, drawdown, tax-adjusted equity), `monte_carlo.py` (probability distributions), `tax_library.py` (tax lot tracking, reporting)
- Depends on: pandas, numpy
- Used by: Services layer (via results), UI layer (for stats/metrics)

**Common/Utility Layer:**
- Purpose: Shared helpers, styling, preset management
- Location: `app/common/`
- Contains: `utils.py` (Streamlit wrappers, preset I/O, color helpers, documentation rendering)
- Depends on: Streamlit, pandas, file system
- Used by: All layers

**Reporting Layer:**
- Purpose: Generate HTML reports from simulation results
- Location: `app/reporting/`
- Contains: `report_generator.py` (HTML generation, multi-portfolio comparison)
- Depends on: pandas, plotly
- Used by: UI results rendering

## Data Flow

**Primary Backtest Flow:**

1. User configures portfolios (tickers, weights, rebalance freq, margin params) via `app/ui/configuration.py`
2. User sets global dates and run parameters via `app/ui/sidebar.py`
3. Main app (`testfol_charting.py`) validates and iterates over configured portfolios
4. For each portfolio:
   - `app/services/testfol_api.fetch_backtest()` fetches historical data (cached)
   - `app/core/shadow_backtest.run_shadow_backtest()` simulates portfolio with leverage
   - `app/core/calculations.generate_stats()` computes CAGR, Sharpe, drawdown
   - `app/core/tax_library` (if enabled) calculates tax impact and lot tracking
5. Results dict containing `port_series`, `stats`, `trades_df` flows to `app/ui/results.render()`
6. `app/ui/charts.render_multi_portfolio_chart()` visualizes performance across portfolios
7. `app/reporting/report_generator.generate_html_report()` exports downloadable HTML

**Data Service Flow:**

1. User requests tickers (e.g., "NDXMEGASIM", "UPRO", "QQQ")
2. `app/services/data_service.fetch_component_data()` resolves tickers:
   - Special handling for simulated tickers (NDXMEGASIM loads from `data/NDXMEGASIM.csv`)
   - Live tickers fetch via yfinance or Testfol API
3. Returns combined DataFrame with price history
4. Passed to backtest engine for simulation

**X-Ray Flow:**

1. User selects portfolio in X-Ray tab (`app/ui/xray_view.py`)
2. `app/services/xray_engine.compute_xray()` expands holdings:
   - Loads ETF holdings from `data/etf_xray/` module
   - Maps to underlying stocks with weights
   - Aggregates to company level
3. Results rendered in expandable table with leverage annotations

**State Management:**

- Streamlit `st.session_state` maintains across reruns:
  - Portfolio list, active portfolio index
  - Global cashflow settings
  - UI widget values (dates, parameters)
- FastAPI (`app/services/backend.py`) uses request-response cycle (no session state)
- Mobile app (`margin-stresser-mobile/`) uses AsyncStorage for persisted scenarios

## Key Abstractions

**Portfolio:**
- Purpose: Represents a single investment strategy configuration
- Location: `app/ui/configuration.py` (definition), `testfol_charting.py` (iteration)
- Pattern: Dictionary with keys: `id`, `name`, `alloc_df`, `rebalance`, `cashflow`
- Contains allocation DataFrame (Ticker, Weight %, Maint %), rebalancing rules, cashflow params

**Backtest Result:**
- Purpose: Complete simulation output for a portfolio
- Examples: Returned from `testfol_api.fetch_backtest()` and `shadow_backtest.run_shadow_backtest()`
- Pattern: Dictionary containing:
  - `port_series`: Time-indexed equity curve (pd.Series)
  - `stats`: Performance metrics dict (cagr, sharpe, max_drawdown, vol)
  - `trades_df`, `pl_by_year`: Trade and P&L breakdowns
  - `start_date`, `end_date`: Period bounds
  - `raw_response`: API response if applicable

**Tax Lot:**
- Purpose: Track individual cost basis for tax reporting
- Examples: `app/core/tax_library.TaxLot` dataclass
- Pattern: Holds ticker, acquisition date, quantity, cost basis; supports fractional sales with remaining lot tracking

**Rebalancing Configuration:**
- Purpose: Define portfolio rebalancing rules
- Pattern: Nested dict with `mode` (Standard/Custom), `freq` (Yearly/Quarterly/Monthly), `month`, `day`, `compare_std` (for hybrid analysis)

## Entry Points

**Streamlit Web App:**
- Location: `/Users/mcruz/Developer/Testfol-MarginStresser/testfol_charting.py`
- Triggers: `streamlit run testfol_charting.py`
- Responsibilities: Main UI orchestration, multi-portfolio iteration, caching wrappers, results aggregation

**FastAPI Backend:**
- Location: `app/services/backend.py`
- Triggers: `uvicorn app.services.backend:app --host 0.0.0.0 --port 8000`
- Responsibilities: REST endpoints for mobile app (/xray, /backtest, /montecarlo)

**Mobile App:**
- Location: `margin-stresser-mobile/`
- Entry: `margin-stresser-mobile/app/_layout.tsx`
- Triggers: `expo start` or `expo start --web`
- Responsibilities: Cross-platform UI (iOS, Android, web) using Expo Router

**Data Rebuild Script:**
- Location: `data/ndx_simulation/scripts/rebuild_all.py`
- Triggers: Called by `data_service.fetch_component_data()` if simulation CSV is corrupted
- Responsibilities: Regenerates NDXMEGASIM.csv from raw data

## Error Handling

**Strategy:** Defensive layering with user-friendly fallbacks

**Patterns:**

1. **API/Data Fetch Errors:**
   - `testfol_api.fetch_backtest()`: Cache corruption detection with auto-retry; MD5 hash validation
   - `data_service.fetch_component_data()`: Fallback to yfinance if Testfol API fails; CSV rebuild if corrupted
   - Try-except with `st.warning()` or `st.error()` for user feedback

2. **Calculation Errors:**
   - `calculations.py`: Empty series guards (returns 0.0 for CAGR, Sharpe, etc.)
   - Division by zero checks (e.g., in Sharpe ratio when std == 0)
   - Graceful degradation in tax calculations (skips lot tracking if tax disabled)

3. **Configuration Validation:**
   - `configuration.py`: Weight validation (ensures totals near 100%)
   - Ticker validation (resolves aliases, warns on missing mappings)
   - Date range checks (prevents end_date before start_date)

4. **UI Rendering:**
   - `results.py`: Checks for empty `port_series` before plotting
   - Clipping logic prevents index errors on date-filtered series
   - Session state initialization prevents UnboundLocalError

## Cross-Cutting Concerns

**Logging:**
- Uses Python's `logging` module (e.g., in `xray_engine.py` for ticker loading)
- Console output via `print()` for debug info
- Streamlit `st.warning()`, `st.error()`, `st.success()` for user messaging

**Validation:**
- Weight normalization in `configuration.py` (ensure portfolio sums to 100%)
- Ticker parsing in `shadow_backtest.py` (query string extraction, mapping to yfinance symbols)
- Date range validation in main loop and Monte Carlo filtering

**Authentication:**
- Bearer token for Testfol API (optional, via `app/ui/sidebar.py` password input)
- Defaults to `TESTFOL_API_KEY` environment variable
- No auth required for FastAPI backend (localhost assumption)

**Caching:**
- Streamlit `@st.cache_data` on `fetch_backtest()` and `run_shadow_backtest()`
- Disk cache in `data/api_cache/` (MD5-hashed pickle files) for API responses
- TTL: 3600 seconds (1 hour) for UI caching, indefinite for disk cache (manual invalidation)

---

*Architecture analysis: 2026-01-23*

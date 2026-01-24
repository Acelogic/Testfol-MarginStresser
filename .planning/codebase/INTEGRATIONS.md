# External Integrations

**Analysis Date:** 2026-01-23

## APIs & External Services

**Financial Data:**
- **Testfol API** - Primary data source for historical total return backtesting
  - SDK/Client: Native HTTP client via `requests` library
  - Base URL: `https://testfol.io/api/backtest`
  - Auth: Bearer token via `TESTFOL_API_KEY` environment variable (optional)
  - Usage: `app/services/testfol_api.py:fetch_backtest()` and `simulate_margin()`
  - Features: Retry strategy with exponential backoff (5 retries, 2s-32s delays) for rate limits and server errors
  - Response format: JSON with chart history (timestamp array + values array), stats, rebalancing events
  - Cache: Local disk cache with MD5-keyed pickle files in `data/api_cache/`
  - Headers: Custom User-Agent and Referer spoofing to appear as browser request

- **Yahoo Finance (yfinance)** - Secondary equity data source
  - SDK/Client: `yfinance` Python package
  - Usage: Real-time price fetching for equity symbols, especially QBIG proxy ticker for simulated portfolios
  - Location: `app/services/data_service.py:fetch_component_data()` (lines 63-82)
  - Purpose: Fallback for live pricing when Testfol API unavailable; primary source for NDXMEGASIM/NDXMEGA2SIM splicing
  - Handles varied return structures (DataFrame with Close/Adj Close columns)

**Economic Data:**
- **FRED (Federal Reserve Economic Data)** - Fed Funds Rate historical data
  - Service: U.S. Federal Reserve Economic Research
  - Data: Daily Fed Funds Rate (FEDFUNDS series) from 1954 to present
  - Endpoint: `https://fred.stlouisfed.org/graph/fredgraph.csv?id=FEDFUNDS`
  - Client: Direct CSV download via `requests` library
  - Cache: Local CSV file at `data/FEDFUNDS.csv` (cached for 30 days before refresh)
  - Usage: `app/services/data_service.py:get_fed_funds_rate()` for variable margin rate calculations
  - Headers: Browser User-Agent required to bypass anti-scraping (403 Forbidden without it)

## Data Storage

**Databases:**
- Not used - Application is stateless with no persistent database (SQLite, PostgreSQL, MongoDB, etc.)

**File Storage:**
- Local filesystem only
  - **Simulation Data**: `data/NDXMEGASIM.csv`, `data/NDXMEGA2SIM.csv` - Pre-computed NDX simulations
  - **API Cache**: `data/api_cache/` - Pickle-serialized backtest responses (MD5-keyed by request parameters)
  - **Historical Rates**: `data/FEDFUNDS.csv` - Fed Funds Rate data
  - **Ticker Mappings**: `data/ndx_simulation/data/assets/company_tickers.json` - SEC company ticker reference (from SEC EDGAR)

**Caching:**
- In-memory caching: Streamlit's `@st.cache_data()` decorator with TTL (3600 seconds) for expensive operations
  - `cached_fetch_backtest()` - Caches API responses across user sessions
  - `cached_run_shadow_backtest_v2()` - Caches computed backtests
  - `fetch_component_data()` - Caches historical price series
  - `get_fed_funds_rate()` - Caches FRED data with 30-day refresh interval
- Disk cache: Pickle-based cache files in `data/api_cache/` with automatic corruption detection and rebuild

## Authentication & Identity

**Auth Provider:**
- None - Application is unauthenticated for public use
- Optional: Testfol API Key can be provided via environment variable (`TESTFOL_API_KEY`) or input field in Streamlit UI
  - Passed as Bearer token in Authorization header
  - Used for rate-limit increases or premium endpoints on Testfol

## Monitoring & Observability

**Error Tracking:**
- Not integrated - Errors logged to console only

**Logs:**
- Console logging via Python `logging` module and `print()` statements
  - Example: `logging.info()`, `logging.warning()` in `app/services/xray_engine.py`
- Streamlit UI feedback via `st.warning()`, `st.error()`, `st.success()` for user-facing messages
- Debug output in API responses (e.g., "DEBUG: Cache HIT/MISS" in `testfol_api.py`)

## CI/CD & Deployment

**Hosting:**
- Streamlit Cloud - Web app: `testfol-marginstresser.streamlit.app`
- Local development: `streamlit run testfol_charting.py`
- Optional: Local FastAPI server for REST API: `python -m uvicorn app.services.backend:app --host 0.0.0.0 --port 8000`
- Expo over-the-air updates for mobile app distribution

**CI Pipeline:**
- Not detected - No GitHub Actions, GitLab CI, or similar CI/CD configuration files present

## Environment Configuration

**Required env vars:**
- `TESTFOL_API_KEY` (optional) - Bearer token for Testfol API authentication
- `PYTHONPATH` (implicit) - Must include project root for module imports

**Secrets location:**
- Environment variables only (no .env file checked in)
- Streamlit Cloud secrets management for deployed instance (not visible in repo)

## Webhooks & Callbacks

**Incoming:**
- Not used - Application is request/response only

**Outgoing:**
- Not used - Application does not send webhooks or background notifications

## Mobile App API Integration

**Backend Connection:**
- REST API endpoint: `http://{LOCAL_IP}:8000/run_stress_test` (FastAPI backend)
  - Defined in `margin-stresser-mobile/app/(tabs)/index.tsx` with hardcoded IP `192.168.68.55:8000`
  - Uses native `fetch()` API for HTTP requests
  - POST request with JSON body containing backtest parameters
  - Error handling: Parses JSON error responses from FastAPI

**Server Endpoints:**
- `GET /` - Health check
- `POST /xray` - X-Ray analysis endpoint (portfolio decomposition)
  - Request body: `{ portfolio: { "QQQ": 0.5, "AAPL": 0.5 }, portfolio_name: "Portfolio" }`
  - Response: Holdings list with allocation details
- `POST /monte_carlo` - Monte Carlo simulation endpoint
  - Request body: Returns list, n_sims, n_years, initial_val, monthly_cashflow, block_size
  - Response: Simulation results with metrics (percentiles, mean, median)
- `POST /backtest` - Full backtest endpoint (inferred from BacktestRequest Pydantic model)
  - Complex request with tickers, margin config, tax settings, rebalance frequency

**CORS:**
- Enabled via FastAPI CORSMiddleware with permissive settings:
  - Allow origins: `["*"]`
  - Allow credentials: `True`
  - Allow methods: `["*"]`
  - Allow headers: `["*"]`

---

*Integration audit: 2026-01-23*

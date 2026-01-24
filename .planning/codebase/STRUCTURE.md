# Codebase Structure

**Analysis Date:** 2026-01-23

## Directory Layout

```
Testfol-MarginStresser/
├── .planning/               # GSD planning documents
├── .git/                    # Git repository
├── app/                     # Main Python application (Streamlit + FastAPI)
│   ├── __init__.py
│   ├── common/              # Shared utilities and helpers
│   ├── core/                # Core business logic (backtesting, calculations)
│   ├── reporting/           # HTML report generation
│   ├── services/            # API clients and data services
│   └── ui/                  # Streamlit UI components
├── data/                    # Data files, caches, simulations
│   ├── api_cache/           # Disk cache for API responses (pickled)
│   ├── etf_xray/            # ETF holdings expansion module
│   ├── ndx_simulation/       # NDX mega-cap simulation data
│   ├── presets.json         # Saved strategy presets
│   ├── NDXMEGASIM.csv       # Simulated mega-cap index
│   └── NDXMEGA2SIM.csv      # Alternative simulation
├── debug_tools/             # Standalone debugging utilities
│   ├── api_debug/
│   ├── excel_debug/
│   └── tax_verification/
├── docs/                    # Documentation files
├── margin-stresser-mobile/  # React Native/Expo mobile app
│   ├── app/                 # App entry point and screens
│   ├── components/          # Reusable React components
│   ├── context/             # React Context for state management
│   ├── constants/           # App constants and config
│   ├── hooks/               # Custom React hooks
│   └── public/              # Static assets
├── testfol_charting.py      # MAIN ENTRY POINT (Streamlit app)
├── requirements.txt         # Python dependencies
├── README.md
├── LICENSE
└── .gitignore
```

## Directory Purposes

**app/:**
- Purpose: Core application code (business logic, UI, services)
- Contains: Python modules organized by layer
- Key files: `__init__.py` for module discovery

**app/common/:**
- Purpose: Shared utilities across layers
- Contains: `utils.py` (preset I/O, Streamlit helpers, styling, documentation)
- Key files: `utils.py` for utility functions

**app/core/:**
- Purpose: Financial computation and simulation logic
- Contains: Shadow backtesting, CAGR/Sharpe calculations, Monte Carlo, tax lot tracking
- Key files:
  - `shadow_backtest.py`: Lever-aware portfolio simulation with rebalancing
  - `calculations.py`: Performance metrics (CAGR, max drawdown, Sharpe, tax-adjusted returns)
  - `monte_carlo.py`: Probability simulation with block bootstrapping
  - `tax_library.py`: Tax lot management and FIFO/specific ID tracking

**app/reporting/:**
- Purpose: Static report generation
- Contains: HTML export logic for results
- Key files: `report_generator.py` (single and multi-portfolio HTML reports)

**app/services/:**
- Purpose: External integrations and data fetching
- Contains: API clients, data loaders, caching
- Key files:
  - `testfol_api.py`: Testfol API client with MD5-hashed disk caching
  - `data_service.py`: Market data fetching (yfinance, special handling for NDXMEGASIM)
  - `xray_engine.py`: ETF holdings expansion and aggregation
  - `backend.py`: FastAPI server for mobile client (POST /backtest, /xray, /montecarlo)

**app/ui/:**
- Purpose: Streamlit interactive UI components
- Contains: Forms, visualizations, navigation
- Key files:
  - `sidebar.py`: Date range picker, API token input, run button
  - `configuration.py`: Multi-portfolio manager, asset allocation editor, rebalance/margin config
  - `results.py`: Performance metrics display and data clipping logic
  - `charts.py`: Plotly charts (equity curves, drawdown, tax impact, multi-portfolio comparison)
  - `xray_view.py`: Expandable ETF holdings table
  - `ndx_scanner.py`: NDX asset screening
  - `asset_explorer.py`: Interactive asset picker

**data/:**
- Purpose: Persistent data storage (caches, simulations, reference files)
- Contains: CSV files, pickle caches, reference data
- Key files:
  - `api_cache/`: Cache directory (MD5-hashed pickle files from testfol_api)
  - `NDXMEGASIM.csv`: Historical price data for mega-cap simulation
  - `presets.json`: JSON list of saved strategy configurations
  - `etf_xray/src/`: ETF holdings database fetcher module
  - `ndx_simulation/src/`: Simulation generators and data builders
  - `ndx_simulation/scripts/rebuild_all.py`: Auto-rebuilds CSV files if corrupted

**debug_tools/:**
- Purpose: Standalone development/debugging scripts
- Contains: API testing, Excel export tools, tax verification utilities
- Status: Development-only, not used in production

**docs/:**
- Purpose: User and developer documentation
- Contains: Markdown files explaining features, API usage
- Status: Reference material

**margin-stresser-mobile/:**
- Purpose: Cross-platform mobile app (iOS, Android, web)
- Entry: `app/_layout.tsx` (Expo Router root layout)
- Contains:
  - `app/(tabs)/`: Tab-based screens (Index/Home, Explore, Scenarios, History, Settings, Guide)
  - `app/context/AppContext.tsx`: React Context for app-wide state (scenarios, history, settings)
  - `components/ui/`: Reusable UI components (buttons, inputs, cards)
  - `hooks/use-color-scheme.ts`: Dark/light theme detection
  - `constants/`: App-wide constants (colors, strings)
  - `public/`: Static web assets

## Key File Locations

**Entry Points:**
- `testfol_charting.py`: Main Streamlit web application
- `margin-stresser-mobile/app/_layout.tsx`: Mobile app entry (Expo Router)
- `app/services/backend.py`: FastAPI server entry

**Configuration:**
- `requirements.txt`: Python dependencies
- `margin-stresser-mobile/package.json`: Node dependencies
- `margin-stresser-mobile/tsconfig.json`: TypeScript config
- `data/presets.json`: Saved strategy presets (JSON)

**Core Logic:**
- `app/core/shadow_backtest.py`: Leverage-aware backtesting engine
- `app/core/calculations.py`: Performance calculations and tax adjustments
- `app/core/monte_carlo.py`: Probability simulations
- `app/core/tax_library.py`: Tax lot tracking and reporting

**Data Fetching:**
- `app/services/testfol_api.py`: Testfol API client with caching
- `app/services/data_service.py`: Market data fetching and special ticker handling
- `app/services/xray_engine.py`: ETF expansion logic

**UI Rendering:**
- `app/ui/sidebar.py`: Navigation and global settings
- `app/ui/configuration.py`: Portfolio and parameter configuration
- `app/ui/results.py`: Results display and metrics
- `app/ui/charts.py`: Plotly visualization logic

**Testing:**
- Not detected (no test files in structure)

## Naming Conventions

**Files:**
- Python modules: snake_case (e.g., `shadow_backtest.py`, `testfol_api.py`)
- React/TypeScript: PascalCase for components (e.g., `AppContext.tsx`), snake_case for hooks (e.g., `use-color-scheme.ts`)
- Data files: ALL_CAPS or descriptive (e.g., `NDXMEGASIM.csv`, `presets.json`)

**Directories:**
- Package directories: snake_case (e.g., `api_cache`, `etf_xray`)
- Feature directories: descriptive, lowercase (e.g., `ndx_simulation`, `debug_tools`)
- Route directories: snake_case with parentheses for Expo Router groups (e.g., `(tabs)`, `(modals)`)

**Functions and Classes:**
- Python functions: snake_case (e.g., `fetch_backtest()`, `render_sidebar()`)
- Python classes: PascalCase (e.g., `TaxLot`)
- React components: PascalCase (e.g., `AppProvider`, `AppContext`)
- React hooks: camelCase starting with "use" (e.g., `useColorScheme()`)

## Where to Add New Code

**New Financial Calculation Feature:**
- Primary code: `app/core/` (create new file or extend `calculations.py`)
- Tests: Not present; would go in `tests/unit/core/` if adding tests
- Example: To add a new metric (volatility ratio), extend `app/core/calculations.py`

**New Portfolio Analysis Tool:**
- Primary code: `app/services/` (new module or extend existing)
- UI: `app/ui/` (new file, e.g., `app/ui/new_tool.py`)
- Integration: Add import and render call in `testfol_charting.py`
- Example: Adding a correlation analyzer would create `app/services/correlation_engine.py` and `app/ui/correlation_view.py`

**New API Endpoint (Mobile):**
- Endpoint: Add to `app/services/backend.py` as a new `@app.post()` or `@app.get()`
- Request/Response Models: Add Pydantic BaseModel classes to `app/services/backend.py`
- Logic: Import from `app/core/` and `app/services/`
- Example: `@app.post("/scenario-analysis")` would call `app.core.monte_carlo.run_monte_carlo()`

**New Mobile Screen:**
- File: `margin-stresser-mobile/app/(tabs)/new-screen.tsx`
- State: Add context in `margin-stresser-mobile/app/context/AppContext.tsx` if needed
- Components: Create reusable components in `margin-stresser-mobile/components/ui/`
- Navigation: Update `margin-stresser-mobile/app/(tabs)/_layout.tsx`

**New Chart Type:**
- Primary code: `app/ui/charts.py` (add new function `render_chart_name()`)
- Integration: Call from `app/ui/results.py` within appropriate tab
- Dependencies: Use plotly.graph_objects for consistency

**Utilities:**
- Shared helpers: `app/common/utils.py`
- Styling: Color functions and Streamlit wrappers in `app/common/utils.py`
- Data transformations: Create service in `app/services/` or core module if financial

## Special Directories

**data/api_cache/:**
- Purpose: Disk cache for Testfol API responses
- Generated: Yes (automatically created by `testfol_api.fetch_backtest()`)
- Committed: No (in `.gitignore`)
- Files: MD5-hashed `.pkl` files (one per unique request)
- Invalidation: Delete files manually or implement TTL expiration

**data/etf_xray/:**
- Purpose: ETF holdings database and fetcher module
- Generated: Yes (populated by periodic external data fetching)
- Committed: Partially (src/ committed, cache/ in .gitignore)
- Key module: `data/etf_xray/src/etf_holdings_fetcher.py` (fetches from external APIs)

**data/ndx_simulation/:**
- Purpose: NDX mega-cap simulation data generation and reference
- Generated: Yes (scripts generate CSVs from raw data)
- Committed: Partially (scripts and source committed, generated CSVs checked in)
- Rebuild trigger: Auto-triggered by `data_service.py` if CSV corruption detected
- Key script: `data/ndx_simulation/scripts/rebuild_all.py`

**margin-stresser-mobile/.expo/:**
- Purpose: Expo CLI metadata and cache
- Generated: Yes
- Committed: Partially (types/ committed, web/ in .gitignore)
- Status: Development-only

**__pycache__/:**
- Purpose: Python bytecode cache
- Generated: Yes (automatic)
- Committed: No (in `.gitignore`)
- Cleanup: Safe to delete; will be regenerated

---

*Structure analysis: 2026-01-23*

# Codebase Concerns

**Analysis Date:** 2026-01-23

## Tech Debt

**Bare Exception Handlers (Silent Failures):**
- Issue: Multiple `except: pass` blocks silently ignore errors without logging, making debugging difficult
- Files:
  - `app/services/xray_engine.py` (lines 94, 102)
  - `app/ui/charts.py` (line 1815)
  - `data/ndx_simulation/scripts/holdings_viewer.py` (lines 176, 181)
- Impact: Errors are swallowed silently. Network failures, data corruption, or API issues provide no feedback to users or logs
- Fix approach: Replace `except: pass` with explicit exception types and logging. At minimum log a warning with the exception details

**Conditional Fallback Logic in Data Fetching:**
- Issue: `app/services/data_service.py` uses nested try/except blocks with silent fallbacks that mask failures
- Files: `app/services/data_service.py` (lines 19-170)
- Impact: When Yahoo Finance fails, falls back to Testfol API silently. If both fail, returns empty DataFrame with only a warning. User doesn't know which data source succeeded or failed
- Fix approach: Add explicit logging for each fallback attempt. Track data source provenance. Return metadata about which source was used

**Streamlit Cache with Mutable State:**
- Issue: `@st.cache_data` decorators are used extensively, but cached functions may contain mutable defaults (lists, dicts) that persist across sessions
- Files: `app/services/data_service.py` (lines 9, 173), `app/ui/charts.py` (multiple decorators)
- Impact: Cache poisoning possible if mutable objects are modified. TTL=3600 for component data may cause stale data to persist across hour boundaries
- Fix approach: Use immutable return types. Consider shorter TTL (e.g., 1800s). Document cache invalidation strategy

**Global State in Tax Library:**
- Issue: `app/core/tax_library.py` uses module-level `global _TAX_TABLES` and `_CAP_GAINS_RATES` that are mutated on first access
- Files: `app/core/tax_library.py` (lines 16, 146)
- Impact: If multiple threads access tax_library concurrently during initialization, race conditions possible. Tax tables could be partially initialized
- Fix approach: Use lazy initialization with locks or move to class-based singleton pattern

**Hardcoded Paths and Cwd Dependencies:**
- Issue: Multiple files use relative paths that depend on current working directory being correct
- Files:
  - `app/services/xray_engine.py` (lines 13, 29, 33)
  - `app/services/data_service.py` (lines 29, 36)
  - `app/common/utils.py` (lines 32-33, 107)
- Impact: Code breaks if run from different directory. IDE/test runners may fail due to CWD assumptions
- Fix approach: Use `__file__` + os.path.dirname() for all relative paths. Create a config module for paths

**Pickle-Based Caching (Security Risk):**
- Issue: `app/services/testfol_api.py` uses pickle for caching API responses without validation
- Files: `app/services/testfol_api.py` (lines 7, 59, 159, 189)
- Impact: Malicious pickle files could execute arbitrary code. Cache corruption handling is basic (deletes file, retries). No integrity checks
- Fix approach: Use JSON-based caching for API responses. Or add pickle validation/signing. Implement cache versioning

---

## Known Bugs

**Margin Calculation Edge Case (Division by Zero Avoided, But Fragile):**
- Symptoms: When portfolio value approaches zero, margin usage calculations may produce Inf or NaN
- Files: `app/services/testfol_api.py` (lines 328, 293-300)
- Trigger: Set very high margin draw or loss that reduces portfolio to near-zero
- Workaround: Code has `if current_loan > 1e-9` check at line 293, but downstream consumers may not handle NaN gracefully

**Empty Series Handling Inconsistent:**
- Symptoms: Some functions return 0.0 for empty series, others return NaN, causing inconsistent metric displays
- Files: `app/core/calculations.py` (lines 5, 9, 14, 21)
- Trigger: Clipping data to date range that produces empty result
- Workaround: Check if series is empty before display, but UI may show blank or "0%" inconsistently

**NDX Simulation Splice Scaling Bug:**
- Symptoms: When splicing NDXMEGASIM historical data with QBIG live data, scale factor may fail if sim_end_val is 0
- Files: `app/services/data_service.py` (lines 94-96)
- Trigger: Load NDXMEGASIM with corrupted CSV where last value is 0
- Current: Has fallback `scale_factor = 1.0 if sim_end_val != 0 else 1.0` (always 1.0, no actual fallback)
- Fix approach: Check for zero explicitly and raise informative error or skip scaling

**Timezone Handling Inconsistency:**
- Symptoms: Data from different sources (yfinance, Testfol API, local CSV) may have different timezone awareness
- Files: `app/services/data_service.py` (lines 140-142)
- Trigger: Mix of timezone-aware and naive datetime indices causes alignment failures
- Current: Only yfinance is explicitly localized to naive, but other sources aren't checked

---

## Security Considerations

**External API Dependency Without Rate Limiting:**
- Risk: Testfol API calls could be rate-limited or blocked. No request queuing or backoff strategy for production scale
- Files: `app/services/testfol_api.py` (lines 115-132)
- Current mitigation: Disk caching with MD5 hash, exponential backoff on retry (5 retries, 2s-32s)
- Recommendations:
  - Add circuit breaker pattern to fail fast after N consecutive failures
  - Implement request queue with rate limiting
  - Add metrics/logging for API health

**Bearer Token in Environment Variable (XRay - Not Directly Exposed):**
- Risk: If `TESTFOL_API_KEY` env var is logged or dumped, token exposed
- Files: `app/services/testfol_api.py` (line 107)
- Current mitigation: Token not printed in debug output, only cache hash
- Recommendations: Never log full token. Log only last 4 chars. Detect if token is hardcoded in requests

**ETF Holdings Fetcher Network Dependency:**
- Risk: `etf_holdings_fetcher` (not shown) makes network calls but provides no timeout or retry logic that we control
- Files: `app/services/xray_engine.py` (lines 144, 184-186)
- Current mitigation: ThreadPoolExecutor with max 10 workers limits concurrency
- Recommendations: Add explicit timeout to ThreadPoolExecutor. Handle connection timeouts gracefully

**User Input Validation Gaps:**
- Risk: Streamlit `st.number_input` accepts any float, but some fields have no minimum bounds
- Files: `app/ui/configuration.py` (lines 308, 312, 339, 344, 372, 382)
- Trigger: User enters negative loan amount or negative income, causing downstream calculation errors
- Current: `max(0, ...)` in some places, but not all
- Recommendations: Add explicit min/max bounds to all number_input calls. Validate tier spreads are non-negative

**JSON File Handling (Presets) Without Size Limits:**
- Risk: User could craft large JSON payload to exhaust memory or disk
- Files: `app/common/utils.py` (lines 35-42, 45-69)
- Impact: No file size check on `json.load()` or write operations
- Recommendations: Add max file size check (e.g., 1MB). Validate JSON schema before accepting

---

## Performance Bottlenecks

**Large Charts File (3104 lines):**
- Problem: `app/ui/charts.py` is monolithic with multiple charting functions, making it slow to import
- Files: `app/ui/charts.py`
- Cause: Each chart function recreates plotly figures. No memoization of expensive calculations (e.g., heatmap z-values)
- Improvement path:
  - Split into separate modules (e.g., `charts/performance.py`, `charts/heatmap.py`)
  - Cache intermediate calculations within chart functions
  - Use st.cache_data for Plotly figure serialization

**Monte Carlo Simulation Without Progress Feedback:**
- Problem: Long-running simulations (100+ iterations) block UI with no progress indication
- Files: `app/ui/results.py` (lines 943-1070), `app/core/monte_carlo.py`
- Cause: NumPy array operations in tight loop, no streaming progress
- Improvement path:
  - Add `st.progress()` callback during simulation
  - Consider using numpy.random.Generator with explicit seed for determinism
  - Optimize array creation (pre-allocate vs append)

**N-Depth ETF X-Ray Expansion:**
- Problem: `compute_xray()` recursively expands ETF holdings but has max_depth=2 hardcoded limit to prevent runaway
- Files: `app/services/xray_engine.py` (lines 106-251)
- Cause: Each level of ETF nesting triggers network fetch for holdings. 10 workers on ThreadPoolExecutor can create 100+ concurrent requests
- Improvement path:
  - Add caching of ETF holdings (e.g., disk cache with 7-day TTL)
  - Implement dynamic depth based on portfolio weight (skip small positions)
  - Pre-fetch in background, show progressive results

**No Result Pagination for Large Reports:**
- Problem: If portfolio has 500+ holdings in X-Ray view, all rows rendered at once
- Files: `app/ui/xray_view.py`
- Cause: Streamlit DataFrames not paginated
- Improvement path: Implement server-side pagination or use AgGrid for scrolling

---

## Fragile Areas

**Tax Calculation Module:**
- Files: `app/core/tax_library.py`
- Why fragile: Complex nested tax logic with 700+ lines of rules spread across multiple functions. Loss carryforwards have state (line 681-704)
- Safe modification: All tax calculation changes require extensive testing with multiple filing statuses and income levels. Create comprehensive test suite with known IRS examples
- Test coverage: No test files found for tax calculations. High risk area

**Rebalancing Event Processing:**
- Files: `app/core/calculations.py` (lines 88-170)
- Why fragile: Consumes API response `rebalancing_events` structure which could change. Cost basis tracking uses in-place list mutations
- Safe modification: Document expected structure of `rebalancing_events`. Add schema validation. Unit test with synthetic events
- Test coverage: No test files found

**Margin Loan Simulation:**
- Files: `app/services/testfol_api.py` (lines 195-329)
- Why fragile: Tiered rate calculation uses loop with manual balance tracking. Variable rate models have multiple code paths (Fixed/Variable/Tiered)
- Safe modification: Each rate mode should have its own class or function. Add property-based tests for edge cases (zero loan, negative rates, etc.)
- Test coverage: No test files found

**Data Source Multiplexing:**
- Files: `app/services/data_service.py` (lines 9-170)
- Why fragile: Falls back through multiple sources (Yahoo -> Testfol -> Local CSV). Each has different error handling
- Safe modification: Create abstraction layer for data sources. Define interface contract for each source
- Test coverage: No test files found

---

## Scaling Limits

**Pickle Cache Directory Growth:**
- Current capacity: No cleanup of old cache files. MD5 hashes are deterministic but cache dir grows unbounded
- Files: `app/services/testfol_api.py` (line 50)
- Limit: Disk space. After running for 1 year with unique backtests, cache dir could exceed 1GB+
- Scaling path:
  - Add cache size limit (e.g., keep 500 most recent files)
  - Implement LRU eviction on cache_dir
  - Monitor cache directory size

**Session State with Multiple Portfolios:**
- Current capacity: UI supports max 5 portfolios hardcoded
- Files: `app/ui/configuration.py` (lines 95, 133, 201), `app/ui/ndx_scanner.py` (line 339)
- Limit: Each portfolio triggers full backtest. 5 portfolios x 1000+ days = ~5000 calculations per page load
- Scaling path:
  - Remove hardcoded 5-portfolio limit
  - Add lazy loading (only calculate visible portfolio)
  - Cache individual portfolio results separately

**API Response Caching Size:**
- Current capacity: No limit on pickle file size. Large allocations with 50+ tickers generate responses >10MB
- Files: `app/services/testfol_api.py` (lines 158-189)
- Limit: Memory exhaustion if many cache files loaded simultaneously
- Scaling path:
  - Implement streaming JSON deserialization
  - Add response size check before pickling
  - Use chunked serialization for large responses

**Streamlit Session State Memory:**
- Current capacity: All portfolios, cache, and data stored in `st.session_state`
- Files: `app/ui/configuration.py` (lines 20-52), `app/ui/asset_explorer.py` (lines 83-92)
- Limit: Streamlit memory grows with # of active sessions. No cleanup of old state
- Scaling path:
  - Move persistent data to database (SQLite or DuckDB)
  - Implement session timeout with explicit cleanup
  - Use `@st.cache_resource` for singleton data

---

## Dependencies at Risk

**yfinance (Unstable Scraping):**
- Risk: Relies on web scraping Yahoo Finance. Breakage common on Yahoo HTML changes
- Impact: All real-ticker data fetches fail, fallback to Testfol API (less historical data)
- Current usage: `app/services/data_service.py` (lines 127-145, 63)
- Migration plan: Consider switching to `pandas_datareader` + FRED for Fed data, or paid API (Alpha Vantage, Polygon)

**Streamlit (UI Coupling):**
- Risk: Heavy dependency on Streamlit internals (session_state, cache decorators). Version upgrades may break
- Impact: Multiple references to `@st.cache_data`, `st.session_state` throughout codebase
- Current usage: Nearly all UI modules
- Migration plan: Abstract Streamlit dependency behind service layer. Use FastAPI backend + React frontend if scalability needed

**requests Library (No Pin):**
- Risk: `requirements.txt` lists `requests` without version, could break due to API changes
- Impact: HTTP retry behavior, timeout behavior could change
- Current usage: `app/services/testfol_api.py` (line 1), `app/services/data_service.py` (line 194)
- Migration plan: Pin to `requests>=2.31.0,<3.0`. Add requests-cache for built-in HTTP caching

**Altair Version Pin Issues:**
- Risk: Recent requirement: "Pin altair<5 for Streamlit compatibility" (commit eada8f5)
- Impact: Older versions may have security issues. Streamlit compatibility may break with future versions
- Current usage: Indirectly through plotly
- Migration plan: Monitor Streamlit/altair compatibility. Consider moving from Plotly to alternative

---

## Missing Critical Features

**No Input Validation on API Authentication:**
- Problem: Bearer token accepted without format validation. Could send invalid token silently, fail at API call
- Files: `app/services/testfol_api.py` (lines 107-113)
- Blocks: Users with auth issues get cryptic HTTP errors instead of clear "Invalid token format" message

**No Data Integrity Checks:**
- Problem: API responses accepted without schema validation. Missing fields cause KeyError downstream
- Files: `app/services/testfol_api.py` (lines 162-169)
- Blocks: Robust error handling for API schema changes

**No Offline Mode:**
- Problem: Application requires constant connectivity. No graceful degradation if API is down
- Files: `app/services/testfol_api.py`, `app/services/data_service.py`
- Blocks: Users cannot view cached results without network

**No Audit Logging:**
- Problem: Tax calculations and margin adjustments are not logged for compliance review
- Files: `app/core/tax_library.py`, `app/services/testfol_api.py`
- Blocks: Financial professionals cannot audit what was calculated

---

## Test Coverage Gaps

**Tax Calculation Logic Untested:**
- What's not tested: All 700+ lines of `app/core/tax_library.py`. Multiple tax scenarios (long-term, short-term, collectibles, NIIT)
- Files: `app/core/tax_library.py`
- Risk: Tax calculations could be incorrect for edge cases (e.g., income exactly at bracket threshold, large capital gains)
- Priority: HIGH - This is core functionality for financial users

**Margin Loan Simulation Untested:**
- What's not tested: Tiered rate calculations, loan evolution over time, edge cases (zero loan, negative draws)
- Files: `app/services/testfol_api.py` (lines 195-329)
- Risk: Margin calculations could be wrong, giving users false sense of portfolio risk
- Priority: HIGH - Users rely on this for portfolio decisions

**Data Splicing Logic Untested:**
- What's not tested: NDXMEGASIM + QBIG splicing. Timezone alignment. Empty data handling
- Files: `app/services/data_service.py` (lines 84-110)
- Risk: Spliced data could have gaps, discontinuities, or be misaligned chronologically
- Priority: HIGH - Affects historical backtest accuracy

**Rebalancing Event Processing Untested:**
- What's not tested: `process_rebalancing_data()` with real-world allocation changes
- Files: `app/core/calculations.py` (lines 88-170)
- Risk: Realized P&L or tax implications could be calculated incorrectly
- Priority: MEDIUM - Affects tax calculations

**ETF X-Ray Expansion Untested:**
- What's not tested: `compute_xray()` with nested ETFs, missing data, alias resolution
- Files: `app/services/xray_engine.py` (lines 106-251)
- Risk: Holdings aggregation could be wrong, weights don't sum to 100%
- Priority: MEDIUM - Informational but could mislead users

**No Integration Tests:**
- What's not tested: End-to-end flow (load portfolio -> fetch data -> calculate stats -> render charts)
- Risk: Changes to one layer could silently break another layer
- Priority: MEDIUM - Would catch regression bugs

---

*Concerns audit: 2026-01-23*

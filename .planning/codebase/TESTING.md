# Testing Patterns

**Analysis Date:** 2026-01-23

## Test Framework Status

**Python:**
- No formal test framework configured
- No pytest, unittest, or nose in `requirements.txt`
- No `pytest.ini`, `setup.cfg`, or `conftest.py`

**TypeScript/React:**
- No test framework configured
- No jest, vitest, or testing-library in `package.json`
- No test files in mobile app directory

## Testing Approach

The codebase uses **manual, scripted testing** rather than automated unit or integration tests.

### Python Test Pattern

**Location:** `debug_tools/` directory
- `debug_tools/test_*.py` files contain manual verification scripts
- `debug_tools/tax_verification/test_*.py` - Tests for tax calculation logic
- Example files: `test_dca.py`, `test_margin_sim.py`, `test_qqqsim_cagr.py`

**Test File Organization:**
```
debug_tools/
├── test_dca.py                  # Manual DCA logic verification
├── test_margin_sim.py           # Margin simulation tests
├── test_qqqsim_cagr.py         # CAGR calculation tests
├── tax_verification/
│   ├── test_dca.py             # DCA with tax logic
│   ├── test_deduction_toggle.py
│   ├── test_hoh.py
│   ├── test_loss_deduction.py
│   ├── test_smart_modern.py
│   ├── test_standard_deduction.py
│   ├── test_tax_lots.py
│   └── test_2000s_tax.py
├── api_debug/
│   ├── test_api_offset.py
│   ├── inspect_api_response.py
│   └── inspect_raw.py
└── excel_debug/
    ├── inspect_excel.py
    ├── inspect_excel_headers.py
    └── inspect_sheet1.py
```

**Run Commands:**
```bash
# No standard test runner - execute scripts directly
python debug_tools/test_margin_sim.py
python debug_tools/test_dca.py
python debug_tools/tax_verification/test_dca.py
```

### Test Structure (Python)

**Typical Pattern:**
```python
import sys
import os
import pandas as pd
import numpy as np
from unittest.mock import MagicMock

# Add parent directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import shadow_backtest

def test_dca_logic():
    print("--- Testing DCA Logic ---")

    # Mock data setup
    dates = pd.date_range(start="2023-01-01", end="2023-12-31", freq="B")
    prices = pd.DataFrame(index=dates)
    prices["TEST"] = 100.0

    # Monkeypatch with mock
    original_fetch = shadow_backtest.fetch_prices
    shadow_backtest.fetch_prices = MagicMock(return_value=(prices, "Mocked Data"))

    try:
        # Run function under test
        allocation = {"TEST": 1.0}
        trades, pl, comp, unrealized, logs = shadow_backtest.run_shadow_backtest(
            allocation=allocation,
            start_val=10000.0,
            start_date="2023-01-01",
            end_date="2023-12-31",
            rebalance_freq="Yearly",
            cashflow=1000.0,
            cashflow_freq="Monthly"
        )

        # Manual assertions
        dca_trades = trades[trades["Realized P&L"] == 0]
        final_comp = comp[comp["Date"] == comp["Date"].max()]
        final_value = final_comp["Value"].sum()

        print(f"Number of DCA Injections: {len(dca_trades)}")
        print(f"Final Value: ${final_value:,.2f}")

        # Assert
        expected_final = 10000 + (len(dca_trades) * 1000)
        assert abs(final_value - expected_final) < 0.01
        print("✓ Test passed")

    finally:
        # Restore
        shadow_backtest.fetch_prices = original_fetch

if __name__ == "__main__":
    test_dca_logic()
```

### Mocking

**Framework:** `unittest.mock` (standard library)

**Patterns:**
```python
from unittest.mock import MagicMock

# Monkeypatch module function
original_fetch = shadow_backtest.fetch_prices
shadow_backtest.fetch_prices = MagicMock(return_value=(prices, "Mocked Data"))

try:
    # ... test code ...
finally:
    # Restore original
    shadow_backtest.fetch_prices = original_fetch
```

**What to Mock:**
- External data fetches: `fetch_prices()`, `yfinance` API calls
- API responses: `testfol_api` calls
- File I/O: `pd.read_csv()` calls in data_service.py

**What NOT to Mock:**
- Core calculation functions (calculations.py, tax_library.py)
- Data structures and transformations
- Business logic (shadow_backtest, monte_carlo)

### Fixtures and Factories

**Test Data Pattern:**
```python
# Inline data generation (no shared fixtures)
dates = pd.date_range(start="2023-01-01", end="2023-12-31", freq="B")
prices = pd.DataFrame(index=dates)
prices["TEST"] = 100.0
```

**Location:** Each test script defines its own data (no centralized fixtures)

**Typical test data:**
- Date ranges: `pd.date_range(start="2023-01-01", periods=100)`
- Constant portfolio: `pd.Series(100000.0, index=dates)`
- Simple price data: `pd.DataFrame({"Close": [100, 101, 102, ...]})`

### Coverage

**Requirements:** Not enforced, no coverage.rc or configuration

**View Coverage:** Not applicable (no coverage tool configured)

**Coverage Status:** Unknown - tests are manual verification, no coverage metrics

---

## Test Types in Codebase

### Manual Verification Tests

**Purpose:** Verify specific calculations and logic before deployment

**Scope:** Component-level testing of core functions

**Examples:**
- `debug_tools/test_dca.py` - DCA injection and tax lot tracking
- `debug_tools/tax_verification/test_tax_lots.py` - Tax lot management
- `debug_tools/test_margin_sim.py` - Margin loan simulation logic

**Structure:**
1. Set up test data
2. Call function under test
3. Print results with f-strings
4. Simple assertions (`assert value == expected`, `assert abs(a - b) < 0.01`)

### Integration Tests (Manual)

**Type:** End-to-end backtest verification

**Examples:**
- `debug_tools/verify_testfol_api.py` - API response validation
- `debug_tools/api_debug/test_api_offset.py` - Offset calculation validation
- `debug_tools/test_qqqsim_cagr.py` - Full CAGR calculation across ETFs

**Pattern:** Mock external data, run full backtest, verify metrics match expected values

### No E2E Tests

**Status:** Not used in codebase
- Streamlit app not tested via Selenium, Playwright, etc.
- Mobile app not tested via Detox, Appium, etc.
- API endpoints not tested via HTTP client integration tests

---

## Common Patterns

### Async Testing

**Not applicable** - No async/await in test suite (Python is synchronous for backtests, React Native uses AsyncStorage but not tested)

### Error Testing

**Pattern:** Verify error conditions via prints/assertions

```python
# Verify empty series handling
if series.empty:
    return 0.0

# Test:
assert calculate_cagr(pd.Series([])) == 0.0
```

**Common error cases tested:**
- Empty DataFrames/Series
- Division by zero (returns 0.0 or default)
- Missing data (forward fill, interpolation)
- Invalid date ranges (fallback to full data)

### Data Validation Testing

**Approach:** Verify DataFrame structure and values

```python
# Check trades structure
dca_trades = trades[trades["Realized P&L"] == 0]
assert "Date" in trades.columns
assert "Ticker" in trades.columns
assert "Trade Amount" in trades.columns

# Verify value ranges
assert all(comp["Value"] >= 0)
```

---

## Testing Gaps & Observations

### What's Not Tested

1. **Unit Tests:** No isolated function testing
   - Calculations are tested only within full backtest context
   - UI components have no test coverage
   - API endpoints have no integration tests

2. **Regression Tests:** No recorded test cases for known bugs
   - Manual verification only
   - No CI/CD pipeline to prevent regressions

3. **Mobile App Testing:** Zero test coverage
   - React Native components not tested
   - AsyncStorage operations not verified
   - API communication not mocked/tested

4. **Edge Cases:** Limited edge case coverage
   - High leverage scenarios assumed to work
   - Tax edge cases (loss carryforwards, AMT) minimally tested
   - Date boundary conditions not systematically tested

### Testing Infrastructure

**Missing:**
- CI/CD pipeline (no GitHub Actions, GitLab CI, etc.)
- Test runner configuration
- Coverage measurement tools
- Automated test execution on commits

**Current Status:** Tests are developer scripts, not part of release process

---

## Running Tests

### Python Tests

**Manual execution:**
```bash
# From project root
python debug_tools/test_margin_sim.py
python debug_tools/test_dca.py
python debug_tools/tax_verification/test_dca.py

# Output: prints to console, assertions fail with traceback
```

**Common issues:**
- Path-dependent import failures (requires correct sys.path setup)
- Data file dependencies (CSVs in `data/` directory)
- Mock restoration important (finally blocks)

### No Automated Test Execution

```bash
# These commands do NOT work (no test framework):
pytest              # Not installed
python -m pytest    # Not configured
npm test           # Not configured (React)
```

---

## Test Recommendations for Future Implementation

If adding automated tests:

1. **Python:** Adopt pytest
   - Fixture-based setup
   - Parametrized tests for multiple scenarios
   - Mock library for data fetches
   - Coverage.py for metrics

2. **TypeScript/React:** Adopt Vitest or Jest
   - Component snapshot tests (UI stability)
   - Context provider tests (AsyncStorage mocking)
   - Mock yfinance/API calls
   - React Testing Library for interaction tests

3. **Integration:** Add HTTP client tests for API
   - FastAPI TestClient
   - Mock database responses
   - Verify error handling

4. **CI/CD:** GitHub Actions workflow
   - Run tests on PR/commit
   - Generate coverage reports
   - Block merge on test failure


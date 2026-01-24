# Coding Conventions

**Analysis Date:** 2026-01-23

## Project Scope

This is a hybrid Python + TypeScript/React codebase:
- **Backend/Analysis:** Python (Streamlit app, FastAPI, data processing)
- **Mobile:** TypeScript/React Native (Expo framework)

---

## Python Conventions

### Naming Patterns

**Files:**
- Lowercase with underscores: `calculations.py`, `data_service.py`, `test_margin_sim.py`
- Purpose-driven names: `testfol_charting.py`, `shadow_backtest.py`, `monte_carlo.py`

**Functions:**
- Snake_case throughout: `calculate_cagr()`, `sync_equity()`, `fetch_component_data()`
- Descriptive verb-first pattern: `simulate_margin()`, `resample_data()`, `load_tax_tables()`
- Prefixed test functions: `test_dca_logic()` (in debug_tools)

**Variables:**
- Snake_case: `starting_loan`, `margin_rate`, `daily_rate`, `tax_series`
- Shorthand in loops/iterations: `p` (portfolio), `d` (date), `idx` (index)
- Prefix for cached/memoized: `cached_fetch_backtest()`, `cached_run_shadow_backtest_v2()`

**Types/Classes:**
- PascalCase for Pydantic models: `BacktestRequest`, `XRayRequest`, `MonteCarloRequest` (in `backend.py`)
- Data structures as TypedDict or direct dict/DataFrame usage (no explicit type hints for most data objects)

### Code Style

**Formatting:**
- No explicit formatter configured (no black, isort in requirements)
- Observed style: 2-4 space indentation (mixed), line length ~100-120 characters
- Single quotes preferred for strings (when quoted)

**Linting:**
- No eslint/pylint configured in Python
- ESLint enabled for JavaScript (expo lint command in package.json)
- ESLint extends expo config: `eslint-config-expo` in `margin-stresser-mobile/eslint.config.js`

### Import Organization

**Order:** (observed pattern)
1. Standard library imports: `import sys`, `import os`, `import pandas as pd`, `import numpy as np`, `from datetime import date, timedelta`
2. Third-party imports: `import streamlit as st`, `import yfinance as yf`, `from fastapi import FastAPI`, `from pydantic import BaseModel`
3. Local imports: `from app.services import testfol_api as api`, `from app.core import calculations`, `from . import asset_explorer`

**Path Aliases:**
- Relative imports within package: `from . import sidebar`, `from . import charts`
- Absolute imports from package root: `from app.common import utils`, `from app.services import fetch_backtest`
- No path aliases configured (standard Python module structure)

### Error Handling

**Patterns:**
- Broad `try/except` blocks with `Exception` catch:
  ```python
  try:
      df_sim = pd.read_csv(csv_path)
  except (pd.errors.ParserError, pd.errors.EmptyDataError, ValueError) as e:
      st.warning(f"Corruption detected in {base}.csv ({e})")
  ```
- Silent fallback with `pass`:
  ```python
  except Exception:
      pass  # Fallback to full series if parsing fails
  ```
- Streamlit-specific error display: `st.error()`, `st.warning()`
- HTTPException for API errors: `raise HTTPException(status_code=500, detail=str(e))`

**Common patterns:**
- Check for empty data: `if series.empty: return 0.0`
- Check for zero/None: `if std == 0: return 0.0`, `if value != 0`
- Graceful degradation with warnings (Streamlit UI)

### Logging

**Framework:** `print()` and `st.*` (Streamlit logging)

**Patterns:**
- Debug prints with f-strings: `print(f"DEBUG: Running Monte Carlo for returns list of length {len(req.returns)}")`
- Streamlit status messages: `st.warning()`, `st.error()`, `st.success()`
- Spinner context: `with st.spinner("Rebuilding Simulation Data..."):`
- Caching wrappers with spinners: `@st.cache_data(show_spinner="Fetching data from Testfol API...", ttl=3600)`

### Comments

**When to Comment:**
- Docstrings on public functions: `"""Cached wrapper for api.fetch_backtest"""`
- Inline comments for complex math/logic:
  ```python
  # B_t = Loan_{t-1} * (r_asset - r_loan) - Draws - Taxes
  loan_component = loan_series * (asset_returns - daily_rate)
  ```
- Section dividers: `# --- Validation & Run ---`, `# --- Sidebar ---`
- Sparse in data processing (self-documenting variable names)

**JSDoc/TSDoc:**
- Not used in Python
- Type hints are minimal (some function signatures use basic types)

### Function Design

**Size:** Functions range from 5-100+ lines
- Short utility functions: `color_return()` (3 lines), `sync_equity()` (4 lines)
- Medium processing: `calculate_tax_adjusted_equity()` (~45 lines), `fetch_component_data()` (~140 lines)
- Large orchestration: `run_shadow_backtest()` (200+ lines) - combines multiple operations

**Parameters:**
- Positional + keyword args mixed
- Default values common: `def run_monte_carlo(returns_series, n_sims=1000, n_years=10, initial_val=10000, monthly_cashflow=0.0, ...)`
- Dict/DataFrame for complex data structures (no dedicated data classes except Pydantic models in API)

**Return Values:**
- Multiple returns via tuple: `return loan_series, equity, equity_pct, usage_pct`
- DataFrame/Series from data functions
- Dict from API endpoints: `{"status": "ok", "service": "Margin Stresser API"}`
- Tuple unpacking common: `trades, pl, comp, unrealized, logs = shadow_backtest.run_shadow_backtest(...)`

### Module Design

**Exports:**
- No explicit `__all__` used
- All public functions implicitly exported
- Private functions use leading underscore: `_TAX_TABLES = {}`

**Barrel Files:**
- Not used in Python (standard module imports instead)
- UI submodule pattern: `from app.ui import render_sidebar, render_config, render_results`

---

## TypeScript/React (Mobile) Conventions

### Naming Patterns

**Files:**
- PascalCase for components: `ThemedText.tsx`, `AppContext.tsx`, `Collapsible.tsx`
- kebab-case for hooks: `use-theme-color.ts`, `use-color-scheme.ts`
- Index files: `_layout.tsx` (Expo router convention), `index.tsx`

**Functions/Components:**
- PascalCase for React components: `HomeScreen`, `ThemedText`, `AppProvider`
- camelCase for hooks: `useAppContext()`, `useThemeColor()`, `useColorScheme()`
- camelCase for utility functions: `loadScenario()`, `addScenario()`, `deleteScenario()`

**Variables:**
- camelCase: `portfolioValue`, `marginDebt`, `interestRate`, `maintenanceMargin`
- Prefixed for state/context: `activeScenarioToLoad`, `storedScenarios`, `storedHistory`
- Styled components: `StyledView`, `StyledText`, `StyledScrollView` (using nativewind)

**Types:**
- PascalCase for types: `Ticker`, `SavedScenario`, `HistoryItem`, `AppSettings`
- `Type` suffix for type definitions: `AppContextType`, `ResultType`
- Union types for state: `type='charts' | 'analysis'`

### Code Style

**Formatting:**
- TypeScript strict mode enabled: `"strict": true` in `tsconfig.json`
- Path aliases configured: `"@/*": ["./*"]` for root imports
- Babel preset: `babel-preset-expo` with `nativewind/babel` plugin
- No prettier/ESLint config file (uses expo defaults)

**Linting:**
- ESLint configured via `eslint.config.js`
- Extends: `eslint-config-expo` (Expo default rules)
- Ignores: `dist/*`

### Import Organization

**Order:**
1. React/React Native: `import React`, `from 'react'`, `from 'react-native'`
2. Third-party libraries: `import AsyncStorage from '@react-native-async-storage/async-storage'`
3. Local components/hooks: `from '@/components/ui/icon-symbol'`, `from '../context/AppContext'`
4. Constants: `from '@/constants/theme'`

**Path Aliases:**
- `@/*` maps to project root: `import { IconSymbol } from '@/components/ui/icon-symbol'`
- Relative imports for local hierarchy: `from '../context/AppContext'`, `from './settings'`

### Error Handling

**Patterns:**
- Try/catch with `console.error()`:
  ```typescript
  try {
      const storedScenarios = await AsyncStorage.getItem('scenarios');
      if (storedScenarios) setScenarios(JSON.parse(storedScenarios));
  } catch (e) {
      console.error("Failed to load data", e);
  }
  ```
- Alert dialogs for user errors: `Alert.prompt()`, `Alert.alert()`
- Silent fallback: `const theme = useColorScheme() ?? 'light';`

### Comments

**When to Comment:**
- Function documentation above function signature
- Inline only for non-obvious logic
- Section markers: `// Load Scenario Effect`, `// Define result type`
- Sparse in UI components (self-documenting JSX)

### Function Design

**Size:** Components typically 80-200 lines
- Small presentational: `ThemedText()` (30 lines), `Collapsible()` (45 lines)
- Screen components: `HomeScreen()` (200+ lines, complex state management)

**Parameters:**
- React Props: `PropsWithChildren & { title: string }`
- Hooks return: `[state, setState]` pattern
- Callback functions passed as props

**Return Values:**
- JSX elements: `return <Text ... />`
- Hooks return: `[scenarios, setScenarios]` (React state)
- Context returns tuple/object with state and methods

### Module Design

**Exports:**
- Named exports for components: `export function ThemedText(...) { ... }`
- Named exports for hooks: `export const useAppContext = () => { ... }`
- Export statement at end: `export const AppProvider: React.FC<...> = ...`

**Styled Components:**
- NativeWind integration: `const StyledView = styled(View);`
- StyleSheet objects: `const styles = StyleSheet.create({ ... })`
- Inline styles for dynamic values: `style={{ transform: [{ rotate: isOpen ? '90deg' : '0deg' }] }}`

---

## Cross-Codebase Patterns

### Data Structures

**Python:**
- pandas DataFrame/Series for time series and tables
- Dict for flexible config: `{"type": "Tiered", "tiers": [...]}`
- Pydantic BaseModel for API contracts only (backend.py)

**TypeScript:**
- Typed interfaces: `SavedScenario`, `HistoryItem`
- State objects: `{ portfolioValue: string, marginDebt: string, ... }`
- Context objects: `{ scenarios, history, settings, methods... }`

### Dependencies Injection / Config

**Python:**
- Environment-based settings (Streamlit session_state)
- Passed as function arguments: `fetch_backtest(*args, **kwargs)`
- Module-level globals with lazy loading: `_TAX_TABLES = {}`

**TypeScript:**
- React Context API for state: `AppContext`, `AppProvider`
- Hooks for accessing context: `useAppContext()`
- Props drilling for simple state

### Testing Organization

**Python:**
- Debug tools in separate `debug_tools/` folder
- Test files named `test_*.py` or `*_test.py`
- Manual testing pattern (not unit tests): Functions called directly with assertions
- No test framework configured (pytest, unittest not in requirements)

**TypeScript:**
- No test files in codebase
- No testing framework configured (jest, vitest not in package.json)

---

## Conventions Summary

| Aspect | Python | TypeScript/React |
|--------|--------|------------------|
| **Naming** | snake_case | camelCase (vars), PascalCase (components) |
| **File names** | lowercase_underscore | PascalCase (components), kebab-case (hooks) |
| **Classes/Types** | CamelCase (Pydantic only) | PascalCase (always) |
| **Error handling** | Broad try/except, Streamlit UI | try/catch, Alert dialogs |
| **Logging** | print(), st.* | console.error() |
| **Import order** | std lib → 3rd party → local | React → 3rd party → local |
| **Comments** | Section dividers, docstrings | Section markers, sparse |
| **Formatting** | Unformatted, ~100-120 char lines | ESLint (expo config), strict TS |


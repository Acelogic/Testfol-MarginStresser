# Debug Tools Reference

**Live App**: [testfol-marginstresser.streamlit.app](https://testfol-marginstresser.streamlit.app/)

This directory contains debugging and testing utilities for the Testfol MarginStresser project, organized by category.

## Directory Structure

### `api_debug/` - API Inspection & Debugging
Utilities for inspecting `testfol_api` responses and debugging data fetching.

| File | Purpose |
|------|---------|
| `inspect_api_response.py` | Fetches backtest data and dumps raw JSON to `api_dump.json` |
| `api_dump.json` | JSON dump of API responses |
| `verify_dump.py` | Verifies contents of `api_dump.json` |
| `inspect_raw.py` | Comprehensive inspection of raw API responses |
| `inspect_rebal.py` | Inspects rebalancing events and statistics |

### `excel_debug/` - Excel Data Inspection
Utilities for inspecting and verifying the Federal Capital Gains Tax Excel file.

| File | Purpose |
|------|---------|
| `list_sheets.py` | Lists all sheet names in the Excel file |
| `find_header.py` | Locates data start in Sheet1 |
| `inspect_sheet1.py` | Inspects header row location in Sheet1 |
| `inspect_excel.py` | Previews all sheets |
| `inspect_excel_headers.py` | Locates header in "Historical Capital Gains Rates" sheet |
| `verify_headers.py` | Verifies correct header row selection |
| `verify_excel_data.py` | Verifies specific years and rates exist |

### `tax_verification/` - Tax & Backtest Verification
Scripts for verifying tax calculations, loss carryforward, and shadow backtest logic.

| File | Purpose |
|------|---------|
| `test_tax_lots.py` | **Key Script**: Verifies FIFO logic and ST/LT gain distinction |
| `verify_fix_st_lt.py` | **Key Script**: Verifies that monthly rebalancing generates ST gains |
| `test_2000s_tax.py` | Tests historical tax calculations for the 2000s era |
| `repro_loss_carryforward.py` | Tests loss carryforward logic |
| `trace_2022.py` | Detailed trace of rebalancing events for 2022 |
| `shadow_backtest.log` | Log file from shadow backtest runs |

## Quick Reference

### Running Tests
All scripts should be run from the **project root** directory to ensure imports work correctly.

```bash
# Verify Short-Term vs Long-Term Fix
python debug_tools/tax_verification/verify_fix_st_lt.py

# Test Tax Lot FIFO Logic
python debug_tools/tax_verification/test_tax_lots.py

# Inspect API Response
python debug_tools/api_debug/inspect_api_response.py
```

## Notes
- Scripts import modules (`tax_library`, `shadow_backtest`, `testfol_api`) from the parent directory.
- Ensure you run these from the root of the repo (e.g., `python debug_tools/subdir/script.py`).

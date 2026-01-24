# API Feature Parity Design

**Date:** 2026-01-24
**Status:** Approved

## Overview

Add new FastAPI endpoints to achieve feature parity between the mobile app and the Streamlit web application. Endpoints are organized as feature bundles aligned with mobile app screens.

## New Endpoints

### 1. `POST /chart_data`

**Purpose:** Powers chart overlay screens with technical analysis data.

**Request Body:**
```python
class ChartDataRequest(BaseModel):
    # Option A: Pass existing series
    portfolio_series: Optional[List[List]] = None  # [[timestamps], [values]]

    # Option B: Fetch fresh backtest
    tickers: Optional[Dict[str, float]] = None
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    rebalance_freq: str = "Quarterly"

    # Analysis options
    ma_windows: List[int] = [150, 200]
    include_technicals: bool = True
    include_stage: bool = True
    tolerance_days: int = 14  # Whipsaw filter for MA breach detection
```

**Response:**
```python
{
    "ma_analysis": {
        "200": {
            "current_ma": float,
            "price_vs_ma_pct": float,
            "is_above_ma": bool,
            "breach_events": [
                {
                    "start_date": str,
                    "end_date": str,
                    "duration_days": int,
                    "max_depth_pct": float,
                    "bottom_date": str,
                    "recovery_days": int,
                    "breach_return_pct": float,
                    "max_depth_return_pct": float,
                    "spysim_breach_return_pct": float,
                    "spysim_maxdepth_return_pct": float,
                    "breach_alpha_pct": float,
                    "maxdepth_alpha_pct": float,
                    "status": str
                }
            ],
            "summary": {
                "total_events": int,
                "avg_duration_days": float,
                "avg_depth_pct": float,
                "breach_win_rate": float,
                "maxdepth_win_rate": float,
                "avg_breach_alpha": float,
                "avg_maxdepth_alpha": float
            }
        }
    },
    "stage_analysis": {
        "current_stage": str,  # "Stage 2 (Advancing)", etc.
        "ma_slope": float,
        "stage_history": [[timestamps], [stage_codes]]
    },
    "technical_levels": [
        {"price": float, "label": str, "type": str}
    ]
}
```

**Implementation:** Uses `calculations.analyze_ma()`, `calculations.compare_breach_events()`, `calculations.analyze_stage()`, `calculations.calculate_cheat_sheet()`.

---

### 2. `POST /performance`

**Purpose:** Detailed returns analysis and statistics for performance screens.

**Request Body:**
```python
class PerformanceRequest(BaseModel):
    tickers: Dict[str, float]
    start_date: str
    end_date: Optional[str] = None
    start_val: float = 10000.0
    rebalance_freq: str = "Quarterly"
    cashflow: float = 0.0
    cashflow_freq: str = "Monthly"
    invest_div: bool = True

    benchmark: Optional[str] = "SPYSIM"
    include_drawdown_series: bool = True
```

**Response:**
```python
{
    "stats": {
        "cagr": float,
        "sharpe": float,
        "sortino": float,
        "max_drawdown": float,
        "volatility": float,
        "best_year": float,
        "worst_year": float
    },
    "benchmark_stats": {
        "cagr": float,
        "sharpe": float,
        "max_drawdown": float,
        "volatility": float
    },
    "alpha": {
        "cagr_diff": float,
        "sharpe_diff": float,
        "max_dd_diff": float
    },
    "returns": {
        "monthly": [{"year": int, "month": int, "value": float, "benchmark": float}],
        "quarterly": [{"year": int, "quarter": int, "value": float, "benchmark": float}],
        "yearly": [{"year": int, "value": float, "benchmark": float}]
    },
    "drawdown_series": {
        "dates": [str],
        "portfolio": [float],
        "benchmark": [float]
    }
}
```

**Implementation:** Uses `calculations.generate_stats()`, `testfol_api.fetch_backtest()` for benchmark.

---

### 3. `POST /tax_rebal`

**Purpose:** Tax impact analysis and rebalancing history for accounting/planning screens.

**Request Body:**
```python
class TaxRebalRequest(BaseModel):
    tickers: Dict[str, float]
    start_date: str
    end_date: Optional[str] = None
    start_val: float = 10000.0
    rebalance_freq: str = "Quarterly"
    cashflow: float = 0.0
    cashflow_freq: str = "Monthly"

    tax_config: TaxConfig
    include_composition: bool = True
    include_trades: bool = True

class TaxConfig(BaseModel):
    income: float = 100000
    filing_status: str = "Single"  # Single, Married, HeadOfHousehold
    state_tax_rate: float = 0.0
    method: str = "2025_fixed"
    use_std_deduction: bool = True
```

**Response:**
```python
{
    "tax_by_year": [
        {
            "year": int,
            "realized_pl": float,
            "short_term_pl": float,
            "long_term_pl": float,
            "federal_tax": float,
            "state_tax": float,
            "total_tax": float,
            "loss_carryforward": float
        }
    ],
    "tax_summary": {
        "total_tax_paid": float,
        "total_realized_pl": float,
        "effective_rate": float,
        "current_loss_carryforward": float
    },
    "trades": [
        {
            "date": str,
            "ticker": str,
            "action": str,  # "BUY" or "SELL"
            "amount": float,
            "realized_pl": float,
            "holding_period": str  # "short_term" or "long_term"
        }
    ],
    "composition": [
        {
            "date": str,
            "holdings": [{"ticker": str, "value": float, "weight": float}],
            "total_value": float
        }
    ],
    "unrealized_pl": {
        "total": float,
        "by_ticker": [
            {"ticker": str, "cost_basis": float, "current_value": float, "unrealized": float}
        ]
    }
}
```

**Implementation:** Uses `shadow_backtest.run_shadow_backtest()`, `tax_library.calculate_tax_series_with_carryforward()`.

---

### 4. `POST /compare`

**Purpose:** Side-by-side comparison of multiple portfolios.

**Request Body:**
```python
class CompareRequest(BaseModel):
    portfolios: List[PortfolioConfig]
    start_date: str
    end_date: Optional[str] = None
    start_val: float = 10000.0

    benchmark: Optional[str] = "SPYSIM"
    align_to_common_start: bool = True

class PortfolioConfig(BaseModel):
    name: str
    tickers: Dict[str, float]
    rebalance_freq: str = "Quarterly"
```

**Response:**
```python
{
    "common_start_date": str,
    "common_end_date": str,
    "portfolios": [
        {
            "name": str,
            "stats": {
                "cagr": float,
                "sharpe": float,
                "max_drawdown": float,
                "volatility": float,
                "final_value": float
            },
            "series": {"dates": [str], "values": [float]},
            "drawdown": {"dates": [str], "values": [float]}
        }
    ],
    "benchmark": {
        "name": str,
        "stats": {...},
        "series": {...}
    },
    "rankings": {
        "by_cagr": [str],
        "by_sharpe": [str],
        "by_max_drawdown": [str]
    }
}
```

**Implementation:** Uses `testfol_api.fetch_backtest()` for each portfolio, `calculations.generate_stats()`.

---

## Existing Endpoints (Unchanged)

| Endpoint | Purpose |
|----------|---------|
| `GET /` | Health check |
| `POST /run_stress_test` | Main backtest with margin simulation |
| `POST /xray` | ETF holdings decomposition |
| `POST /monte_carlo` | Monte Carlo projection |

---

## Implementation Notes

### Pydantic Models
- All request/response models use Pydantic for validation
- Reuse existing models where applicable (e.g., `BacktestRequest` fields)

### JSON Serialization
- Handle numpy types: `np.float64` → `float`, `np.int64` → `int`
- Handle pandas types: `pd.Timestamp` → ISO string, `pd.Series` → list
- Handle NaN/Inf: Convert to `null` or appropriate defaults

### Error Handling
- Return 400 for invalid inputs (bad dates, invalid tickers)
- Return 500 for internal errors with detail message
- Log errors with traceback for debugging

### Caching
- Leverage existing `testfol_api` disk cache for backtest data
- Consider response caching for expensive calculations (MA analysis on long series)

---

## File Changes

| File | Changes |
|------|---------|
| `app/services/backend.py` | Add 4 new endpoints, Pydantic models |
| `app/core/calculations.py` | No changes (reuse existing functions) |
| `app/core/shadow_backtest.py` | No changes |
| `app/core/tax_library.py` | No changes |

---

## Testing Plan

1. Unit test each endpoint with sample data
2. Verify response structure matches design
3. Test error cases (invalid tickers, bad date ranges)
4. Load test with realistic mobile app usage patterns

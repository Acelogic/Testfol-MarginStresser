"""Calculations package — backward-compatible re-exports."""

from app.core.calculations.stats import (
    calculate_cagr,
    calculate_max_drawdown,
    calculate_sharpe_ratio,
    calculate_tax_adjusted_equity,
    process_rebalancing_data,
    generate_stats,
)
from app.core.calculations.moving_averages import (
    analyze_ma,
    analyze_wma,
    compare_breach_events,
    compare_wma_breach_events,
)
from app.core.calculations.technical import (
    calculate_pivot_points,
    calculate_cheat_sheet,
    analyze_stage,
)

__all__ = [
    "calculate_cagr",
    "calculate_max_drawdown",
    "calculate_sharpe_ratio",
    "calculate_tax_adjusted_equity",
    "process_rebalancing_data",
    "generate_stats",
    "analyze_ma",
    "analyze_wma",
    "compare_breach_events",
    "compare_wma_breach_events",
    "calculate_pivot_points",
    "calculate_cheat_sheet",
    "analyze_stage",
]

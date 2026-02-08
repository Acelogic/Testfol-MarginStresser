"""
Charts package — split from monolithic charts.py for maintainability.

Sub-modules:
  portfolio       — multi-portfolio comparison, classic line, dashboard
  candlestick     — candlestick/OHLC chart
  moving_averages — 200DMA, 150MA (Weinstein), Munger 200WMA
  returns         — cheat sheet, returns heatmaps
  rebalancing     — sankey, composition, rebalancing analysis
  analysis        — tax analysis, monte carlo
"""

from app.ui.charts.portfolio import (
    render_multi_portfolio_chart,
    render_classic_chart,
    render_dashboard_view,
)
from app.ui.charts.candlestick import render_candlestick_chart
from app.ui.charts.moving_averages import (
    render_ma_analysis_tab,
    render_munger_wma_tab,
)
from app.ui.charts.returns import (
    render_cheat_sheet,
    render_returns_analysis,
)
from app.ui.charts.rebalancing import (
    render_rebalance_sankey,
    render_portfolio_composition,
    render_rebalancing_analysis,
)
from app.ui.charts.analysis import (
    render_tax_analysis,
    render_monte_carlo_view,
)

__all__ = [
    "render_multi_portfolio_chart",
    "render_classic_chart",
    "render_dashboard_view",
    "render_candlestick_chart",
    "render_ma_analysis_tab",
    "render_munger_wma_tab",
    "render_cheat_sheet",
    "render_returns_analysis",
    "render_rebalance_sankey",
    "render_portfolio_composition",
    "render_rebalancing_analysis",
    "render_tax_analysis",
    "render_monte_carlo_view",
]

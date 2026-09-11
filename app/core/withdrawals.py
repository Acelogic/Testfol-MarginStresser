"""Withdrawal timing shared by the portfolio and margin simulations."""

import pandas as pd

PERFORMANCE_CASHFLOW_POLICY = "independent-margin-v1"


def dca_stop_date(draw_monthly, draw_start_date, retirement_draw, retirement_date, start_date):
    """Return the first active withdrawal date, or None when no draw is scheduled."""
    dates = []
    if draw_monthly > 0:
        dates.append(pd.Timestamp(draw_start_date or start_date).date())
    if retirement_draw > 0 and retirement_date is not None:
        dates.append(pd.Timestamp(retirement_date).date())
    return min(dates) if dates else None

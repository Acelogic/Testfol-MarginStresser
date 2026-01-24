# app/services/json_utils.py
"""JSON serialization helpers for numpy/pandas types."""
import numpy as np
import pandas as pd
from typing import Any, Dict, List, Union


def safe_float(val: Any) -> Union[float, None]:
    """Convert numpy/pandas float to Python float, handling NaN/Inf."""
    if val is None:
        return None
    if isinstance(val, (np.floating, np.integer)):
        val = float(val)
    if isinstance(val, float):
        if np.isnan(val) or np.isinf(val):
            return None
        return val
    return val


def safe_int(val: Any) -> Union[int, None]:
    """Convert numpy int to Python int."""
    if val is None:
        return None
    if isinstance(val, (np.integer,)):
        return int(val)
    return val


def series_to_list(series: pd.Series) -> List[Union[float, None]]:
    """Convert pandas Series to list of floats, handling NaN."""
    return [safe_float(v) for v in series.values]


def dates_to_strings(index: pd.DatetimeIndex) -> List[str]:
    """Convert DatetimeIndex to ISO date strings."""
    return [d.strftime('%Y-%m-%d') for d in index]


def safe_dict(d: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively convert dict values to JSON-safe types."""
    result = {}
    for k, v in d.items():
        if isinstance(v, (np.floating, np.integer)):
            result[k] = safe_float(v)
        elif isinstance(v, dict):
            result[k] = safe_dict(v)
        elif isinstance(v, pd.Timestamp):
            result[k] = v.strftime('%Y-%m-%d')
        elif pd.isna(v):
            result[k] = None
        else:
            result[k] = v
    return result

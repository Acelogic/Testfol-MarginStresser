import json
import os
from functools import lru_cache

import pandas as pd
import requests

import config


OFFICIAL_CONSTITUENTS_CACHE = os.path.join(config.CACHE_DIR, "official_constituents.json")
WEIGHTING_DATA_URL = "https://indexes.nasdaqomx.com/Index/WeightingData"
OFFICIAL_MEMBERSHIP_DIR = os.path.join(config.ASSETS_DIR, "official_membership")
LOCAL_MEMBERSHIP_FILES = {
    "NDX": os.path.join(OFFICIAL_MEMBERSHIP_DIR, "ndx_official_membership_daily.csv"),
    "NDXMEGA": os.path.join(OFFICIAL_MEMBERSHIP_DIR, "ndxmega_official_membership_daily.csv"),
    "NDX30": os.path.join(OFFICIAL_MEMBERSHIP_DIR, "ndx30_official_membership_daily.csv"),
}
MAX_LOCAL_STALENESS_DAYS = 7


def _load_cache():
    if not os.path.exists(OFFICIAL_CONSTITUENTS_CACHE):
        return {}
    try:
        with open(OFFICIAL_CONSTITUENTS_CACHE, "r") as f:
            return json.load(f)
    except Exception:
        return {}


def _save_cache(cache):
    os.makedirs(os.path.dirname(OFFICIAL_CONSTITUENTS_CACHE), exist_ok=True)
    with open(OFFICIAL_CONSTITUENTS_CACHE, "w") as f:
        json.dump(cache, f, indent=2, sort_keys=True)


@lru_cache(maxsize=None)
def _load_local_membership(index_symbol):
    path = LOCAL_MEMBERSHIP_FILES.get(index_symbol)
    if not path or not os.path.exists(path):
        return None

    try:
        df = pd.read_csv(path, parse_dates=["Date"])
    except Exception:
        return None

    if df.empty or "Tickers" not in df.columns:
        return None

    return df.sort_values("Date").set_index("Date")


def _get_local_constituents(index_symbol, trade_date):
    df = _load_local_membership(index_symbol)
    if df is None or df.empty:
        return None

    trade_ts = pd.Timestamp(trade_date).normalize()
    pos = df.index.searchsorted(trade_ts, side="right") - 1
    if pos < 0:
        return None

    matched_date = df.index[pos]
    if matched_date > trade_ts:
        return None
    if (trade_ts - matched_date) > pd.Timedelta(days=MAX_LOCAL_STALENESS_DAYS):
        return None

    tickers = df.iloc[pos].get("Tickers", "")
    if not isinstance(tickers, str) or not tickers:
        return None

    return [ticker for ticker in tickers.split("|") if ticker]


def get_official_constituents(index_symbol, trade_date, time_of_day="SOD"):
    trade_date = pd.Timestamp(trade_date).date().isoformat()
    cache_key = f"{index_symbol}|{trade_date}|{time_of_day}"

    local_tickers = _get_local_constituents(index_symbol, trade_date)
    if local_tickers:
        return local_tickers

    cache = _load_cache()
    if cache_key in cache:
        return cache[cache_key] or None

    try:
        response = requests.post(
            WEIGHTING_DATA_URL,
            data={"id": index_symbol, "tradeDate": trade_date, "timeOfDay": time_of_day},
            timeout=30,
        )
        response.raise_for_status()
        payload = response.json()
        tickers = [
            row.get("Symbol")
            for row in payload.get("aaData", [])
            if row.get("Symbol")
        ]
    except Exception:
        tickers = []

    cache[cache_key] = tickers
    _save_cache(cache)
    return tickers or None

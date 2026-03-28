import json
import os

import pandas as pd
import requests

import config


OFFICIAL_CONSTITUENTS_CACHE = os.path.join(config.CACHE_DIR, "official_constituents.json")
WEIGHTING_DATA_URL = "https://indexes.nasdaqomx.com/Index/WeightingData"


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


def get_official_constituents(index_symbol, trade_date, time_of_day="SOD"):
    trade_date = pd.Timestamp(trade_date).date().isoformat()
    cache_key = f"{index_symbol}|{trade_date}|{time_of_day}"

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

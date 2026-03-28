import concurrent.futures as cf
import json
import os
import sys
import time
from datetime import datetime, timezone

import pandas as pd
import requests

sys.path.append(os.path.join(os.path.dirname(__file__), "../src"))
import config


WEIGHTING_DATA_URL = "https://indexes.nasdaqomx.com/Index/WeightingData"
OUTPUT_DIR = os.path.join(config.ASSETS_DIR, "official_membership")
TIME_OF_DAY = "SOD"
MAX_WORKERS = 12
RETRIES = 3

INDEX_SEARCH_STARTS = {
    "NDX": "2000-01-01",
    "NDXMEGA": "2024-01-01",
    "NDX30": "2024-01-01",
}

VERIFIED_FIRST_AVAILABLE_DATES = {
    "NDX": "2003-01-02",
    "NDXMEGA": "2024-07-29",
    "NDX30": "2024-08-26",
}


def fetch_snapshot(index_symbol, trade_date):
    trade_date = pd.Timestamp(trade_date).date().isoformat()
    last_error = None

    for attempt in range(1, RETRIES + 1):
        try:
            response = requests.post(
                WEIGHTING_DATA_URL,
                data={"id": index_symbol, "tradeDate": trade_date, "timeOfDay": TIME_OF_DAY},
                timeout=20,
            )
            response.raise_for_status()
            payload = response.json()
            rows = payload.get("aaData", [])
            tickers = [row.get("Symbol") for row in rows if row.get("Symbol")]
            names = [row.get("Name") for row in rows if row.get("Symbol")]
            return {
                "Date": trade_date,
                "Count": len(tickers),
                "Tickers": "|".join(tickers),
                "Names": "|".join(names),
            }
        except Exception as exc:
            last_error = exc
            if attempt < RETRIES:
                time.sleep(0.5 * attempt)

    raise RuntimeError(f"{index_symbol} {trade_date}: {last_error}")


def find_first_available_date(index_symbol, search_start, end_date):
    monthly_dates = pd.date_range(search_start, end_date, freq="BMS")
    if monthly_dates.empty:
        monthly_dates = pd.DatetimeIndex([pd.Timestamp(search_start)])

    previous_probe = pd.Timestamp(search_start) - pd.offsets.BDay(1)

    for probe in monthly_dates:
        snapshot = fetch_snapshot(index_symbol, probe)
        if snapshot["Count"] > 0:
            for day in pd.bdate_range(previous_probe + pd.offsets.BDay(1), probe):
                daily_snapshot = fetch_snapshot(index_symbol, day)
                if daily_snapshot["Count"] > 0:
                    return pd.Timestamp(daily_snapshot["Date"])
            return pd.Timestamp(snapshot["Date"])
        previous_probe = probe

    return None


def build_periods(daily_df):
    if daily_df.empty:
        return pd.DataFrame(columns=["StartDate", "EndDate", "Count", "Tickers", "Names"])

    periods = []
    start_row = daily_df.iloc[0]
    prev_row = start_row

    for _, row in daily_df.iloc[1:].iterrows():
        if row["Tickers"] != prev_row["Tickers"]:
            periods.append(
                {
                    "StartDate": start_row["Date"],
                    "EndDate": prev_row["Date"],
                    "Count": start_row["Count"],
                    "Tickers": start_row["Tickers"],
                    "Names": start_row["Names"],
                }
            )
            start_row = row
        prev_row = row

    periods.append(
        {
            "StartDate": start_row["Date"],
            "EndDate": prev_row["Date"],
            "Count": start_row["Count"],
            "Tickers": start_row["Tickers"],
            "Names": start_row["Names"],
        }
    )
    return pd.DataFrame(periods)


def download_index(index_symbol, today):
    search_start = pd.Timestamp(INDEX_SEARCH_STARTS[index_symbol])
    verified_first = VERIFIED_FIRST_AVAILABLE_DATES.get(index_symbol)
    if verified_first is not None:
        first_available = pd.Timestamp(verified_first)
    else:
        first_available = find_first_available_date(index_symbol, search_start, today)
    if first_available is None:
        raise RuntimeError(f"No public membership data found for {index_symbol}")

    trade_days = pd.bdate_range(first_available, today)
    rows = []

    print(
        f"{index_symbol}: first public date {first_available.date()} "
        f"-> probing {len(trade_days)} business days"
    )

    with cf.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_day = {
            executor.submit(fetch_snapshot, index_symbol, trade_day): trade_day
            for trade_day in trade_days
        }
        for idx, future in enumerate(cf.as_completed(future_to_day), start=1):
            row = future.result()
            if row["Count"] > 0:
                rows.append(row)
            if idx % 250 == 0 or idx == len(future_to_day):
                print(f"  {index_symbol}: fetched {idx}/{len(future_to_day)} dates")

    daily_df = pd.DataFrame(rows).sort_values("Date").reset_index(drop=True)
    periods_df = build_periods(daily_df)

    daily_path = os.path.join(OUTPUT_DIR, f"{index_symbol.lower()}_official_membership_daily.csv")
    periods_path = os.path.join(OUTPUT_DIR, f"{index_symbol.lower()}_official_membership_periods.csv")

    daily_df.to_csv(daily_path, index=False)
    periods_df.to_csv(periods_path, index=False)

    return {
        "index_symbol": index_symbol,
        "first_available_date": daily_df.iloc[0]["Date"],
        "last_available_date": daily_df.iloc[-1]["Date"],
        "daily_rows": int(len(daily_df)),
        "period_rows": int(len(periods_df)),
        "daily_file": os.path.basename(daily_path),
        "periods_file": os.path.basename(periods_path),
    }


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    today = pd.Timestamp.today().normalize()
    manifest = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source": WEIGHTING_DATA_URL,
        "time_of_day": TIME_OF_DAY,
        "indexes": {},
    }

    for index_symbol in ["NDX", "NDXMEGA", "NDX30"]:
        summary = download_index(index_symbol, today)
        manifest["indexes"][index_symbol] = summary

    manifest_path = os.path.join(OUTPUT_DIR, "official_membership_manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)

    print(f"Saved manifest to {manifest_path}")


if __name__ == "__main__":
    main()

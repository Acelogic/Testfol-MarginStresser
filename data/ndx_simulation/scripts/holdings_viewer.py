import argparse
import os
import sys

import pandas as pd

sys.path.append(os.path.join(os.path.dirname(__file__), "../src"))
import config


WEIGHTS_FILE = config.WEIGHTS_FILE
HISTORY_FILES = {
    "mega1": {
        "label": "NDX Mega 1.0",
        "path": os.path.join(config.RESULTS_DIR, "ndx_mega_constituents.csv"),
    },
    "mega2": {
        "label": "NDX Mega 2.0",
        "path": os.path.join(config.RESULTS_DIR, "ndx_mega2_constituents.csv"),
    },
    "ndx30": {
        "label": "NDX30",
        "path": os.path.join(config.RESULTS_DIR, "ndx30_constituents.csv"),
    },
}
STRATEGY_ORDER = ["mega1", "mega2", "ndx30"]


def load_weights():
    if not os.path.exists(WEIGHTS_FILE):
        raise FileNotFoundError(f"{WEIGHTS_FILE} not found. Run rebuild_all.py first.")

    weights = pd.read_csv(WEIGHTS_FILE)
    weights["Date"] = pd.to_datetime(weights["Date"])
    weights = weights[weights["Date"] <= pd.Timestamp.now().normalize()]
    return weights.sort_values(["Date", "Weight", "Ticker"], ascending=[True, False, True])


def load_history(path):
    if not os.path.exists(path):
        return pd.DataFrame()

    history = pd.read_csv(path)
    if history.empty:
        return history

    history["Date"] = pd.to_datetime(history["Date"])
    return history.sort_values("Date")


def build_ticker_name_map(weights):
    named = weights[["Ticker", "Name", "Date"]].dropna(subset=["Ticker", "Name"]).copy()
    if named.empty:
        return {}

    named = named.sort_values(["Ticker", "Date"])
    return named.groupby("Ticker")["Name"].last().to_dict()


def pick_effective_date(available_dates, requested_date):
    dates = pd.DatetimeIndex(pd.to_datetime(pd.Series(available_dates).dropna().unique())).sort_values()
    if dates.empty:
        return None, "no dates available"

    if requested_date is None:
        return dates[-1], "latest available"

    requested = pd.Timestamp(requested_date).normalize()
    exact = dates[dates == requested]
    if len(exact):
        return exact[-1], "exact match"

    prior = dates[dates <= requested]
    if len(prior):
        return prior[-1], "latest rebalance on or before requested date"

    return dates[0], "earliest available after requested date"


def parse_requested_date(value, weights):
    if value is None:
        return None

    value = value.strip()
    if not value:
        return None

    if value.isdigit() and len(value) == 4:
        year = int(value)
        year_dates = pd.DatetimeIndex(weights.loc[weights["Date"].dt.year == year, "Date"].unique()).sort_values()
        if year_dates.empty:
            raise ValueError(f"No rebalance dates available for {year}.")
        return year_dates[-1]

    return pd.Timestamp(value).normalize()


def get_weight_slice(weights, requested_date):
    effective_date, reason = pick_effective_date(weights["Date"], requested_date)
    if effective_date is None:
        return pd.DataFrame(), None, reason

    subset = weights[weights["Date"] == effective_date].copy()
    return subset.sort_values(["Weight", "Ticker"], ascending=[False, True]), effective_date, reason


def get_history_snapshot(history, requested_date, ticker_name_map):
    if history.empty:
        return None

    effective_date, reason = pick_effective_date(history["Date"], requested_date)
    if effective_date is None:
        return None

    row = history.loc[history["Date"] == effective_date].iloc[-1]
    tickers = [ticker for ticker in str(row.get("Tickers", "")).split("|") if ticker]
    weights = [float(weight) for weight in str(row.get("Weights", "")).split("|") if weight]
    count = min(len(tickers), len(weights))

    holdings = pd.DataFrame(
        {
            "Ticker": tickers[:count],
            "Weight": weights[:count],
        }
    )
    holdings["Name"] = holdings["Ticker"].map(ticker_name_map).fillna("Unknown")
    holdings = holdings.sort_values(["Weight", "Ticker"], ascending=[False, True]).reset_index(drop=True)

    metadata = {
        key: row[key]
        for key in history.columns
        if key not in {"Date", "Tickers", "Weights"} and pd.notna(row[key])
    }

    return {
        "effective_date": effective_date,
        "reason": reason,
        "holdings": holdings,
        "metadata": metadata,
    }


def print_table(title, df, weight_col):
    print(f"\n--- {title} ---")
    if df.empty:
        print("No holdings available.")
        return

    print(f"{'Ticker':<10} {'Name':<42} {'Weight':>10}")
    print("-" * 68)
    for _, row in df.iterrows():
        print(f"{row['Ticker']:<10} {str(row['Name'])[:42]:<42} {row[weight_col]:>9.2%}")
    print("-" * 68)


def print_ndx_top(weights_slice, effective_date, reason, limit):
    print("\n" + "=" * 72)
    print("NDX Weight Snapshot")
    print(f"Using rebalance date: {effective_date.date()} ({reason})")
    top = weights_slice[["Ticker", "Name", "Weight"]].head(limit).copy()
    print_table(f"Top {limit}", top, "Weight")


def print_strategy_view(strategy_key, snapshot):
    config_row = HISTORY_FILES[strategy_key]
    label = config_row["label"]

    if snapshot is None:
        print("\n" + "=" * 72)
        print(label)
        print("No constituent history available.")
        return

    holdings = snapshot["holdings"]
    metadata = snapshot["metadata"]
    effective_date = snapshot["effective_date"]
    reason = snapshot["reason"]

    print("\n" + "=" * 72)
    print(label)
    print(f"Using rebalance date: {effective_date.date()} ({reason})")

    meta_parts = []
    for key in ["Type", "BufferRule", "Count", "Top"]:
        if key in metadata:
            meta_parts.append(f"{key}={metadata[key]}")
    if meta_parts:
        print("Metadata: " + ", ".join(meta_parts))

    print(f"Constituents: {len(holdings)} | Weight sum: {holdings['Weight'].sum():.2%}")
    print_table("Holdings", holdings, "Weight")


def interactive_date_prompt(weights):
    min_date = weights["Date"].min().date()
    max_date = weights["Date"].max().date()
    print(f"Weight data available from {min_date} to {max_date}.")
    print("Enter a rebalance date like 2005-06-20, a calendar date like 2005-07-01,")
    print("or just a year like 2005 to use the last rebalance in that year.")
    user_input = input("Date [blank for latest]: ").strip()
    return parse_requested_date(user_input, weights)


def build_parser():
    parser = argparse.ArgumentParser(
        description="View historical NDX, NDX Mega, NDX Mega 2.0, and NDX30 holdings.",
    )
    parser.add_argument(
        "--date",
        help="Target date (YYYY-MM-DD) or year (YYYY). Defaults to interactive prompt, or latest if stdin is not a TTY.",
    )
    parser.add_argument(
        "--strategy",
        choices=["all"] + STRATEGY_ORDER,
        default="all",
        help="Which strategy to display. Defaults to all.",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=25,
        help="Number of top NDX names to show from the reconstructed weight snapshot.",
    )
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    try:
        weights = load_weights()
    except FileNotFoundError as exc:
        print(f"Error: {exc}")
        sys.exit(1)

    ticker_name_map = build_ticker_name_map(weights)
    history = {key: load_history(item["path"]) for key, item in HISTORY_FILES.items()}

    try:
        if args.date is not None:
            requested_date = parse_requested_date(args.date, weights)
        elif sys.stdin.isatty():
            requested_date = interactive_date_prompt(weights)
        else:
            requested_date = None
    except Exception as exc:
        print(f"Error parsing date: {exc}")
        sys.exit(1)

    weights_slice, weight_date, weight_reason = get_weight_slice(weights, requested_date)
    if weights_slice.empty or weight_date is None:
        print("No weight data available for the requested date.")
        sys.exit(1)

    print_ndx_top(weights_slice, weight_date, weight_reason, args.top)

    requested_keys = STRATEGY_ORDER if args.strategy == "all" else [args.strategy]
    for key in requested_keys:
        snapshot = get_history_snapshot(history[key], requested_date, ticker_name_map)
        print_strategy_view(key, snapshot)


if __name__ == "__main__":
    main()

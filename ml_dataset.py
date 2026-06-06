from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from main import DB_PATH, load_prices

BASE_FEATURES = [
    "open",
    "high",
    "low",
    "close",
    "volume",
    "ma_5",
    "ma_20",
    "rsi_14",
    "daily_return",
]

ENGINEERED_FEATURES = [
    "momentum_5d",
    "momentum_10d",
    "volatility_10d",
    "volatility_20d",
    "ma_ratio",
    "price_distance_ma20",
    "rsi_change",
    "volume_change",
]

TARGET_COLUMNS = [
    "tomorrow_close",
    "tomorrow_return",
    "target_up",
]

FEATURE_COLUMNS = BASE_FEATURES + ENGINEERED_FEATURES


def build_ml_dataset(prices: pd.DataFrame) -> pd.DataFrame:
    data = prices.sort_values("date").copy()
    data["momentum_5d"] = data["close"].pct_change(5)
    data["momentum_10d"] = data["close"].pct_change(10)
    data["volatility_10d"] = data["daily_return"].rolling(10).std()
    data["volatility_20d"] = data["daily_return"].rolling(20).std()
    data["ma_ratio"] = data["ma_5"] / data["ma_20"]
    data["price_distance_ma20"] = data["close"] / data["ma_20"] - 1
    data["rsi_change"] = data["rsi_14"].diff()
    data["volume_change"] = data["volume"].pct_change()

    data["tomorrow_close"] = data["close"].shift(-1)
    data["tomorrow_return"] = data["tomorrow_close"] / data["close"] - 1
    data["target_up"] = (data["tomorrow_close"] > data["close"]).astype(int)

    columns = ["ticker", "date"] + FEATURE_COLUMNS + TARGET_COLUMNS
    dataset = data[columns].replace([float("inf"), float("-inf")], pd.NA)
    return dataset.dropna(subset=FEATURE_COLUMNS + TARGET_COLUMNS).reset_index(drop=True)


def build_dataset_for_ticker(ticker: str, db_path: Path = DB_PATH) -> pd.DataFrame:
    return build_ml_dataset(load_prices(ticker, db_path))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build an ML feature dataset for stock direction prediction.")
    parser.add_argument("ticker", help="Ticker symbol, for example: AAPL")
    parser.add_argument("--db", type=Path, default=DB_PATH, help="SQLite market data path")
    parser.add_argument("--output", type=Path, help="Optional CSV output path")
    return parser


if __name__ == "__main__":
    args = build_parser().parse_args()
    ml_dataset = build_dataset_for_ticker(args.ticker, args.db)
    print(ml_dataset.tail().to_string(index=False))
    print(f"\nRows: {len(ml_dataset)}")
    print(f"Features: {', '.join(FEATURE_COLUMNS)}")
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        ml_dataset.to_csv(args.output, index=False)
        print(f"Saved: {args.output}")

from __future__ import annotations

import argparse
import sqlite3
from pathlib import Path

import pandas as pd
import yfinance as yf

DB_PATH = Path("data/market_data.sqlite")


def ensure_database(db_path: Path = DB_PATH) -> None:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS stock_prices (
                ticker TEXT NOT NULL,
                date TEXT NOT NULL,
                open REAL,
                high REAL,
                low REAL,
                close REAL,
                adj_close REAL,
                volume INTEGER,
                daily_return REAL,
                ma_5 REAL,
                ma_20 REAL,
                rsi_14 REAL,
                PRIMARY KEY (ticker, date)
            )
            """
        )


def download_prices(ticker: str, period: str = "1y") -> pd.DataFrame:
    frame = yf.download(ticker, period=period, auto_adjust=False, progress=False)
    if frame.empty:
        raise ValueError(f"No data returned for ticker '{ticker}'.")

    if isinstance(frame.columns, pd.MultiIndex):
        frame.columns = frame.columns.get_level_values(0)

    frame = frame.reset_index()
    frame.columns = [str(column).lower().replace(" ", "_") for column in frame.columns]
    frame["ticker"] = ticker.upper()
    return frame


def calculate_indicators(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.sort_values("date").copy()
    close = frame["close"]

    frame["daily_return"] = close.pct_change()
    frame["ma_5"] = close.rolling(window=5, min_periods=5).mean()
    frame["ma_20"] = close.rolling(window=20, min_periods=20).mean()

    delta = close.diff()
    gains = delta.clip(lower=0)
    losses = -delta.clip(upper=0)
    avg_gain = gains.rolling(window=14, min_periods=14).mean()
    avg_loss = losses.rolling(window=14, min_periods=14).mean()
    relative_strength = avg_gain / avg_loss
    frame["rsi_14"] = 100 - (100 / (1 + relative_strength))

    return frame


def save_prices(frame: pd.DataFrame, db_path: Path = DB_PATH) -> None:
    ensure_database(db_path)
    columns = [
        "ticker",
        "date",
        "open",
        "high",
        "low",
        "close",
        "adj_close",
        "volume",
        "daily_return",
        "ma_5",
        "ma_20",
        "rsi_14",
    ]
    frame = frame[columns].copy()
    frame["date"] = pd.to_datetime(frame["date"]).dt.strftime("%Y-%m-%d")

    with sqlite3.connect(db_path) as conn:
        conn.executemany(
            """
            INSERT OR REPLACE INTO stock_prices (
                ticker, date, open, high, low, close, adj_close, volume,
                daily_return, ma_5, ma_20, rsi_14
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            frame.where(pd.notnull(frame), None).itertuples(index=False, name=None),
        )


def load_prices(ticker: str | None = None, db_path: Path = DB_PATH) -> pd.DataFrame:
    ensure_database(db_path)
    query = "SELECT * FROM stock_prices"
    params: tuple[str, ...] = ()
    if ticker:
        query += " WHERE ticker = ?"
        params = (ticker.upper(),)
    query += " ORDER BY ticker, date"

    with sqlite3.connect(db_path) as conn:
        return pd.read_sql_query(query, conn, params=params, parse_dates=["date"])


def run_pipeline(tickers: list[str], period: str, db_path: Path = DB_PATH) -> None:
    for ticker in tickers:
        prices = download_prices(ticker, period=period)
        enriched = calculate_indicators(prices)
        save_prices(enriched, db_path=db_path)
        print(f"Saved {len(enriched)} rows for {ticker.upper()} to {db_path}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Stock data pipeline using Yahoo Finance and SQLite.")
    parser.add_argument("tickers", nargs="+", help="Ticker symbols, for example: AAPL MSFT NVDA")
    parser.add_argument("--period", default="1y", help="Yahoo Finance period, for example: 6mo, 1y, 5y")
    parser.add_argument("--db", type=Path, default=DB_PATH, help="SQLite database path")
    return parser


if __name__ == "__main__":
    args = build_parser().parse_args()
    run_pipeline(args.tickers, args.period, args.db)

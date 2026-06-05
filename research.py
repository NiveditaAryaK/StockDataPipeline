from __future__ import annotations

import argparse
import sqlite3
from pathlib import Path

import pandas as pd

from backtest import BollingerBandStrategy, MACrossoverStrategy, MomentumStrategy, RSIStrategy, buy_hold_return, run_strategy_backtest
from main import DB_PATH, load_prices, run_pipeline

RESULTS_DIR = Path("research_results")
RESULTS_CSV = RESULTS_DIR / "strategy_results.csv"
RESULTS_DB = RESULTS_DIR / "strategy_results.sqlite"
DEFAULT_TICKERS = ["AAPL", "MSFT", "NVDA", "TSLA", "AMZN", "GOOGL", "META", "SPY"]
DEFAULT_WINDOWS = [5, 10]


def strategy_suite() -> list:
    return [
        RSIStrategy(),
        MACrossoverStrategy(),
        MomentumStrategy(),
        BollingerBandStrategy(),
    ]


def filter_years(prices: pd.DataFrame, years: int) -> pd.DataFrame:
    data = prices.sort_values("date").copy()
    if data.empty:
        return data
    end_date = data["date"].max()
    start_date = end_date - pd.DateOffset(years=years)
    return data[data["date"] >= start_date].copy()


def run_research(
    tickers: list[str],
    windows: list[int],
    initial_cash: float = 10_000,
    db_path: Path = DB_PATH,
) -> pd.DataFrame:
    rows = []
    for ticker in tickers:
        prices = load_prices(ticker, db_path)
        for years in windows:
            window_prices = filter_years(prices, years)
            if len(window_prices.dropna(subset=["close"])) < 2:
                rows.append(error_row(ticker, years, "Buy Hold", "Not enough price data"))
                continue

            benchmark_return = buy_hold_return(window_prices)
            rows.append(
                {
                    "ticker": ticker.upper(),
                    "window_years": years,
                    "strategy": "Buy Hold",
                    "total_return": benchmark_return,
                    "buy_hold_return": benchmark_return,
                    "alpha": 0.0,
                    "total_profit": initial_cash * benchmark_return,
                    "win_rate": None,
                    "max_drawdown": None,
                    "sharpe_ratio": None,
                    "trades": 1,
                    "winning_trades": None,
                    "ending_equity": initial_cash * (1 + benchmark_return),
                    "error": None,
                }
            )

            for strategy in strategy_suite():
                try:
                    result = run_strategy_backtest(window_prices, ticker, strategy, initial_cash)
                except ValueError as error:
                    rows.append(error_row(ticker, years, strategy.name, str(error)))
                    continue

                rows.append(
                    {
                        "ticker": ticker.upper(),
                        "window_years": years,
                        "strategy": result.strategy_name,
                        "total_return": result.total_return,
                        "buy_hold_return": result.buy_hold_return,
                        "alpha": result.alpha,
                        "total_profit": result.total_profit,
                        "win_rate": result.win_rate,
                        "max_drawdown": result.max_drawdown,
                        "sharpe_ratio": result.sharpe_ratio,
                        "trades": result.trade_count,
                        "winning_trades": result.winning_trades,
                        "ending_equity": result.ending_equity,
                        "error": None,
                    }
                )

    return pd.DataFrame(rows)


def error_row(ticker: str, years: int, strategy: str, error: str) -> dict[str, object]:
    return {
        "ticker": ticker.upper(),
        "window_years": years,
        "strategy": strategy,
        "total_return": None,
        "buy_hold_return": None,
        "alpha": None,
        "total_profit": None,
        "win_rate": None,
        "max_drawdown": None,
        "sharpe_ratio": None,
        "trades": 0,
        "winning_trades": None,
        "ending_equity": None,
        "error": error,
    }


def save_results(results: pd.DataFrame, csv_path: Path = RESULTS_CSV, sqlite_path: Path = RESULTS_DB) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(csv_path, index=False)
    with sqlite3.connect(sqlite_path) as conn:
        results.to_sql("strategy_results", conn, if_exists="replace", index=False)


def format_results(results: pd.DataFrame) -> pd.DataFrame:
    display = results.copy()
    for column in ["total_return", "buy_hold_return", "alpha", "win_rate", "max_drawdown"]:
        display[column] = display[column].map(lambda value: None if pd.isna(value) else round(value * 100, 2))
    for column in ["total_profit", "ending_equity"]:
        display[column] = display[column].map(lambda value: None if pd.isna(value) else round(value, 2))
    display["sharpe_ratio"] = display["sharpe_ratio"].map(lambda value: None if pd.isna(value) else round(value, 2))
    return display


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run long-horizon strategy research.")
    parser.add_argument("--tickers", nargs="+", default=DEFAULT_TICKERS, help="Ticker symbols to test")
    parser.add_argument("--windows", nargs="+", type=int, default=DEFAULT_WINDOWS, help="Lookback windows in years")
    parser.add_argument("--cash", type=float, default=10_000, help="Starting cash for each test")
    parser.add_argument("--db", type=Path, default=DB_PATH, help="SQLite market data path")
    parser.add_argument("--refresh", action="store_true", help="Download 10 years of data before testing")
    return parser


if __name__ == "__main__":
    args = build_parser().parse_args()
    if args.refresh:
        run_pipeline(args.tickers, "10y", args.db)

    research_results = run_research(args.tickers, args.windows, args.cash, args.db)
    save_results(research_results)
    print(format_results(research_results).to_string(index=False))
    print(f"\nSaved CSV: {RESULTS_CSV}")
    print(f"Saved SQLite: {RESULTS_DB}")

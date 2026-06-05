from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from main import DB_PATH, load_prices

TRADING_DAYS_PER_YEAR = 252


@dataclass(frozen=True)
class BacktestResult:
    ticker: str
    total_profit: float
    total_return: float
    win_rate: float
    max_drawdown: float
    sharpe_ratio: float
    trade_count: int
    winning_trades: int
    ending_equity: float
    equity_curve: pd.DataFrame
    trades: pd.DataFrame
    signals: pd.DataFrame


def run_rsi_backtest(
    prices: pd.DataFrame,
    ticker: str,
    initial_cash: float = 10_000,
    buy_threshold: float = 30,
    sell_threshold: float = 70,
) -> BacktestResult:
    required_columns = {"date", "open", "close", "rsi_14"}
    missing_columns = required_columns.difference(prices.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"Historical data is missing required columns: {missing}")

    data = prices.dropna(subset=["date", "open", "close", "rsi_14"]).sort_values("date").copy()
    if len(data) < 2:
        raise ValueError("At least two usable rows are required for next-day execution.")

    cash = float(initial_cash)
    shares = 0.0
    entry_price: float | None = None
    entry_date: pd.Timestamp | None = None
    trades: list[dict[str, object]] = []
    equity_rows: list[dict[str, object]] = []
    signal_rows: list[dict[str, object]] = []

    rows = list(data.itertuples(index=False))
    for index, row in enumerate(rows):
        date = pd.Timestamp(row.date)
        open_price = float(row.open)
        close = float(row.close)
        rsi = float(row.rsi_14)
        signal = "HOLD"

        if index > 0:
            signal_rsi = float(rows[index - 1].rsi_14)
            signal_date = pd.Timestamp(rows[index - 1].date)
            if shares == 0 and signal_rsi < buy_threshold:
                shares = cash / open_price
                cash = 0.0
                entry_price = open_price
                entry_date = date
                signal = "BUY"
                signal_rows.append(
                    {
                        "signal_date": signal_date,
                        "execution_date": date,
                        "signal": signal,
                        "rsi_14": signal_rsi,
                        "execution_price": open_price,
                    }
                )
            elif shares > 0 and signal_rsi > sell_threshold:
                cash = shares * open_price
                profit = (open_price - float(entry_price)) * shares
                trades.append(
                    {
                        "entry_date": entry_date,
                        "exit_date": date,
                        "entry_price": entry_price,
                        "exit_price": open_price,
                        "shares": shares,
                        "profit": profit,
                        "return": open_price / float(entry_price) - 1,
                        "exit_reason": "rsi_sell",
                    }
                )
                signal = "SELL"
                signal_rows.append(
                    {
                        "signal_date": signal_date,
                        "execution_date": date,
                        "signal": signal,
                        "rsi_14": signal_rsi,
                        "execution_price": open_price,
                    }
                )
                shares = 0.0
                entry_price = None
                entry_date = None

        equity = cash + shares * close
        equity_rows.append(
            {
                "date": date,
                "open": open_price,
                "close": close,
                "rsi_14": rsi,
                "cash": cash,
                "shares": shares,
                "equity": equity,
                "position": 1 if shares > 0 else 0,
                "signal": signal,
            }
        )

    if shares > 0:
        final_row = rows[-1]
        final_date = pd.Timestamp(final_row.date)
        final_close = float(final_row.close)
        cash = shares * final_close
        profit = (final_close - float(entry_price)) * shares
        trades.append(
            {
                "entry_date": entry_date,
                "exit_date": final_date,
                "entry_price": entry_price,
                "exit_price": final_close,
                "shares": shares,
                "profit": profit,
                "return": final_close / float(entry_price) - 1,
                "exit_reason": "end_of_data",
            }
        )
        signal_rows.append(
            {
                "signal_date": final_date,
                "execution_date": final_date,
                "signal": "FORCED_SELL",
                "rsi_14": float(final_row.rsi_14),
                "execution_price": final_close,
            }
        )
        shares = 0.0

        equity_rows[-1]["cash"] = cash
        equity_rows[-1]["shares"] = shares
        equity_rows[-1]["equity"] = cash
        equity_rows[-1]["position"] = 0
        equity_rows[-1]["signal"] = "FORCED_SELL"

    equity_curve = pd.DataFrame(equity_rows)
    trades_frame = pd.DataFrame(trades)
    signals_frame = pd.DataFrame(signal_rows)
    ending_equity = float(equity_curve["equity"].iloc[-1])
    total_profit = ending_equity - initial_cash
    total_return = ending_equity / initial_cash - 1

    if trades_frame.empty:
        winning_trades = 0
        win_rate = 0.0
    else:
        winning_trades = int((trades_frame["profit"] > 0).sum())
        win_rate = winning_trades / len(trades_frame)

    running_max = equity_curve["equity"].cummax()
    drawdown = equity_curve["equity"] / running_max - 1
    max_drawdown = float(drawdown.min())

    daily_returns = equity_curve["equity"].pct_change().dropna()
    if daily_returns.empty or daily_returns.std() == 0:
        sharpe_ratio = 0.0
    else:
        sharpe_ratio = float((daily_returns.mean() / daily_returns.std()) * (TRADING_DAYS_PER_YEAR**0.5))

    return BacktestResult(
        ticker=ticker.upper(),
        total_profit=float(total_profit),
        total_return=float(total_return),
        win_rate=float(win_rate),
        max_drawdown=max_drawdown,
        sharpe_ratio=sharpe_ratio,
        trade_count=len(trades_frame),
        winning_trades=winning_trades,
        ending_equity=ending_equity,
        equity_curve=equity_curve,
        trades=trades_frame,
        signals=signals_frame,
    )


def format_percent(value: float) -> str:
    return f"{value * 100:.2f}%"


def print_result(result: BacktestResult) -> None:
    print(f"RSI Backtest: {result.ticker}")
    print(f"Ending Equity: ${result.ending_equity:,.2f}")
    print(f"Total Profit: ${result.total_profit:,.2f}")
    print(f"Total Return: {format_percent(result.total_return)}")
    print(f"Win Rate: {format_percent(result.win_rate)}")
    print(f"Maximum Drawdown: {format_percent(result.max_drawdown)}")
    print(f"Sharpe Ratio: {result.sharpe_ratio:.2f}")
    print(f"Trades: {result.trade_count} ({result.winning_trades} winners)")
    print(f"Signals: {len(result.signals)}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Backtest an RSI strategy using stored historical data.")
    parser.add_argument("ticker", help="Ticker symbol already stored in SQLite, for example: AAPL")
    parser.add_argument("--cash", type=float, default=10_000, help="Starting cash")
    parser.add_argument("--buy-rsi", type=float, default=30, help="Buy when RSI is below this value")
    parser.add_argument("--sell-rsi", type=float, default=70, help="Sell when RSI is above this value")
    parser.add_argument("--db", type=Path, default=DB_PATH, help="SQLite database path")
    parser.add_argument("--show-signals", action="store_true", help="Print every generated signal")
    return parser


if __name__ == "__main__":
    args = build_parser().parse_args()
    historical_prices = load_prices(args.ticker, args.db)
    backtest_result = run_rsi_backtest(
        historical_prices,
        args.ticker,
        initial_cash=args.cash,
        buy_threshold=args.buy_rsi,
        sell_threshold=args.sell_rsi,
    )
    print_result(backtest_result)
    if args.show_signals:
        print()
        print(backtest_result.signals.to_string(index=False))

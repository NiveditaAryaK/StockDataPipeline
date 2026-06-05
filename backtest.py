from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from main import DB_PATH, load_prices
from strategies import MACrossoverStrategy, RSIStrategy, STRATEGY_CLASSES, Strategy, build_strategy

TRADING_DAYS_PER_YEAR = 252


@dataclass(frozen=True)
class BacktestResult:
    strategy_name: str
    ticker: str
    total_profit: float
    total_return: float
    buy_hold_return: float
    alpha: float
    win_rate: float
    max_drawdown: float
    sharpe_ratio: float
    trade_count: int
    winning_trades: int
    ending_equity: float
    equity_curve: pd.DataFrame
    trades: pd.DataFrame
    signals: pd.DataFrame


def buy_hold_return(prices: pd.DataFrame) -> float:
    data = prices.dropna(subset=["close"]).sort_values("date")
    if len(data) < 2:
        raise ValueError("At least two close prices are required for buy-and-hold return.")
    return float(data["close"].iloc[-1] / data["close"].iloc[0] - 1)


def build_result(
    strategy_name: str,
    ticker: str,
    initial_cash: float,
    source_prices: pd.DataFrame,
    equity_rows: list[dict[str, object]],
    trades: list[dict[str, object]],
    signal_rows: list[dict[str, object]],
) -> BacktestResult:
    equity_curve = pd.DataFrame(equity_rows)
    trades_frame = pd.DataFrame(trades)
    signals_frame = pd.DataFrame(signal_rows)
    ending_equity = float(equity_curve["equity"].iloc[-1])
    total_profit = ending_equity - initial_cash
    total_return = ending_equity / initial_cash - 1
    benchmark_return = buy_hold_return(source_prices)

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
        strategy_name=strategy_name,
        ticker=ticker.upper(),
        total_profit=float(total_profit),
        total_return=float(total_return),
        buy_hold_return=benchmark_return,
        alpha=float(total_return - benchmark_return),
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


def run_strategy_backtest(
    prices: pd.DataFrame,
    ticker: str,
    strategy: Strategy,
    initial_cash: float = 10_000,
) -> BacktestResult:
    required_columns = {"date", "open", "close"}.union(strategy.required_columns)
    missing_columns = required_columns.difference(prices.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"Historical data is missing required columns: {missing}")

    data = prices.dropna(subset=sorted(required_columns)).sort_values("date").copy().reset_index(drop=True)
    if len(data) < strategy.minimum_rows:
        raise ValueError(f"At least {strategy.minimum_rows} usable rows are required for {strategy.name}.")

    generated_signals = strategy.generate_signals(data)
    signals_by_date = {}
    for signal_row in generated_signals.itertuples(index=False):
        signals_by_date[pd.Timestamp(signal_row.date)] = signal_row._asdict()

    cash = float(initial_cash)
    shares = 0.0
    entry_price: float | None = None
    entry_date: pd.Timestamp | None = None
    trades: list[dict[str, object]] = []
    equity_rows: list[dict[str, object]] = []
    signal_rows: list[dict[str, object]] = []

    for row in data.itertuples(index=False):
        date = pd.Timestamp(row.date)
        open_price = float(row.open)
        close = float(row.close)
        signal_record = signals_by_date.get(date)
        signal = "HOLD"

        if signal_record:
            generated_signal = signal_record["signal"]
            if shares == 0 and generated_signal == "BUY":
                shares = cash / open_price
                cash = 0.0
                entry_price = open_price
                entry_date = date
                signal = "BUY"
                signal_rows.append(format_signal_row(signal_record, date, signal, open_price))
            elif shares > 0 and generated_signal == "SELL":
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
                        "exit_reason": "strategy_sell",
                    }
                )
                signal = "SELL"
                signal_rows.append(format_signal_row(signal_record, date, signal, open_price))
                shares = 0.0
                entry_price = None
                entry_date = None

        equity_row = {
            "date": date,
            "open": open_price,
            "close": close,
            "cash": cash,
            "shares": shares,
            "equity": cash + shares * close,
            "position": 1 if shares > 0 else 0,
            "signal": signal,
        }
        for column in sorted(strategy.required_columns - {"date", "open", "close"}):
            equity_row[column] = getattr(row, column)
        equity_rows.append(equity_row)

    if shares > 0:
        final_row = data.iloc[-1]
        final_date = pd.Timestamp(final_row["date"])
        final_close = float(final_row["close"])
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
        forced_signal = {
            "signal_date": final_date,
            "signal": "FORCED_SELL",
            "execution_price": final_close,
        }
        for column in sorted(strategy.required_columns - {"date", "open", "close"}):
            forced_signal[column] = final_row[column]
        signal_rows.append(format_signal_row(forced_signal, final_date, "FORCED_SELL", final_close))
        shares = 0.0

        equity_rows[-1]["cash"] = cash
        equity_rows[-1]["shares"] = shares
        equity_rows[-1]["equity"] = cash
        equity_rows[-1]["position"] = 0
        equity_rows[-1]["signal"] = "FORCED_SELL"

    return build_result(strategy.name, ticker, initial_cash, data, equity_rows, trades, signal_rows)


def format_signal_row(
    signal_record: dict[str, object],
    execution_date: pd.Timestamp,
    signal: str,
    execution_price: float,
) -> dict[str, object]:
    row = {
        "signal_date": signal_record["signal_date"],
        "execution_date": execution_date,
        "signal": signal,
        "execution_price": execution_price,
    }
    for key, value in signal_record.items():
        if key not in {"date", "signal_date", "signal", "execution_price"}:
            row[key] = value
    return row


def run_rsi_backtest(
    prices: pd.DataFrame,
    ticker: str,
    initial_cash: float = 10_000,
    buy_threshold: float = 30,
    sell_threshold: float = 70,
) -> BacktestResult:
    return run_strategy_backtest(
        prices,
        ticker,
        RSIStrategy(buy_threshold, sell_threshold),
        initial_cash,
    )


def run_ma_crossover_backtest(
    prices: pd.DataFrame,
    ticker: str,
    initial_cash: float = 10_000,
) -> BacktestResult:
    return run_strategy_backtest(prices, ticker, MACrossoverStrategy(), initial_cash)


def run_rsi_parameter_sweep(
    prices: pd.DataFrame,
    ticker: str,
    initial_cash: float = 10_000,
    parameter_pairs: tuple[tuple[int, int], ...] = ((20, 80), (25, 75), (30, 70), (35, 65), (40, 60)),
) -> pd.DataFrame:
    rows = []
    for buy_threshold, sell_threshold in parameter_pairs:
        result = run_rsi_backtest(prices, ticker, initial_cash, buy_threshold, sell_threshold)
        rows.append(
            {
                "buy_rsi": buy_threshold,
                "sell_rsi": sell_threshold,
                "total_return": result.total_return,
                "buy_hold_return": result.buy_hold_return,
                "alpha": result.alpha,
                "total_profit": result.total_profit,
                "win_rate": result.win_rate,
                "max_drawdown": result.max_drawdown,
                "sharpe_ratio": result.sharpe_ratio,
                "trades": result.trade_count,
            }
        )
    return pd.DataFrame(rows).sort_values(["sharpe_ratio", "alpha"], ascending=False)


def best_strategy_label(returns: dict[str, float], tolerance: float = 0.0001) -> str:
    best_return = max(returns.values())
    winners = [name for name, value in returns.items() if abs(value - best_return) <= tolerance]
    return "/".join(winners)


def run_strategy_comparison(
    tickers: list[str],
    initial_cash: float = 10_000,
    db_path: Path = DB_PATH,
    buy_threshold: float = 30,
    sell_threshold: float = 70,
) -> pd.DataFrame:
    rows = []
    for ticker in tickers:
        prices = load_prices(ticker, db_path)
        shared_data = prices.dropna(subset=["date", "open", "close", "rsi_14", "ma_5", "ma_20"]).sort_values("date")
        if len(shared_data) < 3:
            rows.append(
                {
                    "ticker": ticker.upper(),
                    "rsi_return": None,
                    "ma_return": None,
                    "buy_hold_return": None,
                    "best": "Not enough data",
                    "rsi_trades": 0,
                    "ma_trades": 0,
                }
            )
            continue

        rsi_result = run_rsi_backtest(shared_data, ticker, initial_cash, buy_threshold, sell_threshold)
        ma_result = run_ma_crossover_backtest(shared_data, ticker, initial_cash)
        benchmark_return = buy_hold_return(shared_data)
        returns = {
            "RSI": rsi_result.total_return,
            "MA": ma_result.total_return,
            "Buy Hold": benchmark_return,
        }
        rows.append(
            {
                "ticker": ticker.upper(),
                "rsi_return": rsi_result.total_return,
                "ma_return": ma_result.total_return,
                "buy_hold_return": benchmark_return,
                "best": best_strategy_label(returns),
                "rsi_trades": rsi_result.trade_count,
                "ma_trades": ma_result.trade_count,
            }
        )

    return pd.DataFrame(rows).sort_values("ticker")


def format_percent(value: float) -> str:
    return f"{value * 100:.2f}%"


def print_result(result: BacktestResult) -> None:
    print(f"{result.strategy_name} Backtest: {result.ticker}")
    print(f"Ending Equity: ${result.ending_equity:,.2f}")
    print(f"Total Profit: ${result.total_profit:,.2f}")
    print(f"Strategy Return: {format_percent(result.total_return)}")
    print(f"Buy & Hold Return: {format_percent(result.buy_hold_return)}")
    print(f"Alpha: {format_percent(result.alpha)}")
    print(f"Win Rate: {format_percent(result.win_rate)}")
    print(f"Maximum Drawdown: {format_percent(result.max_drawdown)}")
    print(f"Sharpe Ratio: {result.sharpe_ratio:.2f}")
    print(f"Trades: {result.trade_count} ({result.winning_trades} winners)")
    print(f"Signals: {len(result.signals)}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Backtest an RSI strategy using stored historical data.")
    parser.add_argument("ticker", nargs="?", help="Ticker symbol already stored in SQLite, for example: AAPL")
    parser.add_argument("--strategy", choices=sorted(STRATEGY_CLASSES), default="rsi", help="Strategy to backtest")
    parser.add_argument("--cash", type=float, default=10_000, help="Starting cash")
    parser.add_argument("--buy-rsi", type=float, default=30, help="Buy when RSI is below this value")
    parser.add_argument("--sell-rsi", type=float, default=70, help="Sell when RSI is above this value")
    parser.add_argument("--db", type=Path, default=DB_PATH, help="SQLite database path")
    parser.add_argument("--show-signals", action="store_true", help="Print every generated signal")
    parser.add_argument("--sweep", action="store_true", help="Run the RSI parameter sweep")
    parser.add_argument("--compare", nargs="+", help="Compare RSI, MA, and buy-and-hold across tickers")
    return parser


if __name__ == "__main__":
    args = build_parser().parse_args()
    if args.compare:
        comparison = run_strategy_comparison(args.compare, args.cash, args.db, args.buy_rsi, args.sell_rsi)
        print(comparison.to_string(index=False, formatters={
            "rsi_return": lambda value: "N/A" if pd.isna(value) else format_percent(value),
            "ma_return": lambda value: "N/A" if pd.isna(value) else format_percent(value),
            "buy_hold_return": lambda value: "N/A" if pd.isna(value) else format_percent(value),
        }))
        raise SystemExit

    if not args.ticker:
        raise SystemExit("Provide a ticker or use --compare TICKER [TICKER ...].")

    historical_prices = load_prices(args.ticker, args.db)
    if args.sweep:
        sweep = run_rsi_parameter_sweep(historical_prices, args.ticker, args.cash)
        print(sweep.to_string(index=False, formatters={
            "total_return": format_percent,
            "buy_hold_return": format_percent,
            "alpha": format_percent,
            "win_rate": format_percent,
            "max_drawdown": format_percent,
            "total_profit": lambda value: f"${value:,.2f}",
            "sharpe_ratio": lambda value: f"{value:.2f}",
        }))
        raise SystemExit

    strategy = build_strategy(args.strategy, args.buy_rsi, args.sell_rsi)
    backtest_result = run_strategy_backtest(historical_prices, args.ticker, strategy, initial_cash=args.cash)
    print_result(backtest_result)
    if args.show_signals:
        print()
        print(backtest_result.signals.to_string(index=False))

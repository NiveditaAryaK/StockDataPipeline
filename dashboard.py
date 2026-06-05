from __future__ import annotations

import sqlite3
from pathlib import Path

import pandas as pd
import streamlit as st

from backtest import run_rsi_parameter_sweep, run_strategy_backtest, run_strategy_comparison
from main import DB_PATH, load_prices, run_pipeline
from paper_trading import run_paper_trading_simulation
from strategies import STRATEGY_LABELS, build_strategy


st.set_page_config(page_title="Stock Data Pipeline", layout="wide")


def available_tickers(db_path: Path = DB_PATH) -> list[str]:
    if not db_path.exists():
        return []
    with sqlite3.connect(db_path) as conn:
        rows = conn.execute("SELECT DISTINCT ticker FROM stock_prices ORDER BY ticker").fetchall()
    return [row[0] for row in rows]


st.title("Stock Data Pipeline")

tickers = available_tickers()


def sync_refresh_ticker() -> None:
    st.session_state["ticker_input"] = st.session_state["view_ticker"]


if "ticker_input" not in st.session_state:
    st.session_state["ticker_input"] = tickers[0] if tickers else "AAPL MSFT NVDA"

with st.sidebar:
    if tickers:
        selected_tickers = st.multiselect("Compare Tickers", tickers, default=tickers)
        selected_ticker = st.selectbox("View Ticker", tickers, key="view_ticker", on_change=sync_refresh_ticker)
    else:
        selected_tickers = []
        selected_ticker = None

    st.header("Update Data")
    ticker_input = st.text_input("Tickers", key="ticker_input")
    period = st.selectbox("Period", ["1mo", "3mo", "6mo", "1y", "2y", "5y"], index=3)
    if st.button("Download / Refresh", type="primary"):
        tickers = [ticker.strip().upper() for ticker in ticker_input.split() if ticker.strip()]
        with st.spinner("Fetching market data..."):
            run_pipeline(tickers, period)
        st.success("Data refreshed.")

if not tickers:
    st.info("Use the sidebar to download stock data into SQLite.")
    st.stop()

data = load_prices(selected_ticker)

latest = data.dropna(subset=["close"]).iloc[-1]
latest_return = latest["daily_return"] * 100 if pd.notna(latest["daily_return"]) else 0

metric_cols = st.columns(4)
metric_cols[0].metric("Close", f"${latest['close']:.2f}")
metric_cols[1].metric("Daily Return", f"{latest_return:.2f}%")
metric_cols[2].metric("5-Day MA", f"${latest['ma_5']:.2f}" if pd.notna(latest["ma_5"]) else "N/A")
metric_cols[3].metric("RSI 14", f"{latest['rsi_14']:.1f}" if pd.notna(latest["rsi_14"]) else "N/A")

chart_data = data.set_index("date")[["close", "ma_5", "ma_20"]]
st.subheader(f"{selected_ticker} Price and Moving Averages")
st.line_chart(chart_data)

st.subheader("Strategy Comparison")
if selected_tickers:
    comparison = run_strategy_comparison(selected_tickers, initial_cash=10_000)
    display_comparison = comparison.copy()
    for column in ["rsi_return", "ma_return", "buy_hold_return"]:
        display_comparison[column] = (display_comparison[column] * 100).round(2)
    st.dataframe(display_comparison, width="stretch")
else:
    st.info("Select at least one ticker to compare.")

st.subheader("Daily Returns")
st.bar_chart(data.set_index("date")["daily_return"])

st.subheader("Strategy Backtest")
with st.container():
    strategy_key = st.selectbox(
        "Strategy",
        list(STRATEGY_LABELS),
        format_func=lambda key: STRATEGY_LABELS[key],
    )
    col_a, col_b, col_c = st.columns(3)
    initial_cash = col_a.number_input("Initial Cash", min_value=100.0, value=10_000.0, step=500.0)
    buy_rsi = col_b.number_input("Buy RSI", min_value=1.0, max_value=99.0, value=30.0, step=1.0)
    sell_rsi = col_c.number_input("Sell RSI", min_value=1.0, max_value=99.0, value=70.0, step=1.0)

try:
    selected_strategy = build_strategy(strategy_key, buy_rsi, sell_rsi)
    result = run_strategy_backtest(data, selected_ticker, selected_strategy, initial_cash)
except ValueError as error:
    st.warning(str(error))
else:
    backtest_cols = st.columns(5)
    backtest_cols[0].metric("Total Profit", f"${result.total_profit:,.2f}")
    backtest_cols[1].metric("Strategy Return", f"{result.total_return * 100:.2f}%")
    backtest_cols[2].metric("Buy & Hold", f"{result.buy_hold_return * 100:.2f}%")
    backtest_cols[3].metric("Alpha", f"{result.alpha * 100:.2f}%")
    backtest_cols[4].metric("Trades", result.trade_count)

    risk_cols = st.columns(3)
    risk_cols[0].metric("Win Rate", f"{result.win_rate * 100:.2f}%")
    risk_cols[1].metric("Max Drawdown", f"{result.max_drawdown * 100:.2f}%")
    risk_cols[2].metric("Sharpe Ratio", f"{result.sharpe_ratio:.2f}")

    strategy_chart = result.equity_curve[["date", "close", "equity"]].copy()
    strategy_chart["Buy & Hold Equity"] = initial_cash * strategy_chart["close"] / strategy_chart["close"].iloc[0]
    strategy_chart = strategy_chart.rename(columns={"equity": f"{result.strategy_name} Equity"}).set_index("date")
    st.line_chart(strategy_chart[[f"{result.strategy_name} Equity", "Buy & Hold Equity"]], height=320)
    if result.trades.empty:
        st.info("No completed trades for this strategy.")
    else:
        st.dataframe(result.trades.sort_values("exit_date", ascending=False), width="stretch")
    if not result.signals.empty:
        st.subheader("Signal Log")
        st.dataframe(result.signals.sort_values("execution_date", ascending=False), width="stretch")

st.subheader("RSI Parameter Sweep")
try:
    sweep = run_rsi_parameter_sweep(data, selected_ticker, initial_cash)
except ValueError as error:
    st.warning(str(error))
else:
    display_sweep = sweep.copy()
    for column in ["total_return", "buy_hold_return", "alpha", "win_rate", "max_drawdown"]:
        display_sweep[column] = (display_sweep[column] * 100).round(2)
    display_sweep["total_profit"] = display_sweep["total_profit"].round(2)
    display_sweep["sharpe_ratio"] = display_sweep["sharpe_ratio"].round(2)
    st.dataframe(display_sweep, width="stretch")

st.subheader("Paper Trading Simulator")
with st.container():
    paper_cols = st.columns(2)
    paper_cash = paper_cols[0].number_input("Paper Cash", min_value=100.0, value=10_000.0, step=500.0)
    allocation = paper_cols[1].slider("Allocation Per Trade", min_value=0.05, max_value=1.0, value=1.0, step=0.05)

try:
    simulation = run_paper_trading_simulation(
        {ticker: load_prices(ticker) for ticker in selected_tickers or [selected_ticker]},
        selected_strategy,
        initial_cash=paper_cash,
        allocation_per_trade=allocation,
    )
except ValueError as error:
    st.warning(str(error))
else:
    paper_metric_cols = st.columns(5)
    paper_metric_cols[0].metric("Cash", f"${simulation.portfolio.cash:,.2f}")
    paper_metric_cols[1].metric("Positions", f"${simulation.portfolio.positions_value:,.2f}")
    paper_metric_cols[2].metric("Portfolio Value", f"${simulation.portfolio.value:,.2f}")
    paper_metric_cols[3].metric("Realized P&L", f"${simulation.portfolio.realized_pnl:,.2f}")
    paper_metric_cols[4].metric("Unrealized P&L", f"${simulation.portfolio.unrealized_pnl:,.2f}")

    if not simulation.equity_curve.empty:
        st.line_chart(simulation.equity_curve.set_index("date")[["portfolio_value", "cash", "positions_value"]], height=320)
    if simulation.trades.empty:
        st.info("No paper trades generated for this strategy and ticker set.")
    else:
        st.dataframe(simulation.trades.sort_values("date", ascending=False), width="stretch")
    if simulation.positions.empty:
        st.info("No open positions at the end of the simulation.")
    else:
        st.subheader("Open Positions")
        st.dataframe(simulation.positions, width="stretch")

st.subheader("Stored Rows")
st.dataframe(data.sort_values("date", ascending=False), width="stretch")

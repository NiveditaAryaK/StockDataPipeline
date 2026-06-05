from __future__ import annotations

import sqlite3
from pathlib import Path

import pandas as pd
import streamlit as st

from backtest import run_rsi_backtest
from main import DB_PATH, load_prices, run_pipeline


st.set_page_config(page_title="Stock Data Pipeline", layout="wide")


def available_tickers(db_path: Path = DB_PATH) -> list[str]:
    if not db_path.exists():
        return []
    with sqlite3.connect(db_path) as conn:
        rows = conn.execute("SELECT DISTINCT ticker FROM stock_prices ORDER BY ticker").fetchall()
    return [row[0] for row in rows]


st.title("Stock Data Pipeline")

with st.sidebar:
    st.header("Update Data")
    ticker_input = st.text_input("Tickers", value="AAPL MSFT NVDA")
    period = st.selectbox("Period", ["1mo", "3mo", "6mo", "1y", "2y", "5y"], index=3)
    if st.button("Download / Refresh", type="primary"):
        tickers = [ticker.strip().upper() for ticker in ticker_input.split() if ticker.strip()]
        with st.spinner("Fetching market data..."):
            run_pipeline(tickers, period)
        st.success("Data refreshed.")

tickers = available_tickers()
if not tickers:
    st.info("Use the sidebar to download stock data into SQLite.")
    st.stop()

selected_ticker = st.sidebar.selectbox("View Ticker", tickers)
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

st.subheader("Daily Returns")
st.bar_chart(data.set_index("date")["daily_return"])

st.subheader("RSI Strategy Backtest")
with st.container():
    col_a, col_b, col_c = st.columns(3)
    initial_cash = col_a.number_input("Initial Cash", min_value=100.0, value=10_000.0, step=500.0)
    buy_rsi = col_b.number_input("Buy RSI", min_value=1.0, max_value=99.0, value=30.0, step=1.0)
    sell_rsi = col_c.number_input("Sell RSI", min_value=1.0, max_value=99.0, value=70.0, step=1.0)

try:
    result = run_rsi_backtest(data, selected_ticker, initial_cash, buy_rsi, sell_rsi)
except ValueError as error:
    st.warning(str(error))
else:
    backtest_cols = st.columns(5)
    backtest_cols[0].metric("Total Profit", f"${result.total_profit:,.2f}")
    backtest_cols[1].metric("Win Rate", f"{result.win_rate * 100:.2f}%")
    backtest_cols[2].metric("Max Drawdown", f"{result.max_drawdown * 100:.2f}%")
    backtest_cols[3].metric("Sharpe Ratio", f"{result.sharpe_ratio:.2f}")
    backtest_cols[4].metric("Trades", result.trade_count)

    st.line_chart(result.equity_curve.set_index("date")["equity"])
    if result.trades.empty:
        st.info("No completed trades for these RSI thresholds.")
    else:
        st.dataframe(result.trades.sort_values("exit_date", ascending=False), width="stretch")
    if not result.signals.empty:
        st.subheader("Signal Log")
        st.dataframe(result.signals.sort_values("execution_date", ascending=False), width="stretch")

st.subheader("Stored Rows")
st.dataframe(data.sort_values("date", ascending=False), width="stretch")

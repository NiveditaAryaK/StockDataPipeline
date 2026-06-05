from __future__ import annotations

import sqlite3
from pathlib import Path

import pandas as pd
import streamlit as st

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

st.subheader("Stored Rows")
st.dataframe(data.sort_values("date", ascending=False), use_container_width=True)

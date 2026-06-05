# Stock Data Pipeline

Download Yahoo Finance market data, store it in SQLite, calculate indicators, backtest an RSI strategy, and view the result in a Streamlit dashboard.

## Setup

```bash
source .venv/bin/activate
pip install -r requirements.txt
```

## Run the Pipeline

```bash
python main.py AAPL MSFT NVDA --period 1y
```

Data is saved to `data/market_data.sqlite`.

## Start the Dashboard

```bash
streamlit run dashboard.py
```

## Run the Backtest

```bash
python backtest.py AAPL --cash 10000 --buy-rsi 30 --sell-rsi 70
```

The built-in strategy buys when RSI is below 30 and sells when RSI is above 70. Signals are generated from the previous trading day's RSI and executed on the next trading day's open. If a position is still open at the end of the data, it is closed at the final close and recorded as `end_of_data`.

To inspect every signal:

```bash
python backtest.py AAPL --show-signals
```

## Backtest Metrics

- Total profit
- Win rate
- Maximum drawdown
- Sharpe ratio
- Trade count

## Indicators

- Daily returns
- 5-day moving average
- 20-day moving average
- 14-day RSI

# Stock Data Pipeline

Download Yahoo Finance market data, store it in SQLite, calculate indicators, backtest multiple strategies, and view the result in a Streamlit dashboard.

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

Run the moving-average crossover strategy:

```bash
python backtest.py AAPL --strategy ma
```

Run another registered strategy:

```bash
python backtest.py AAPL --strategy momentum
python backtest.py AAPL --strategy bollinger
```

Run an RSI parameter sweep:

```bash
python backtest.py AAPL --sweep
```

Compare strategies across tickers:

```bash
python backtest.py --compare AAPL MSFT NVDA META GOOGL SPY
```

Run 5-year and 10-year strategy research:

```bash
python research.py --refresh
```

Results are saved to:

- `research_results/strategy_results.csv`
- `research_results/strategy_results.sqlite`

## Backtest Metrics

- Total profit
- Strategy return
- Buy-and-hold return
- Alpha
- Win rate
- Maximum drawdown
- Sharpe ratio
- Trade count
- Strategy comparison table

## Strategy Classes

Strategies implement:

```python
class Strategy:
    def generate_signals(self, prices):
        ...
```

Current registered strategies:

- `RSIStrategy`
- `MACrossoverStrategy`
- `MomentumStrategy`
- `BollingerBandStrategy`

The backtest engine is shared. A strategy only needs to generate `BUY`, `SELL`, or `HOLD` signals.

The Bollinger strategy is mean-reversion based: it buys below the lower band and sells when price recovers above the middle band.

## Indicators

- Daily returns
- 5-day moving average
- 20-day moving average
- 14-day RSI

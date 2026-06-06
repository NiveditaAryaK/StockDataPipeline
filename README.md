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

## Paper Trading Simulator

The dashboard includes a historical paper trading simulator:

```text
Market Feed -> Strategy Engine -> Paper Broker -> Portfolio -> Dashboard
```

It tracks:

- Cash balance
- Open positions
- Trade execution
- Portfolio value
- Realized P&L
- Unrealized P&L

## Agent Quant Researcher

Run the automated research agent:

```bash
python research_agent.py AAPL --period-years 5 --iterations 10
```

The agent:

```text
Strategy Generator -> Backtest Engine -> Metrics Analyzer -> Strategy Ranking
```

It starts with a broad family scan, then generates variations around the best-performing candidates. For example, if `Momentum 20D` wins the seed scan, it will test shorter and longer momentum lookbacks next.

It saves experiments and results to `data/agent_research.sqlite`.

Tables:

- `experiments`
- `experiment_results`

## ML Prediction Engine

Phase 1 builds a supervised-learning dataset for predicting whether tomorrow's close is higher than today's close:

```bash
python ml_dataset.py AAPL --output research_results/aapl_ml_dataset.csv
```

Current feature set:

- OHLCV
- MA5, MA20, RSI14, daily return
- Momentum 5D, momentum 10D
- Volatility 10D, volatility 20D
- MA ratio
- Price distance from MA20
- RSI change
- Volume change

Phase 2 trains the first baseline model:

```bash
python ml_training.py AAPL --model logistic --test-size 0.2
```

This uses Logistic Regression with a chronological split. The older rows are used for training and the newest rows are kept as unseen test data.

Train the first non-linear model:

```bash
python ml_training.py AAPL --model random_forest --test-size 0.2
```

Random Forest can learn interaction effects between indicators, such as momentum behaving differently during high-volatility periods.

Train the gradient-boosted tree model:

```bash
python ml_training.py AAPL --model xgboost --test-size 0.2
```

XGBoost builds trees sequentially, with each new tree trying to correct mistakes from the previous trees.

Compare a model across tickers:

```bash
python ml_training.py --model xgboost --compare AAPL MSFT NVDA META AMZN GOOGL SPY TSLA --threshold 0.25
```

This prints ticker-level return, buy-and-hold return, alpha, and summary alpha statistics.

Current ML metrics:

- Accuracy
- Precision
- Recall
- F1 score
- AUC ROC
- Latest probability up
- Feature importance from model coefficients
- Probability distribution diagnostics
- Threshold win-rate sweep
- One-day threshold backtest

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

Strategy classes live in `strategies.py` and implement:

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

## Module Layout

- `main.py`: market data download, indicators, SQLite persistence
- `strategies.py`: reusable strategy classes and strategy registry
- `backtest.py`: backtest engine, metrics, CLI
- `research.py`: 5-year and 10-year batch research runner
- `research_agent.py`: automated strategy generation, evaluation, and ranking
- `ml_dataset.py`: ML feature and target dataset builder
- `ml_training.py`: ML model training, prediction, and classification metrics
- `paper_trading.py`: market feed, strategy engine, paper broker, portfolio
- `dashboard.py`: Streamlit dashboard

## Indicators

- Daily returns
- 5-day moving average
- 20-day moving average
- 14-day RSI

# Stock Data Pipeline

Download Yahoo Finance market data, store it in SQLite, calculate indicators, and view the result in a Streamlit dashboard.

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

## Indicators

- Daily returns
- 5-day moving average
- 20-day moving average
- 14-day RSI

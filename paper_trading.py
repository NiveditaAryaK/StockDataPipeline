from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from backtest import Strategy


@dataclass
class Position:
    ticker: str
    shares: float
    average_price: float
    last_price: float

    @property
    def market_value(self) -> float:
        return self.shares * self.last_price

    @property
    def unrealized_pnl(self) -> float:
        return (self.last_price - self.average_price) * self.shares


@dataclass(frozen=True)
class Order:
    date: pd.Timestamp
    ticker: str
    side: str
    price: float


@dataclass(frozen=True)
class Fill:
    date: pd.Timestamp
    ticker: str
    side: str
    shares: float
    price: float
    cash_after: float
    realized_pnl: float


class MarketFeed:
    def __init__(self, prices_by_ticker: dict[str, pd.DataFrame]) -> None:
        self.prices_by_ticker = {
            ticker.upper(): prices.sort_values("date").reset_index(drop=True)
            for ticker, prices in prices_by_ticker.items()
        }

    def dates(self) -> list[pd.Timestamp]:
        return sorted(
            {
                pd.Timestamp(date)
                for prices in self.prices_by_ticker.values()
                for date in prices["date"].dropna()
            }
        )

    def latest_prices(self, date: pd.Timestamp) -> dict[str, float]:
        prices = {}
        for ticker, history in self.prices_by_ticker.items():
            available = history[history["date"] <= date]
            if not available.empty:
                prices[ticker] = float(available.iloc[-1]["close"])
        return prices


class StrategyEngine:
    def __init__(self, strategy: Strategy) -> None:
        self.strategy = strategy

    def generate_orders(self, ticker: str, prices: pd.DataFrame) -> pd.DataFrame:
        required_columns = {"date", "open", "close"}.union(self.strategy.required_columns)
        data = prices.dropna(subset=sorted(required_columns)).sort_values("date").reset_index(drop=True)
        if len(data) < self.strategy.minimum_rows:
            return pd.DataFrame(columns=["date", "ticker", "side", "price"])

        signals = self.strategy.generate_signals(data)
        if signals.empty:
            return pd.DataFrame(columns=["date", "ticker", "side", "price"])

        orders = signals[signals["signal"].isin(["BUY", "SELL"])].copy()
        if orders.empty:
            return pd.DataFrame(columns=["date", "ticker", "side", "price"])

        orders["ticker"] = ticker.upper()
        orders["side"] = orders["signal"]
        orders["price"] = orders["execution_price"]
        return orders[["date", "ticker", "side", "price"]]


class Portfolio:
    def __init__(self, initial_cash: float) -> None:
        self.initial_cash = float(initial_cash)
        self.cash = float(initial_cash)
        self.realized_pnl = 0.0
        self.positions: dict[str, Position] = {}

    def buy(self, ticker: str, shares: float, price: float) -> None:
        cost = shares * price
        if cost > self.cash and cost - self.cash > 0.01:
            raise ValueError("Insufficient cash.")
        if cost > self.cash:
            shares = self.cash / price
            cost = self.cash

        position = self.positions.get(ticker)
        if position:
            total_shares = position.shares + shares
            position.average_price = ((position.shares * position.average_price) + cost) / total_shares
            position.shares = total_shares
            position.last_price = price
        else:
            self.positions[ticker] = Position(ticker, shares, price, price)
        self.cash -= cost

    def sell_all(self, ticker: str, price: float) -> tuple[float, float]:
        position = self.positions.get(ticker)
        if not position:
            return 0.0, 0.0

        shares = position.shares
        proceeds = shares * price
        realized = (price - position.average_price) * shares
        self.cash += proceeds
        self.realized_pnl += realized
        del self.positions[ticker]
        return shares, realized

    def update_prices(self, prices: dict[str, float]) -> None:
        for ticker, price in prices.items():
            if ticker in self.positions:
                self.positions[ticker].last_price = price

    @property
    def positions_value(self) -> float:
        return sum(position.market_value for position in self.positions.values())

    @property
    def unrealized_pnl(self) -> float:
        return sum(position.unrealized_pnl for position in self.positions.values())

    @property
    def value(self) -> float:
        return self.cash + self.positions_value

    def positions_frame(self) -> pd.DataFrame:
        rows = [
            {
                "ticker": position.ticker,
                "shares": position.shares,
                "average_price": position.average_price,
                "last_price": position.last_price,
                "market_value": position.market_value,
                "unrealized_pnl": position.unrealized_pnl,
            }
            for position in self.positions.values()
        ]
        return pd.DataFrame(rows)


class PaperBroker:
    def __init__(self, portfolio: Portfolio, allocation_per_trade: float = 1.0) -> None:
        self.portfolio = portfolio
        self.allocation_per_trade = allocation_per_trade
        self.fills: list[Fill] = []

    def execute(self, order: Order) -> None:
        if order.side == "BUY" and order.ticker not in self.portfolio.positions:
            cash_to_spend = self.portfolio.cash * self.allocation_per_trade
            if cash_to_spend <= 0:
                return
            shares = cash_to_spend / order.price
            self.portfolio.buy(order.ticker, shares, order.price)
            realized_pnl = 0.0
        elif order.side == "SELL" and order.ticker in self.portfolio.positions:
            shares, realized_pnl = self.portfolio.sell_all(order.ticker, order.price)
        else:
            return

        self.fills.append(
            Fill(
                date=order.date,
                ticker=order.ticker,
                side=order.side,
                shares=shares,
                price=order.price,
                cash_after=self.portfolio.cash,
                realized_pnl=realized_pnl,
            )
        )


@dataclass(frozen=True)
class PaperTradingResult:
    portfolio: Portfolio
    equity_curve: pd.DataFrame
    trades: pd.DataFrame
    positions: pd.DataFrame


def run_paper_trading_simulation(
    prices_by_ticker: dict[str, pd.DataFrame],
    strategy: Strategy,
    initial_cash: float = 10_000,
    allocation_per_trade: float = 1.0,
) -> PaperTradingResult:
    feed = MarketFeed(prices_by_ticker)
    engine = StrategyEngine(strategy)
    portfolio = Portfolio(initial_cash)
    broker = PaperBroker(portfolio, allocation_per_trade)

    order_frames = [
        engine.generate_orders(ticker, prices)
        for ticker, prices in prices_by_ticker.items()
    ]
    order_frames = [orders for orders in order_frames if not orders.empty]
    orders = (
        pd.concat(order_frames, ignore_index=True).sort_values(["date", "ticker"])
        if order_frames
        else pd.DataFrame(columns=["date", "ticker", "side", "price"])
    )

    equity_rows = []
    for date in feed.dates():
        todays_orders = orders[orders["date"] == date]
        for order in todays_orders.itertuples(index=False):
            broker.execute(Order(pd.Timestamp(order.date), order.ticker, order.side, float(order.price)))

        portfolio.update_prices(feed.latest_prices(date))
        equity_rows.append(
            {
                "date": date,
                "cash": portfolio.cash,
                "positions_value": portfolio.positions_value,
                "portfolio_value": portfolio.value,
                "realized_pnl": portfolio.realized_pnl,
                "unrealized_pnl": portfolio.unrealized_pnl,
            }
        )

    trades = pd.DataFrame([fill.__dict__ for fill in broker.fills])
    return PaperTradingResult(portfolio, pd.DataFrame(equity_rows), trades, portfolio.positions_frame())

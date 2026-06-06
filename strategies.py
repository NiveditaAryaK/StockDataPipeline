from __future__ import annotations

from abc import ABC, abstractmethod

import pandas as pd


class Strategy(ABC):
    name: str
    required_columns: set[str]
    minimum_rows: int = 2

    @abstractmethod
    def generate_signals(self, prices: pd.DataFrame) -> pd.DataFrame:
        """Return date, signal_date, signal, execution_price, and optional diagnostics."""


class RSIStrategy(Strategy):
    required_columns = {"date", "open", "close", "rsi_14"}
    minimum_rows = 2

    def __init__(self, buy_threshold: float = 30, sell_threshold: float = 70) -> None:
        self.buy_threshold = buy_threshold
        self.sell_threshold = sell_threshold
        self.name = f"RSI {buy_threshold:g}/{sell_threshold:g}"

    def generate_signals(self, prices: pd.DataFrame) -> pd.DataFrame:
        rows = []
        for index in range(1, len(prices)):
            previous = prices.iloc[index - 1]
            current = prices.iloc[index]
            if previous["rsi_14"] < self.buy_threshold:
                signal = "BUY"
            elif previous["rsi_14"] > self.sell_threshold:
                signal = "SELL"
            else:
                signal = "HOLD"
            rows.append(
                {
                    "date": current["date"],
                    "signal_date": previous["date"],
                    "signal": signal,
                    "rsi_14": previous["rsi_14"],
                    "execution_price": current["open"],
                }
            )
        return pd.DataFrame(rows)


class MACrossoverStrategy(Strategy):
    required_columns = {"date", "open", "close"}

    def __init__(self, fast_window: int = 5, slow_window: int = 20) -> None:
        if fast_window >= slow_window:
            raise ValueError("Fast MA window must be smaller than slow MA window.")
        self.fast_window = fast_window
        self.slow_window = slow_window
        self.minimum_rows = slow_window + 2
        self.name = f"MA {fast_window}/{slow_window} Crossover"

    def generate_signals(self, prices: pd.DataFrame) -> pd.DataFrame:
        data = prices.copy()
        fast_column = f"ma_{self.fast_window}"
        slow_column = f"ma_{self.slow_window}"
        data[fast_column] = data["close"].rolling(self.fast_window).mean()
        data[slow_column] = data["close"].rolling(self.slow_window).mean()

        rows = []
        for index in range(self.slow_window + 1, len(data)):
            before_previous = data.iloc[index - 2]
            previous = data.iloc[index - 1]
            current = data.iloc[index]
            crossed_above = before_previous[fast_column] <= before_previous[slow_column] and previous[fast_column] > previous[slow_column]
            crossed_below = before_previous[fast_column] >= before_previous[slow_column] and previous[fast_column] < previous[slow_column]
            if crossed_above:
                signal = "BUY"
            elif crossed_below:
                signal = "SELL"
            else:
                signal = "HOLD"
            rows.append(
                {
                    "date": current["date"],
                    "signal_date": previous["date"],
                    "signal": signal,
                    fast_column: previous[fast_column],
                    slow_column: previous[slow_column],
                    "execution_price": current["open"],
                }
            )
        return pd.DataFrame(rows)


class MomentumStrategy(Strategy):
    required_columns = {"date", "open", "close"}

    def __init__(self, lookback_days: int = 20) -> None:
        self.lookback_days = lookback_days
        self.minimum_rows = lookback_days + 2
        self.name = f"Momentum {lookback_days}D"

    def generate_signals(self, prices: pd.DataFrame) -> pd.DataFrame:
        rows = []
        for index in range(self.lookback_days + 1, len(prices)):
            previous = prices.iloc[index - 1]
            lookback = prices.iloc[index - 1 - self.lookback_days]
            current = prices.iloc[index]
            if previous["close"] > lookback["close"]:
                signal = "BUY"
            elif previous["close"] < lookback["close"]:
                signal = "SELL"
            else:
                signal = "HOLD"
            rows.append(
                {
                    "date": current["date"],
                    "signal_date": previous["date"],
                    "signal": signal,
                    "momentum_return": previous["close"] / lookback["close"] - 1,
                    "execution_price": current["open"],
                }
            )
        return pd.DataFrame(rows)


class BollingerBandStrategy(Strategy):
    required_columns = {"date", "open", "close"}

    def __init__(self, window: int = 20, standard_deviations: float = 2) -> None:
        self.window = window
        self.standard_deviations = standard_deviations
        self.minimum_rows = window + 2
        self.name = f"Bollinger {window}D/{standard_deviations:g}SD"

    def generate_signals(self, prices: pd.DataFrame) -> pd.DataFrame:
        data = prices.copy()
        rolling_mean = data["close"].rolling(self.window).mean()
        rolling_std = data["close"].rolling(self.window).std()
        data["middle_band"] = rolling_mean
        data["lower_band"] = rolling_mean - self.standard_deviations * rolling_std
        data["upper_band"] = rolling_mean + self.standard_deviations * rolling_std

        rows = []
        for index in range(self.window, len(data)):
            previous = data.iloc[index - 1]
            current = data.iloc[index]
            if previous["close"] < previous["lower_band"]:
                signal = "BUY"
            elif previous["close"] > previous["middle_band"]:
                signal = "SELL"
            else:
                signal = "HOLD"
            rows.append(
                {
                    "date": current["date"],
                    "signal_date": previous["date"],
                    "signal": signal,
                    "middle_band": previous["middle_band"],
                    "lower_band": previous["lower_band"],
                    "upper_band": previous["upper_band"],
                    "execution_price": current["open"],
                }
            )
        return pd.DataFrame(rows)


STRATEGY_CLASSES = {
    "rsi": RSIStrategy,
    "ma": MACrossoverStrategy,
    "momentum": MomentumStrategy,
    "bollinger": BollingerBandStrategy,
}


STRATEGY_LABELS = {
    "rsi": "RSI",
    "ma": "MA 5/20 Crossover",
    "momentum": "Momentum",
    "bollinger": "Bollinger Bands",
}


def build_strategy(strategy_key: str, buy_rsi: float = 30, sell_rsi: float = 70) -> Strategy:
    if strategy_key == "rsi":
        return RSIStrategy(buy_rsi, sell_rsi)
    return STRATEGY_CLASSES[strategy_key]()

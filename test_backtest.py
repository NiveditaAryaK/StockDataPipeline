from __future__ import annotations

import unittest

import pandas as pd

from backtest import run_rsi_backtest


class BacktestTests(unittest.TestCase):
    def test_rsi_strategy_calculates_core_metrics(self) -> None:
        prices = pd.DataFrame(
            {
                "date": pd.date_range("2024-01-01", periods=8, freq="D"),
                "open": [100, 96, 91, 101, 111, 121, 116, 131],
                "close": [100, 95, 90, 100, 110, 120, 115, 130],
                "rsi_14": [50, 25, 20, 45, 72, 50, 28, 75],
            }
        )

        result = run_rsi_backtest(prices, "TEST", initial_cash=1_000)

        self.assertEqual(result.trade_count, 2)
        self.assertEqual(result.winning_trades, 1)
        self.assertAlmostEqual(result.win_rate, 0.5)
        self.assertGreater(result.total_profit, 0)
        self.assertLessEqual(result.max_drawdown, 0)
        self.assertIn("equity", result.equity_curve.columns)
        self.assertEqual(len(result.signals), 4)

    def test_backtest_records_open_position_at_end(self) -> None:
        prices = pd.DataFrame(
            {
                "date": pd.date_range("2024-01-01", periods=4, freq="D"),
                "open": [100, 96, 95, 94],
                "close": [100, 95, 94, 90],
                "rsi_14": [50, 25, 20, 22],
            }
        )

        result = run_rsi_backtest(prices, "TEST", initial_cash=1_000)

        self.assertEqual(result.trade_count, 1)
        self.assertEqual(result.winning_trades, 0)
        self.assertEqual(result.win_rate, 0)
        self.assertEqual(result.trades.iloc[0]["exit_reason"], "end_of_data")
        self.assertLess(result.total_profit, 0)

    def test_backtest_rejects_missing_columns(self) -> None:
        prices = pd.DataFrame({"date": pd.date_range("2024-01-01", periods=2), "close": [10, 11]})

        with self.assertRaises(ValueError):
            run_rsi_backtest(prices, "TEST")


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import unittest

import pandas as pd

from backtest import best_strategy_label, run_ma_crossover_backtest, run_rsi_backtest, run_rsi_parameter_sweep


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
        self.assertAlmostEqual(result.alpha, result.total_return - result.buy_hold_return)
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

    def test_ma_crossover_strategy_records_trades(self) -> None:
        prices = pd.DataFrame(
            {
                "date": pd.date_range("2024-01-01", periods=8, freq="D"),
                "open": [100, 101, 103, 104, 103, 102, 101, 100],
                "close": [100, 102, 104, 103, 102, 101, 100, 99],
                "ma_5": [9, 9, 11, 12, 11, 9, 8, 8],
                "ma_20": [10, 10, 10, 10, 10, 10, 10, 10],
            }
        )

        result = run_ma_crossover_backtest(prices, "TEST", initial_cash=1_000)

        self.assertEqual(result.trade_count, 1)
        self.assertEqual(len(result.signals), 2)
        self.assertEqual(result.strategy_name, "MA 5/20 Crossover")

    def test_rsi_parameter_sweep_returns_parameter_rows(self) -> None:
        prices = pd.DataFrame(
            {
                "date": pd.date_range("2024-01-01", periods=8, freq="D"),
                "open": [100, 96, 91, 101, 111, 121, 116, 131],
                "close": [100, 95, 90, 100, 110, 120, 115, 130],
                "rsi_14": [50, 25, 20, 45, 72, 50, 28, 75],
            }
        )

        sweep = run_rsi_parameter_sweep(prices, "TEST", initial_cash=1_000)

        self.assertEqual(len(sweep), 5)
        self.assertIn("alpha", sweep.columns)
        self.assertIn("sharpe_ratio", sweep.columns)

    def test_best_strategy_label_handles_ties(self) -> None:
        label = best_strategy_label({"RSI": 0.1, "MA": 0.05, "Buy Hold": 0.1})

        self.assertEqual(label, "RSI/Buy Hold")


if __name__ == "__main__":
    unittest.main()

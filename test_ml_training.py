from __future__ import annotations

import unittest

import pandas as pd

from ml_dataset import build_ml_dataset
from ml_training import chronological_split, probability_to_signal, train_logistic_regression


def make_training_dataset(rows: int = 120) -> pd.DataFrame:
    close = pd.Series([100 + (index * 0.05) + ((index % 9) - 4) * 0.4 for index in range(rows)])
    prices = pd.DataFrame(
        {
            "ticker": ["TEST"] * rows,
            "date": pd.date_range("2023-01-01", periods=rows, freq="D"),
            "open": close + 0.1,
            "high": close + 1.0,
            "low": close - 1.0,
            "close": close,
            "volume": [1000 + ((index % 11) * 25) for index in range(rows)],
            "daily_return": close.pct_change(),
            "ma_5": close.rolling(5).mean(),
            "ma_20": close.rolling(20).mean(),
            "rsi_14": [45 + ((index % 12) * 1.5) for index in range(rows)],
        }
    )
    return build_ml_dataset(prices)


class MLTrainingTests(unittest.TestCase):
    def test_chronological_split_keeps_future_rows_in_test_set(self) -> None:
        dataset = make_training_dataset()

        train_df, test_df = chronological_split(dataset, test_size=0.25)

        self.assertLess(train_df["date"].max(), test_df["date"].min())
        self.assertEqual(len(train_df) + len(test_df), len(dataset))

    def test_probability_to_signal_uses_thresholds(self) -> None:
        self.assertEqual(probability_to_signal(0.60), "BUY")
        self.assertEqual(probability_to_signal(0.40), "SELL")
        self.assertEqual(probability_to_signal(0.50), "HOLD")

    def test_train_logistic_regression_returns_metrics_and_importance(self) -> None:
        dataset = make_training_dataset()

        result = train_logistic_regression(dataset, "TEST", test_size=0.25)

        self.assertEqual(result.model_name, "Logistic Regression")
        self.assertGreater(result.train_rows, result.test_rows)
        self.assertTrue(0 <= result.accuracy <= 1)
        self.assertTrue(0 <= result.precision <= 1)
        self.assertTrue(0 <= result.recall <= 1)
        self.assertTrue(0 <= result.f1 <= 1)
        self.assertTrue(0 <= result.latest_probability_up <= 1)
        self.assertIn(result.latest_signal, {"BUY", "SELL", "HOLD"})
        self.assertIn("probability_up", result.predictions.columns)
        self.assertIn("predicted_up", result.predictions.columns)
        self.assertIn("feature", result.feature_importance.columns)
        self.assertFalse(result.feature_importance.empty)


if __name__ == "__main__":
    unittest.main()

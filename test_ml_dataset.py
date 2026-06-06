from __future__ import annotations

import unittest

import pandas as pd

from ml_dataset import FEATURE_COLUMNS, build_ml_dataset


class MLDatasetTests(unittest.TestCase):
    def test_build_ml_dataset_creates_features_and_target(self) -> None:
        rows = 40
        prices = pd.DataFrame(
            {
                "ticker": ["TEST"] * rows,
                "date": pd.date_range("2024-01-01", periods=rows, freq="D"),
                "open": [100 + index for index in range(rows)],
                "high": [101 + index for index in range(rows)],
                "low": [99 + index for index in range(rows)],
                "close": [100 + index for index in range(rows)],
                "volume": [1000 + index for index in range(rows)],
                "daily_return": pd.Series([100 + index for index in range(rows)]).pct_change(),
                "ma_5": pd.Series([100 + index for index in range(rows)]).rolling(5).mean(),
                "ma_20": pd.Series([100 + index for index in range(rows)]).rolling(20).mean(),
                "rsi_14": [50 + (index % 5) for index in range(rows)],
            }
        )

        dataset = build_ml_dataset(prices)

        self.assertFalse(dataset.empty)
        self.assertTrue(set(FEATURE_COLUMNS).issubset(dataset.columns))
        self.assertIn("target_up", dataset.columns)
        self.assertTrue(dataset["target_up"].isin([0, 1]).all())
        self.assertEqual(dataset["target_up"].iloc[-1], 1)


if __name__ == "__main__":
    unittest.main()

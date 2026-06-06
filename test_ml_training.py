from __future__ import annotations

import unittest
from unittest.mock import patch

import pandas as pd

from ml_dataset import build_ml_dataset
from ml_training import (
    MODEL_TRAINERS,
    MLTrainingResult,
    compare_model_across_tickers,
    chronological_split,
    probability_to_signal,
    train_logistic_regression,
    train_random_forest,
    train_xgboost,
    walk_forward_validation,
)


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
        self.assertTrue(0 <= result.actual_up_rate <= 1)
        self.assertTrue(0 <= result.predicted_up_rate <= 1)
        self.assertTrue(0 <= result.probability_min <= result.probability_max <= 1)
        self.assertTrue(0 <= result.probability_avg <= 1)
        self.assertAlmostEqual(
            result.buy_signal_rate + result.hold_signal_rate + result.sell_signal_rate,
            1.0,
        )
        self.assertTrue(0 <= result.latest_probability_up <= 1)
        self.assertIn(result.latest_signal, {"BUY", "SELL", "HOLD"})
        self.assertIn("probability_up", result.predictions.columns)
        self.assertIn("predicted_up", result.predictions.columns)
        self.assertIn("feature", result.feature_importance.columns)
        self.assertFalse(result.feature_importance.empty)
        self.assertIn("threshold", result.threshold_sweep.columns)
        self.assertIn("buy_win_rate", result.threshold_sweep.columns)
        self.assertIn("sell_win_rate", result.threshold_sweep.columns)
        self.assertFalse(result.threshold_sweep.empty)
        self.assertIn("total_return", result.threshold_backtest.columns)
        self.assertIn("buy_hold_return", result.threshold_backtest.columns)
        self.assertIn("sharpe_ratio", result.threshold_backtest.columns)
        self.assertFalse(result.threshold_backtest.empty)

    def test_train_random_forest_returns_metrics_and_importance(self) -> None:
        dataset = make_training_dataset()

        result = train_random_forest(dataset, "TEST", test_size=0.25)

        self.assertEqual(result.model_name, "Random Forest")
        self.assertTrue(0 <= result.accuracy <= 1)
        self.assertTrue(0 <= result.actual_up_rate <= 1)
        self.assertTrue(0 <= result.predicted_up_rate <= 1)
        self.assertTrue(0 <= result.probability_min <= result.probability_max <= 1)
        self.assertTrue(0 <= result.probability_avg <= 1)
        self.assertAlmostEqual(
            result.buy_signal_rate + result.hold_signal_rate + result.sell_signal_rate,
            1.0,
        )
        self.assertTrue(0 <= result.latest_probability_up <= 1)
        self.assertIn("probability_up", result.predictions.columns)
        self.assertIn("feature", result.feature_importance.columns)
        self.assertIn("importance", result.feature_importance.columns)
        self.assertFalse(result.feature_importance.empty)
        self.assertIn("threshold", result.threshold_sweep.columns)
        self.assertIn("buy_win_rate", result.threshold_sweep.columns)
        self.assertIn("sell_win_rate", result.threshold_sweep.columns)
        self.assertFalse(result.threshold_sweep.empty)
        self.assertIn("total_return", result.threshold_backtest.columns)
        self.assertIn("buy_hold_return", result.threshold_backtest.columns)
        self.assertIn("sharpe_ratio", result.threshold_backtest.columns)
        self.assertFalse(result.threshold_backtest.empty)

    def test_train_xgboost_returns_metrics_and_importance(self) -> None:
        dataset = make_training_dataset()

        result = train_xgboost(dataset, "TEST", test_size=0.25)

        self.assertEqual(result.model_name, "XGBoost")
        self.assertTrue(0 <= result.accuracy <= 1)
        self.assertTrue(0 <= result.actual_up_rate <= 1)
        self.assertTrue(0 <= result.predicted_up_rate <= 1)
        self.assertTrue(0 <= result.probability_min <= result.probability_max <= 1)
        self.assertTrue(0 <= result.probability_avg <= 1)
        self.assertAlmostEqual(
            result.buy_signal_rate + result.hold_signal_rate + result.sell_signal_rate,
            1.0,
        )
        self.assertTrue(0 <= result.latest_probability_up <= 1)
        self.assertIn("probability_up", result.predictions.columns)
        self.assertIn("feature", result.feature_importance.columns)
        self.assertIn("importance", result.feature_importance.columns)
        self.assertFalse(result.feature_importance.empty)
        self.assertIn("threshold", result.threshold_sweep.columns)
        self.assertIn("total_return", result.threshold_backtest.columns)
        self.assertFalse(result.threshold_backtest.empty)

    def test_model_trainers_exposes_supported_models(self) -> None:
        self.assertIn("logistic", MODEL_TRAINERS)
        self.assertIn("random_forest", MODEL_TRAINERS)
        self.assertIn("xgboost", MODEL_TRAINERS)

    def test_compare_model_across_tickers_returns_alpha_summary(self) -> None:
        def fake_trainer(dataset, ticker, test_size=0.2):
            alpha = {"AAA": 0.10, "BBB": -0.02}[ticker]
            return MLTrainingResult(
                ticker=ticker,
                model_name="Fake Model",
                train_rows=10,
                test_rows=5,
                train_start=pd.Timestamp("2023-01-01"),
                train_end=pd.Timestamp("2023-01-10"),
                test_start=pd.Timestamp("2023-01-11"),
                test_end=pd.Timestamp("2023-01-15"),
                accuracy=0.5,
                precision=0.5,
                recall=0.5,
                f1=0.5,
                auc_roc=0.55,
                actual_up_rate=0.5,
                predicted_up_rate=0.5,
                probability_min=0.1,
                probability_25pct=0.2,
                probability_avg=0.3,
                probability_75pct=0.4,
                probability_max=0.5,
                buy_signal_rate=0.2,
                hold_signal_rate=0.6,
                sell_signal_rate=0.2,
                latest_probability_up=0.4,
                latest_signal="HOLD",
                predictions=pd.DataFrame(),
                feature_importance=pd.DataFrame({"feature": ["x"], "importance": [1.0]}),
                threshold_sweep=pd.DataFrame(),
                threshold_backtest=pd.DataFrame(
                    [
                        {
                            "threshold": 0.25,
                            "trades": 3,
                            "win_rate": 2 / 3,
                            "total_return": 0.20 + alpha,
                            "buy_hold_return": 0.20,
                            "alpha": alpha,
                            "max_drawdown": -0.05,
                            "sharpe_ratio": 1.0,
                        }
                    ]
                ),
                pipeline=None,
            )

        with patch("ml_training.build_dataset_for_ticker", return_value=pd.DataFrame()):
            with patch.dict("ml_training.MODEL_TRAINERS", {"fake": fake_trainer}):
                comparison, summary = compare_model_across_tickers(["AAA", "BBB"], model_key="fake")

        self.assertEqual(list(comparison["ticker"]), ["AAA", "BBB"])
        self.assertAlmostEqual(summary["mean_alpha"], 0.04)
        self.assertAlmostEqual(summary["median_alpha"], 0.04)
        self.assertAlmostEqual(summary["positive_alpha_rate"], 0.5)

    def test_walk_forward_validation_returns_window_summary(self) -> None:
        dataset = make_training_dataset(rows=365 * 8)

        results, summary = walk_forward_validation(
            dataset,
            "TEST",
            model_key="logistic",
            threshold=0.25,
            train_years=3,
            test_years=1,
            step_years=1,
        )

        self.assertGreater(len(results), 1)
        self.assertIn("train_window", results.columns)
        self.assertIn("test_window", results.columns)
        self.assertIn("alpha", results.columns)
        self.assertGreater(summary["windows"], 1)
        self.assertIn("compounded_alpha", summary)
        self.assertIn("positive_alpha_rate", summary)


if __name__ == "__main__":
    unittest.main()

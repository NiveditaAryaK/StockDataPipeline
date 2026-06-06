from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from main import DB_PATH
from ml_dataset import FEATURE_COLUMNS, build_dataset_for_ticker


@dataclass(frozen=True)
class MLTrainingResult:
    ticker: str
    model_name: str
    train_rows: int
    test_rows: int
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp
    accuracy: float
    precision: float
    recall: float
    f1: float
    auc_roc: float | None
    latest_probability_up: float
    latest_signal: str
    predictions: pd.DataFrame
    feature_importance: pd.DataFrame
    pipeline: Pipeline


def chronological_split(dataset: pd.DataFrame, test_size: float = 0.2) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not 0 < test_size < 1:
        raise ValueError("test_size must be between 0 and 1.")

    ordered = dataset.sort_values("date").reset_index(drop=True)
    split_index = int(len(ordered) * (1 - test_size))
    if split_index <= 0 or split_index >= len(ordered):
        raise ValueError("Not enough rows to create both train and test datasets.")

    return ordered.iloc[:split_index].copy(), ordered.iloc[split_index:].copy()


def probability_to_signal(
    probability_up: float,
    buy_threshold: float = 0.55,
    sell_threshold: float = 0.45,
) -> str:
    if not sell_threshold < buy_threshold:
        raise ValueError("sell_threshold must be lower than buy_threshold.")
    if probability_up >= buy_threshold:
        return "BUY"
    if probability_up <= sell_threshold:
        return "SELL"
    return "HOLD"


def train_logistic_regression(
    dataset: pd.DataFrame,
    ticker: str,
    feature_columns: list[str] | None = None,
    test_size: float = 0.2,
    buy_threshold: float = 0.55,
    sell_threshold: float = 0.45,
) -> MLTrainingResult:
    features = feature_columns or FEATURE_COLUMNS
    missing_columns = [column for column in features + ["date", "target_up"] if column not in dataset.columns]
    if missing_columns:
        raise ValueError(f"Dataset is missing required columns: {', '.join(missing_columns)}")

    train_df, test_df = chronological_split(dataset, test_size)
    if train_df["target_up"].nunique() < 2:
        raise ValueError("Training data needs both up and down examples.")

    pipeline = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(max_iter=1000)),
        ]
    )

    x_train = train_df[features]
    y_train = train_df["target_up"]
    x_test = test_df[features]
    y_test = test_df["target_up"]

    pipeline.fit(x_train, y_train)
    predicted = pipeline.predict(x_test)
    probabilities_up = pipeline.predict_proba(x_test)[:, 1]

    auc_roc = None
    if y_test.nunique() == 2:
        auc_roc = float(roc_auc_score(y_test, probabilities_up))

    latest_probability_up = float(pipeline.predict_proba(dataset.sort_values("date")[features].tail(1))[:, 1][0])
    latest_signal = probability_to_signal(latest_probability_up, buy_threshold, sell_threshold)

    predictions = test_df[["ticker", "date", "close", "tomorrow_close", "target_up"]].copy()
    predictions["probability_up"] = probabilities_up
    predictions["predicted_up"] = predicted
    predictions["signal"] = predictions["probability_up"].apply(
        lambda probability: probability_to_signal(float(probability), buy_threshold, sell_threshold)
    )

    model = pipeline.named_steps["model"]
    coefficients = model.coef_[0]
    feature_importance = pd.DataFrame(
        {
            "feature": features,
            "coefficient": coefficients,
            "importance": abs(coefficients),
        }
    ).sort_values("importance", ascending=False, ignore_index=True)

    return MLTrainingResult(
        ticker=ticker.upper(),
        model_name="Logistic Regression",
        train_rows=len(train_df),
        test_rows=len(test_df),
        train_start=pd.to_datetime(train_df["date"].iloc[0]),
        train_end=pd.to_datetime(train_df["date"].iloc[-1]),
        test_start=pd.to_datetime(test_df["date"].iloc[0]),
        test_end=pd.to_datetime(test_df["date"].iloc[-1]),
        accuracy=float(accuracy_score(y_test, predicted)),
        precision=float(precision_score(y_test, predicted, zero_division=0)),
        recall=float(recall_score(y_test, predicted, zero_division=0)),
        f1=float(f1_score(y_test, predicted, zero_division=0)),
        auc_roc=auc_roc,
        latest_probability_up=latest_probability_up,
        latest_signal=latest_signal,
        predictions=predictions,
        feature_importance=feature_importance,
        pipeline=pipeline,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train ML models for next-day stock direction prediction.")
    parser.add_argument("ticker", help="Ticker symbol, for example: AAPL")
    parser.add_argument("--db", type=Path, default=DB_PATH, help="SQLite market data path")
    parser.add_argument("--test-size", type=float, default=0.2, help="Future holdout share, for example 0.2")
    parser.add_argument("--buy-threshold", type=float, default=0.55, help="Probability needed for a BUY signal")
    parser.add_argument("--sell-threshold", type=float, default=0.45, help="Probability needed for a SELL signal")
    return parser


if __name__ == "__main__":
    args = build_parser().parse_args()
    ml_dataset = build_dataset_for_ticker(args.ticker, args.db)
    result = train_logistic_regression(
        ml_dataset,
        args.ticker,
        test_size=args.test_size,
        buy_threshold=args.buy_threshold,
        sell_threshold=args.sell_threshold,
    )

    print(f"{result.model_name} for {result.ticker}")
    print(f"Train: {result.train_start.date()} -> {result.train_end.date()} ({result.train_rows} rows)")
    print(f"Test:  {result.test_start.date()} -> {result.test_end.date()} ({result.test_rows} rows)")
    print(f"Accuracy:  {result.accuracy:.2%}")
    print(f"Precision: {result.precision:.2%}")
    print(f"Recall:    {result.recall:.2%}")
    print(f"F1 Score:  {result.f1:.2%}")
    if result.auc_roc is not None:
        print(f"AUC ROC:   {result.auc_roc:.3f}")
    else:
        print("AUC ROC:   unavailable")
    print(f"Latest Probability Up: {result.latest_probability_up:.2%}")
    print(f"Latest Signal: {result.latest_signal}")
    print("\nTop feature importance:")
    print(result.feature_importance.head(10).to_string(index=False))

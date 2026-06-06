from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from main import DB_PATH
from ml_dataset import FEATURE_COLUMNS, build_dataset_for_ticker

TRADING_DAYS_PER_YEAR = 252


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
    actual_up_rate: float
    predicted_up_rate: float
    probability_min: float
    probability_25pct: float
    probability_avg: float
    probability_75pct: float
    probability_max: float
    buy_signal_rate: float
    hold_signal_rate: float
    sell_signal_rate: float
    latest_probability_up: float
    latest_signal: str
    predictions: pd.DataFrame
    feature_importance: pd.DataFrame
    threshold_sweep: pd.DataFrame
    threshold_backtest: pd.DataFrame
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
    pipeline = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(max_iter=1000)),
        ]
    )
    return train_classifier(
        dataset=dataset,
        ticker=ticker,
        model_name="Logistic Regression",
        pipeline=pipeline,
        feature_columns=feature_columns,
        test_size=test_size,
        buy_threshold=buy_threshold,
        sell_threshold=sell_threshold,
    )


def train_random_forest(
    dataset: pd.DataFrame,
    ticker: str,
    feature_columns: list[str] | None = None,
    test_size: float = 0.2,
    buy_threshold: float = 0.55,
    sell_threshold: float = 0.45,
) -> MLTrainingResult:
    pipeline = Pipeline(
        steps=[
            (
                "model",
                RandomForestClassifier(
                    n_estimators=300,
                    min_samples_leaf=5,
                    random_state=42,
                    n_jobs=-1,
                ),
            ),
        ]
    )
    return train_classifier(
        dataset=dataset,
        ticker=ticker,
        model_name="Random Forest",
        pipeline=pipeline,
        feature_columns=feature_columns,
        test_size=test_size,
        buy_threshold=buy_threshold,
        sell_threshold=sell_threshold,
    )


def train_xgboost(
    dataset: pd.DataFrame,
    ticker: str,
    feature_columns: list[str] | None = None,
    test_size: float = 0.2,
    buy_threshold: float = 0.55,
    sell_threshold: float = 0.45,
) -> MLTrainingResult:
    pipeline = Pipeline(
        steps=[
            (
                "model",
                XGBClassifier(
                    n_estimators=200,
                    max_depth=3,
                    learning_rate=0.03,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    eval_metric="logloss",
                    random_state=42,
                    n_jobs=-1,
                ),
            ),
        ]
    )
    return train_classifier(
        dataset=dataset,
        ticker=ticker,
        model_name="XGBoost",
        pipeline=pipeline,
        feature_columns=feature_columns,
        test_size=test_size,
        buy_threshold=buy_threshold,
        sell_threshold=sell_threshold,
    )


def train_classifier(
    dataset: pd.DataFrame,
    ticker: str,
    model_name: str,
    pipeline: Pipeline,
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
    signal_rates = predictions["signal"].value_counts(normalize=True)

    feature_importance = build_feature_importance(pipeline, features)
    probability_summary = pd.Series(probabilities_up).quantile([0, 0.25, 0.75, 1])
    threshold_sweep = build_threshold_sweep(probabilities_up, y_test)
    threshold_backtest = build_threshold_backtest(test_df, probabilities_up)

    return MLTrainingResult(
        ticker=ticker.upper(),
        model_name=model_name,
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
        actual_up_rate=float(y_test.mean()),
        predicted_up_rate=float(predicted.mean()),
        probability_min=float(probability_summary.loc[0]),
        probability_25pct=float(probability_summary.loc[0.25]),
        probability_avg=float(probabilities_up.mean()),
        probability_75pct=float(probability_summary.loc[0.75]),
        probability_max=float(probability_summary.loc[1]),
        buy_signal_rate=float(signal_rates.get("BUY", 0)),
        hold_signal_rate=float(signal_rates.get("HOLD", 0)),
        sell_signal_rate=float(signal_rates.get("SELL", 0)),
        latest_probability_up=latest_probability_up,
        latest_signal=latest_signal,
        predictions=predictions,
        feature_importance=feature_importance,
        threshold_sweep=threshold_sweep,
        threshold_backtest=threshold_backtest,
        pipeline=pipeline,
    )


def build_feature_importance(pipeline: Pipeline, features: list[str]) -> pd.DataFrame:
    model = pipeline.named_steps["model"]
    if hasattr(model, "coef_"):
        coefficients = model.coef_[0]
        return pd.DataFrame(
            {
                "feature": features,
                "coefficient": coefficients,
                "importance": abs(coefficients),
            }
        ).sort_values("importance", ascending=False, ignore_index=True)

    if hasattr(model, "feature_importances_"):
        return pd.DataFrame(
            {
                "feature": features,
                "importance": model.feature_importances_,
            }
        ).sort_values("importance", ascending=False, ignore_index=True)

    return pd.DataFrame({"feature": features, "importance": [0.0] * len(features)})


def build_threshold_sweep(
    probabilities_up,
    actual_up: pd.Series,
    thresholds: tuple[float, ...] = (0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60),
) -> pd.DataFrame:
    actual = actual_up.reset_index(drop=True)
    probabilities = pd.Series(probabilities_up)
    rows = []
    for threshold in thresholds:
        buy_mask = probabilities >= threshold
        sell_mask = probabilities <= threshold
        buy_count = int(buy_mask.sum())
        sell_count = int(sell_mask.sum())
        rows.append(
            {
                "threshold": threshold,
                "buy_count": buy_count,
                "buy_rate": buy_count / len(probabilities),
                "buy_win_rate": float(actual[buy_mask].mean()) if buy_count else None,
                "sell_count": sell_count,
                "sell_rate": sell_count / len(probabilities),
                "sell_win_rate": float((1 - actual[sell_mask]).mean()) if sell_count else None,
            }
        )
    return pd.DataFrame(rows)


def build_threshold_backtest(
    test_df: pd.DataFrame,
    probabilities_up,
    thresholds: tuple[float, ...] = (0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60),
) -> pd.DataFrame:
    probabilities = pd.Series(probabilities_up).reset_index(drop=True)
    next_day_returns = test_df["tomorrow_return"].reset_index(drop=True)
    buy_hold_return = float((1 + next_day_returns).prod() - 1)
    rows = []

    for threshold in thresholds:
        buy_mask = probabilities >= threshold
        strategy_returns = next_day_returns.where(buy_mask, 0.0)
        trade_returns = next_day_returns[buy_mask]
        equity_curve = (1 + strategy_returns).cumprod()
        drawdown = equity_curve / equity_curve.cummax() - 1

        if strategy_returns.std() == 0:
            sharpe_ratio = 0.0
        else:
            sharpe_ratio = float((strategy_returns.mean() / strategy_returns.std()) * (TRADING_DAYS_PER_YEAR**0.5))

        wins = trade_returns[trade_returns > 0]
        losses = trade_returns[trade_returns <= 0]
        total_return = float(equity_curve.iloc[-1] - 1)
        rows.append(
            {
                "threshold": threshold,
                "trades": int(buy_mask.sum()),
                "win_rate": float((trade_returns > 0).mean()) if len(trade_returns) else None,
                "avg_win": float(wins.mean()) if len(wins) else None,
                "avg_loss": float(losses.mean()) if len(losses) else None,
                "total_return": total_return,
                "buy_hold_return": buy_hold_return,
                "alpha": total_return - buy_hold_return,
                "max_drawdown": float(drawdown.min()),
                "sharpe_ratio": sharpe_ratio,
            }
        )

    return pd.DataFrame(rows)


MODEL_TRAINERS = {
    "logistic": train_logistic_regression,
    "random_forest": train_random_forest,
    "xgboost": train_xgboost,
}


def compare_model_across_tickers(
    tickers: list[str],
    model_key: str = "xgboost",
    threshold: float = 0.25,
    test_size: float = 0.2,
    db_path: Path = DB_PATH,
) -> tuple[pd.DataFrame, dict[str, float]]:
    if model_key not in MODEL_TRAINERS:
        raise ValueError(f"Unsupported model '{model_key}'. Choose from: {', '.join(sorted(MODEL_TRAINERS))}")

    rows = []
    for ticker in tickers:
        dataset = build_dataset_for_ticker(ticker, db_path)
        result = MODEL_TRAINERS[model_key](dataset, ticker, test_size=test_size)
        threshold_rows = result.threshold_backtest[result.threshold_backtest["threshold"] == threshold]
        if threshold_rows.empty:
            raise ValueError(f"Threshold {threshold:.2f} is not available in the threshold backtest.")

        threshold_result = threshold_rows.iloc[0]
        rows.append(
            {
                "ticker": ticker.upper(),
                "model": result.model_name,
                "threshold": threshold,
                "auc_roc": result.auc_roc,
                "trades": int(threshold_result["trades"]),
                "win_rate": threshold_result["win_rate"],
                "total_return": float(threshold_result["total_return"]),
                "buy_hold_return": float(threshold_result["buy_hold_return"]),
                "alpha": float(threshold_result["alpha"]),
                "max_drawdown": float(threshold_result["max_drawdown"]),
                "sharpe_ratio": float(threshold_result["sharpe_ratio"]),
            }
        )

    comparison = pd.DataFrame(rows).sort_values("alpha", ascending=False, ignore_index=True)
    summary = {
        "mean_alpha": float(comparison["alpha"].mean()),
        "median_alpha": float(comparison["alpha"].median()),
        "positive_alpha_rate": float((comparison["alpha"] > 0).mean()),
    }
    return comparison, summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train ML models for next-day stock direction prediction.")
    parser.add_argument("ticker", nargs="?", default="AAPL", help="Ticker symbol, for example: AAPL")
    parser.add_argument(
        "--model",
        choices=sorted(MODEL_TRAINERS),
        default="logistic",
        help="ML model to train",
    )
    parser.add_argument("--db", type=Path, default=DB_PATH, help="SQLite market data path")
    parser.add_argument("--test-size", type=float, default=0.2, help="Future holdout share, for example 0.2")
    parser.add_argument("--buy-threshold", type=float, default=0.55, help="Probability needed for a BUY signal")
    parser.add_argument("--sell-threshold", type=float, default=0.45, help="Probability needed for a SELL signal")
    parser.add_argument(
        "--compare",
        nargs="+",
        help="Compare the selected model across multiple tickers instead of printing one detailed ticker report",
    )
    parser.add_argument("--threshold", type=float, default=0.25, help="Threshold used for multi-ticker comparison")
    return parser


if __name__ == "__main__":
    args = build_parser().parse_args()
    if args.compare:
        comparison, summary = compare_model_across_tickers(
            args.compare,
            model_key=args.model,
            threshold=args.threshold,
            test_size=args.test_size,
            db_path=args.db,
        )
        print(f"{args.model} comparison at threshold {args.threshold:.0%}")
        print(
            comparison.to_string(
                index=False,
                formatters={
                    "threshold": "{:.0%}".format,
                    "auc_roc": lambda value: "n/a" if pd.isna(value) else f"{value:.3f}",
                    "win_rate": lambda value: "n/a" if pd.isna(value) else f"{value:.2%}",
                    "total_return": "{:.2%}".format,
                    "buy_hold_return": "{:.2%}".format,
                    "alpha": "{:.2%}".format,
                    "max_drawdown": "{:.2%}".format,
                    "sharpe_ratio": "{:.2f}".format,
                },
            )
        )
        print("\nSummary:")
        print(f"Mean Alpha: {summary['mean_alpha']:.2%}")
        print(f"Median Alpha: {summary['median_alpha']:.2%}")
        print(f"Positive Alpha Rate: {summary['positive_alpha_rate']:.2%}")
        raise SystemExit(0)

    ml_dataset = build_dataset_for_ticker(args.ticker, args.db)
    result = MODEL_TRAINERS[args.model](
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
    print(f"Actual Up Rate:    {result.actual_up_rate:.2%}")
    print(f"Predicted Up Rate: {result.predicted_up_rate:.2%}")
    print("\nProbability Up Distribution:")
    print(f"Min:     {result.probability_min:.2%}")
    print(f"25%:     {result.probability_25pct:.2%}")
    print(f"Average: {result.probability_avg:.2%}")
    print(f"75%:     {result.probability_75pct:.2%}")
    print(f"Max:     {result.probability_max:.2%}")
    print("\nThreshold Signal Mix:")
    print(f"BUY >= threshold:  {result.buy_signal_rate:.2%}")
    print(f"HOLD:              {result.hold_signal_rate:.2%}")
    print(f"SELL <= threshold: {result.sell_signal_rate:.2%}")
    print(f"Latest Probability Up: {result.latest_probability_up:.2%}")
    print(f"Latest Signal: {result.latest_signal}")
    print("\nThreshold Sweep:")
    print(
        result.threshold_sweep.to_string(
            index=False,
            formatters={
                "threshold": "{:.0%}".format,
                "buy_rate": "{:.2%}".format,
                "buy_win_rate": lambda value: "n/a" if pd.isna(value) else f"{value:.2%}",
                "sell_rate": "{:.2%}".format,
                "sell_win_rate": lambda value: "n/a" if pd.isna(value) else f"{value:.2%}",
            },
        )
    )
    print("\nThreshold Backtest:")
    print(
        result.threshold_backtest.to_string(
            index=False,
            formatters={
                "threshold": "{:.0%}".format,
                "win_rate": lambda value: "n/a" if pd.isna(value) else f"{value:.2%}",
                "avg_win": lambda value: "n/a" if pd.isna(value) else f"{value:.2%}",
                "avg_loss": lambda value: "n/a" if pd.isna(value) else f"{value:.2%}",
                "total_return": "{:.2%}".format,
                "buy_hold_return": "{:.2%}".format,
                "alpha": "{:.2%}".format,
                "max_drawdown": "{:.2%}".format,
                "sharpe_ratio": "{:.2f}".format,
            },
        )
    )
    print("\nTop feature importance:")
    print(result.feature_importance.to_string(index=False))

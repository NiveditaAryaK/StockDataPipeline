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
MODEL_NAMES = {
    "logistic": "Logistic Regression",
    "random_forest": "Random Forest",
    "xgboost": "XGBoost",
}


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
    return train_classifier(
        dataset=dataset,
        ticker=ticker,
        model_name=MODEL_NAMES["logistic"],
        pipeline=build_model_pipeline("logistic"),
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
    return train_classifier(
        dataset=dataset,
        ticker=ticker,
        model_name=MODEL_NAMES["random_forest"],
        pipeline=build_model_pipeline("random_forest"),
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
    return train_classifier(
        dataset=dataset,
        ticker=ticker,
        model_name=MODEL_NAMES["xgboost"],
        pipeline=build_model_pipeline("xgboost"),
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
    return train_classifier_on_frames(
        train_df=train_df,
        test_df=test_df,
        ticker=ticker,
        model_name=model_name,
        pipeline=pipeline,
        feature_columns=features,
        buy_threshold=buy_threshold,
        sell_threshold=sell_threshold,
    )


def train_classifier_on_frames(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    ticker: str,
    model_name: str,
    pipeline: Pipeline,
    feature_columns: list[str],
    buy_threshold: float = 0.55,
    sell_threshold: float = 0.45,
) -> MLTrainingResult:
    if train_df["target_up"].nunique() < 2:
        raise ValueError("Training data needs both up and down examples.")

    x_train = train_df[feature_columns]
    y_train = train_df["target_up"]
    x_test = test_df[feature_columns]
    y_test = test_df["target_up"]

    pipeline.fit(x_train, y_train)
    predicted = pipeline.predict(x_test)
    probabilities_up = pipeline.predict_proba(x_test)[:, 1]

    auc_roc = None
    if y_test.nunique() == 2:
        auc_roc = float(roc_auc_score(y_test, probabilities_up))

    latest_probability_up = float(pipeline.predict_proba(test_df.sort_values("date")[feature_columns].tail(1))[:, 1][0])
    latest_signal = probability_to_signal(latest_probability_up, buy_threshold, sell_threshold)

    predictions = test_df[["ticker", "date", "close", "tomorrow_close", "target_up"]].copy()
    predictions["probability_up"] = probabilities_up
    predictions["predicted_up"] = predicted
    predictions["signal"] = predictions["probability_up"].apply(
        lambda probability: probability_to_signal(float(probability), buy_threshold, sell_threshold)
    )
    signal_rates = predictions["signal"].value_counts(normalize=True)

    feature_importance = build_feature_importance(pipeline, feature_columns)
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


def build_model_pipeline(model_key: str) -> Pipeline:
    if model_key == "logistic":
        return Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                ("model", LogisticRegression(max_iter=1000)),
            ]
        )

    if model_key == "random_forest":
        return Pipeline(
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

    if model_key == "xgboost":
        return Pipeline(
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

    raise ValueError(f"Unsupported model '{model_key}'. Choose from: {', '.join(sorted(MODEL_NAMES))}")


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


def walk_forward_validation(
    dataset: pd.DataFrame,
    ticker: str,
    model_key: str = "xgboost",
    threshold: float = 0.25,
    train_years: int = 5,
    test_years: int = 1,
    step_years: int = 1,
    feature_columns: list[str] | None = None,
) -> tuple[pd.DataFrame, dict[str, float]]:
    if model_key not in MODEL_NAMES:
        raise ValueError(f"Unsupported model '{model_key}'. Choose from: {', '.join(sorted(MODEL_NAMES))}")
    if train_years < 1 or test_years < 1 or step_years < 1:
        raise ValueError("train_years, test_years, and step_years must be positive integers.")

    features = feature_columns or FEATURE_COLUMNS
    data = dataset.sort_values("date").reset_index(drop=True)
    data["year"] = pd.to_datetime(data["date"]).dt.year
    first_year = int(data["year"].min())
    last_year = int(data["year"].max())
    rows = []

    for start_year in range(first_year, last_year - train_years - test_years + 2, step_years):
        train_start_year = start_year
        train_end_year = start_year + train_years - 1
        test_start_year = train_end_year + 1
        test_end_year = test_start_year + test_years - 1

        train_df = data[(data["year"] >= train_start_year) & (data["year"] <= train_end_year)].drop(columns=["year"])
        test_df = data[(data["year"] >= test_start_year) & (data["year"] <= test_end_year)].drop(columns=["year"])
        if train_df.empty or test_df.empty or train_df["target_up"].nunique() < 2:
            continue

        result = train_classifier_on_frames(
            train_df=train_df,
            test_df=test_df,
            ticker=ticker,
            model_name=MODEL_NAMES[model_key],
            pipeline=build_model_pipeline(model_key),
            feature_columns=features,
        )
        threshold_rows = result.threshold_backtest[
            (result.threshold_backtest["threshold"] - threshold).abs() < 0.000001
        ]
        if threshold_rows.empty:
            raise ValueError(f"Threshold {threshold:.2f} is not available in the threshold backtest.")

        threshold_result = threshold_rows.iloc[0]
        rows.append(
            {
                "ticker": ticker.upper(),
                "model": result.model_name,
                "threshold": threshold,
                "train_window": f"{train_start_year}-{train_end_year}",
                "test_window": f"{test_start_year}-{test_end_year}",
                "train_rows": result.train_rows,
                "test_rows": result.test_rows,
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

    if not rows:
        raise ValueError("No walk-forward windows could be created from this dataset.")

    results = pd.DataFrame(rows)
    compounded_strategy_return = float((1 + results["total_return"]).prod() - 1)
    compounded_buy_hold_return = float((1 + results["buy_hold_return"]).prod() - 1)
    summary = {
        "windows": float(len(results)),
        "mean_alpha": float(results["alpha"].mean()),
        "median_alpha": float(results["alpha"].median()),
        "positive_alpha_rate": float((results["alpha"] > 0).mean()),
        "compounded_strategy_return": compounded_strategy_return,
        "compounded_buy_hold_return": compounded_buy_hold_return,
        "compounded_alpha": compounded_strategy_return - compounded_buy_hold_return,
        "mean_auc_roc": float(results["auc_roc"].dropna().mean()) if results["auc_roc"].notna().any() else 0.0,
    }
    return results, summary


def compare_walk_forward_across_tickers(
    tickers: list[str],
    model_key: str = "xgboost",
    threshold: float = 0.25,
    train_years: int = 5,
    test_years: int = 1,
    step_years: int = 1,
    db_path: Path = DB_PATH,
) -> tuple[pd.DataFrame, dict[str, float]]:
    rows = []
    for ticker in tickers:
        dataset = build_dataset_for_ticker(ticker, db_path)
        _, summary = walk_forward_validation(
            dataset,
            ticker,
            model_key=model_key,
            threshold=threshold,
            train_years=train_years,
            test_years=test_years,
            step_years=step_years,
        )
        rows.append(
            {
                "ticker": ticker.upper(),
                "model": MODEL_NAMES[model_key],
                "threshold": threshold,
                "windows": int(summary["windows"]),
                "mean_alpha": summary["mean_alpha"],
                "median_alpha": summary["median_alpha"],
                "positive_alpha_rate": summary["positive_alpha_rate"],
                "compounded_strategy_return": summary["compounded_strategy_return"],
                "compounded_buy_hold_return": summary["compounded_buy_hold_return"],
                "compounded_alpha": summary["compounded_alpha"],
                "mean_auc_roc": summary["mean_auc_roc"],
            }
        )

    comparison = pd.DataFrame(rows).sort_values("compounded_alpha", ascending=False, ignore_index=True)
    summary = {
        "mean_alpha": float(comparison["mean_alpha"].mean()),
        "median_alpha": float(comparison["mean_alpha"].median()),
        "mean_compounded_alpha": float(comparison["compounded_alpha"].mean()),
        "median_compounded_alpha": float(comparison["compounded_alpha"].median()),
        "positive_ticker_rate": float((comparison["compounded_alpha"] > 0).mean()),
        "mean_positive_window_rate": float(comparison["positive_alpha_rate"].mean()),
        "mean_auc_roc": float(comparison["mean_auc_roc"].mean()),
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
    parser.add_argument("--walk-forward", action="store_true", help="Run rolling calendar-year walk-forward validation")
    parser.add_argument("--train-years", type=int, default=5, help="Training years per walk-forward window")
    parser.add_argument("--test-years", type=int, default=1, help="Testing years per walk-forward window")
    parser.add_argument("--step-years", type=int, default=1, help="Years to move forward after each window")
    return parser


if __name__ == "__main__":
    args = build_parser().parse_args()
    if args.compare:
        if args.walk_forward:
            comparison, summary = compare_walk_forward_across_tickers(
                args.compare,
                model_key=args.model,
                threshold=args.threshold,
                train_years=args.train_years,
                test_years=args.test_years,
                step_years=args.step_years,
                db_path=args.db,
            )
            print(f"{args.model} walk-forward comparison at threshold {args.threshold:.0%}")
            print(
                comparison.to_string(
                    index=False,
                    formatters={
                        "threshold": "{:.0%}".format,
                        "mean_alpha": "{:.2%}".format,
                        "median_alpha": "{:.2%}".format,
                        "positive_alpha_rate": "{:.2%}".format,
                        "compounded_strategy_return": "{:.2%}".format,
                        "compounded_buy_hold_return": "{:.2%}".format,
                        "compounded_alpha": "{:.2%}".format,
                        "mean_auc_roc": "{:.3f}".format,
                    },
                )
            )
            print("\nSummary:")
            print(f"Mean Alpha: {summary['mean_alpha']:.2%}")
            print(f"Median Alpha: {summary['median_alpha']:.2%}")
            print(f"Mean Compounded Alpha: {summary['mean_compounded_alpha']:.2%}")
            print(f"Median Compounded Alpha: {summary['median_compounded_alpha']:.2%}")
            print(f"Positive Ticker Rate: {summary['positive_ticker_rate']:.2%}")
            print(f"Mean Positive Window Rate: {summary['mean_positive_window_rate']:.2%}")
            print(f"Mean AUC ROC: {summary['mean_auc_roc']:.3f}")
            raise SystemExit(0)

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
    if args.walk_forward:
        walk_results, walk_summary = walk_forward_validation(
            ml_dataset,
            args.ticker,
            model_key=args.model,
            threshold=args.threshold,
            train_years=args.train_years,
            test_years=args.test_years,
            step_years=args.step_years,
        )
        print(
            f"{args.model} walk-forward validation for {args.ticker.upper()} "
            f"at threshold {args.threshold:.0%}"
        )
        print(
            walk_results.to_string(
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
        print(f"Windows: {walk_summary['windows']:.0f}")
        print(f"Mean Alpha: {walk_summary['mean_alpha']:.2%}")
        print(f"Median Alpha: {walk_summary['median_alpha']:.2%}")
        print(f"Positive Alpha Rate: {walk_summary['positive_alpha_rate']:.2%}")
        print(f"Compounded Strategy Return: {walk_summary['compounded_strategy_return']:.2%}")
        print(f"Compounded Buy & Hold Return: {walk_summary['compounded_buy_hold_return']:.2%}")
        print(f"Compounded Alpha: {walk_summary['compounded_alpha']:.2%}")
        print(f"Mean AUC ROC: {walk_summary['mean_auc_roc']:.3f}")
        raise SystemExit(0)

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

from __future__ import annotations

import argparse
import json
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from backtest import BacktestResult, run_strategy_backtest
from main import DB_PATH, load_prices
from strategies import BollingerBandStrategy, MACrossoverStrategy, MomentumStrategy, RSIStrategy, Strategy

AGENT_DB_PATH = Path("data/agent_research.sqlite")


@dataclass(frozen=True)
class StrategyCandidate:
    strategy_id: str
    strategy_name: str
    parameters: dict[str, float | int | str]
    strategy: Strategy
    rationale: str = "seed candidate"


@dataclass(frozen=True)
class AgentRunResult:
    experiments: pd.DataFrame
    rankings: pd.DataFrame


class StrategyGenerator:
    def generate(self) -> list[StrategyCandidate]:
        return self.seed_candidates()

    def seed_candidates(self) -> list[StrategyCandidate]:
        return [
            self.rsi_candidate(30, 70, "broad family scan"),
            self.ma_candidate(5, 20, "broad family scan"),
            self.momentum_candidate(20, "broad family scan"),
            self.bollinger_candidate(20, 2, "broad family scan"),
        ]

    def variations(self, best: StrategyCandidate, existing_ids: set[str]) -> list[StrategyCandidate]:
        parameters = best.parameters
        candidates: list[StrategyCandidate] = []
        if "lookback_days" in parameters:
            lookback = int(parameters["lookback_days"])
            for candidate_lookback in sorted({max(3, lookback + delta) for delta in [-10, -5, -2, 2, 5, 10]}):
                candidates.append(self.momentum_candidate(candidate_lookback, f"explore around {best.strategy_name}"))
        elif "buy_rsi" in parameters:
            buy_rsi = int(parameters["buy_rsi"])
            sell_rsi = int(parameters["sell_rsi"])
            for candidate_buy, candidate_sell in {
                (max(5, buy_rsi - 5), min(95, sell_rsi + 5)),
                (min(50, buy_rsi + 5), max(50, sell_rsi - 5)),
                (min(50, buy_rsi + 10), max(50, sell_rsi - 10)),
            }:
                if candidate_buy < candidate_sell:
                    candidates.append(self.rsi_candidate(candidate_buy, candidate_sell, f"explore around {best.strategy_name}"))
        elif "fast_ma" in parameters:
            fast_ma = int(parameters["fast_ma"])
            slow_ma = int(parameters["slow_ma"])
            for candidate_fast, candidate_slow in {
                (max(2, fast_ma - 2), max(fast_ma + 2, slow_ma - 5)),
                (fast_ma + 1, slow_ma + 5),
                (fast_ma + 3, slow_ma + 10),
                (max(2, fast_ma - 1), slow_ma + 10),
            }:
                if candidate_fast < candidate_slow:
                    candidates.append(self.ma_candidate(candidate_fast, candidate_slow, f"explore around {best.strategy_name}"))
        elif "window" in parameters:
            window = int(parameters["window"])
            standard_deviations = float(parameters["standard_deviations"])
            for candidate_window, candidate_std in {
                (max(10, window - 10), standard_deviations),
                (window + 10, standard_deviations),
                (window, max(1.0, standard_deviations - 0.5)),
                (window, standard_deviations + 0.5),
            }:
                candidates.append(self.bollinger_candidate(candidate_window, candidate_std, f"explore around {best.strategy_name}"))

        return [candidate for candidate in candidates if candidate.strategy_id not in existing_ids]

    def rsi_candidate(self, buy_rsi: int, sell_rsi: int, rationale: str) -> StrategyCandidate:
        return StrategyCandidate(
            strategy_id=f"rsi_{buy_rsi}_{sell_rsi}",
            strategy_name=f"RSI {buy_rsi}/{sell_rsi}",
            parameters={"buy_rsi": buy_rsi, "sell_rsi": sell_rsi},
            strategy=RSIStrategy(buy_rsi, sell_rsi),
            rationale=rationale,
        )

    def ma_candidate(self, fast_ma: int, slow_ma: int, rationale: str) -> StrategyCandidate:
        return StrategyCandidate(
            strategy_id=f"ma_{fast_ma}_{slow_ma}",
            strategy_name=f"MA {fast_ma}/{slow_ma} Crossover",
            parameters={"fast_ma": fast_ma, "slow_ma": slow_ma},
            strategy=MACrossoverStrategy(fast_ma, slow_ma),
            rationale=rationale,
        )

    def momentum_candidate(self, lookback_days: int, rationale: str) -> StrategyCandidate:
        return StrategyCandidate(
            strategy_id=f"momentum_{lookback_days}",
            strategy_name=f"Momentum {lookback_days}D",
            parameters={"lookback_days": lookback_days},
            strategy=MomentumStrategy(lookback_days),
            rationale=rationale,
        )

    def bollinger_candidate(self, window: int, standard_deviations: float, rationale: str) -> StrategyCandidate:
        return StrategyCandidate(
            strategy_id=f"bollinger_{window}_{str(standard_deviations).replace('.', '_')}",
            strategy_name=f"Bollinger {window}D/{standard_deviations:g}SD",
            parameters={"window": window, "standard_deviations": standard_deviations},
            strategy=BollingerBandStrategy(window, standard_deviations),
            rationale=rationale,
        )


class MetricsAnalyzer:
    def score(self, result: BacktestResult) -> float:
        trade_penalty = 0.25 if result.trade_count < 5 else 0.0
        drawdown_penalty = abs(result.max_drawdown) * 0.5
        return float(result.sharpe_ratio + result.alpha + result.total_return - drawdown_penalty - trade_penalty)

    def notes(self, result: BacktestResult) -> str:
        notes = []
        if result.trade_count < 5:
            notes.append("low trade count")
        if result.alpha < 0:
            notes.append("underperformed buy-and-hold")
        if result.max_drawdown < -0.3:
            notes.append("large drawdown")
        if result.sharpe_ratio > 1:
            notes.append("strong risk-adjusted return")
        return ", ".join(notes) if notes else "acceptable"


class ExperimentStore:
    def __init__(self, db_path: Path = AGENT_DB_PATH) -> None:
        self.db_path = db_path
        self.ensure_tables()

    def ensure_tables(self) -> None:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(self.db_path)
        try:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS experiments (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    strategy_id TEXT NOT NULL,
                    strategy_name TEXT NOT NULL,
                    parameters TEXT NOT NULL,
                    ticker TEXT NOT NULL,
                    period TEXT NOT NULL,
                    status TEXT NOT NULL,
                    created_at TEXT NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS experiment_results (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    experiment_id INTEGER NOT NULL,
                    return_pct REAL,
                    sharpe REAL,
                    drawdown REAL,
                    trade_count INTEGER,
                    alpha_pct REAL,
                    win_rate REAL,
                    score REAL,
                    notes TEXT,
                    FOREIGN KEY (experiment_id) REFERENCES experiments(id)
                )
                """
            )
            conn.commit()
        finally:
            conn.close()

    def create_experiment(self, candidate: StrategyCandidate, ticker: str, period: str, status: str) -> int:
        conn = sqlite3.connect(self.db_path)
        try:
            cursor = conn.execute(
                """
                INSERT INTO experiments (
                    strategy_id, strategy_name, parameters, ticker, period, status, created_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    candidate.strategy_id,
                    candidate.strategy_name,
                    json.dumps(candidate.parameters, sort_keys=True),
                    ticker.upper(),
                    period,
                    status,
                    datetime.now(timezone.utc).isoformat(),
                ),
            )
            conn.commit()
            return int(cursor.lastrowid)
        finally:
            conn.close()

    def save_result(
        self,
        experiment_id: int,
        result: BacktestResult,
        score: float,
        notes: str,
    ) -> None:
        conn = sqlite3.connect(self.db_path)
        try:
            conn.execute("UPDATE experiments SET status = ? WHERE id = ?", ("completed", experiment_id))
            conn.execute(
                """
                INSERT INTO experiment_results (
                    experiment_id, return_pct, sharpe, drawdown, trade_count,
                    alpha_pct, win_rate, score, notes
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    experiment_id,
                    result.total_return * 100,
                    result.sharpe_ratio,
                    result.max_drawdown * 100,
                    result.trade_count,
                    result.alpha * 100,
                    result.win_rate * 100,
                    score,
                    notes,
                ),
            )
            conn.commit()
        finally:
            conn.close()

    def mark_failed(self, experiment_id: int, notes: str) -> None:
        conn = sqlite3.connect(self.db_path)
        try:
            conn.execute("UPDATE experiments SET status = ? WHERE id = ?", ("failed", experiment_id))
            conn.execute(
                """
                INSERT INTO experiment_results (
                    experiment_id, return_pct, sharpe, drawdown, trade_count,
                    alpha_pct, win_rate, score, notes
                )
                VALUES (?, NULL, NULL, NULL, 0, NULL, NULL, NULL, ?)
                """,
                (experiment_id, notes),
            )
            conn.commit()
        finally:
            conn.close()

    def load_history(self, ticker: str | None = None, limit: int = 200) -> pd.DataFrame:
        query = """
            SELECT
                e.id AS experiment_id,
                e.strategy_id,
                e.strategy_name,
                e.parameters,
                e.ticker,
                e.period,
                e.status,
                e.created_at,
                r.return_pct,
                r.sharpe,
                r.drawdown,
                r.trade_count,
                r.alpha_pct,
                r.win_rate,
                r.score,
                r.notes
            FROM experiments e
            LEFT JOIN experiment_results r ON r.experiment_id = e.id
        """
        params: list[str | int] = []
        if ticker:
            query += " WHERE e.ticker = ?"
            params.append(ticker.upper())
        query += " ORDER BY e.id DESC LIMIT ?"
        params.append(limit)
        conn = sqlite3.connect(self.db_path)
        try:
            return pd.read_sql_query(query, conn, params=params)
        finally:
            conn.close()


class ResearchAgent:
    def __init__(
        self,
        generator: StrategyGenerator | None = None,
        analyzer: MetricsAnalyzer | None = None,
        store: ExperimentStore | None = None,
    ) -> None:
        self.generator = generator or StrategyGenerator()
        self.analyzer = analyzer or MetricsAnalyzer()
        self.store = store or ExperimentStore()

    def run(
        self,
        ticker: str,
        period_years: int = 5,
        iterations: int = 10,
        initial_cash: float = 10_000,
        market_db_path: Path = DB_PATH,
    ) -> AgentRunResult:
        prices = filter_years(load_prices(ticker, market_db_path), period_years)
        period = f"{period_years}y"
        rows = []
        tested_ids: set[str] = set()
        candidate_queue = self.generator.seed_candidates()

        for iteration in range(1, iterations + 1):
            if not candidate_queue:
                break
            candidate = candidate_queue.pop(0)
            if candidate.strategy_id in tested_ids:
                continue
            tested_ids.add(candidate.strategy_id)
            experiment_id = self.store.create_experiment(candidate, ticker, period, "running")
            row = {
                "iteration": iteration,
                "experiment_id": experiment_id,
                "strategy": candidate.strategy_name,
                "parameters": candidate.parameters,
                "rationale": candidate.rationale,
                "status": "running",
            }
            try:
                result = run_strategy_backtest(prices, ticker, candidate.strategy, initial_cash)
                score = self.analyzer.score(result)
                notes = self.analyzer.notes(result)
                self.store.save_result(experiment_id, result, score, notes)
                row.update(
                    {
                        "status": "completed",
                        "return_pct": result.total_return * 100,
                        "sharpe": result.sharpe_ratio,
                        "drawdown": result.max_drawdown * 100,
                        "trade_count": result.trade_count,
                        "alpha_pct": result.alpha * 100,
                        "win_rate": result.win_rate * 100,
                        "score": score,
                        "notes": notes,
                    }
                )
            except ValueError as error:
                notes = str(error)
                self.store.mark_failed(experiment_id, notes)
                row.update({"status": "failed", "notes": notes})
            rows.append(row)

            if any(queued.rationale == "broad family scan" for queued in candidate_queue):
                continue

            completed_rows = [completed for completed in rows if completed.get("status") == "completed"]
            if completed_rows:
                top_rows = sorted(completed_rows, key=lambda completed: completed["score"], reverse=True)[:2]
                queued_ids = {queued.strategy_id for queued in candidate_queue}
                existing_ids = tested_ids.union(queued_ids)
                candidates_by_id = {candidate.strategy_id: candidate for candidate in self.generator.seed_candidates()}
                candidates_by_id[candidate.strategy_id] = candidate
                for completed in rows:
                    strategy_id = self.strategy_id_from_row(completed)
                    if strategy_id not in candidates_by_id:
                        candidates_by_id[strategy_id] = self.candidate_from_row(completed)
                new_variations: list[StrategyCandidate] = []
                for top_row in top_rows:
                    top_candidate = candidates_by_id.get(self.strategy_id_from_row(top_row))
                    if top_candidate:
                        new_variations.extend(self.generator.variations(top_candidate, existing_ids))
                        existing_ids.update(candidate.strategy_id for candidate in new_variations)
                candidate_queue = new_variations + candidate_queue

        experiments = pd.DataFrame(rows)
        rankings = (
            experiments[experiments["status"] == "completed"]
            .sort_values(["score", "sharpe", "return_pct"], ascending=False)
            .reset_index(drop=True)
        )
        if not rankings.empty:
            rankings.insert(0, "rank", range(1, len(rankings) + 1))
        return AgentRunResult(experiments, rankings)

    def strategy_id_from_row(self, row: dict[str, object]) -> str:
        parameters = row.get("parameters", {})
        if isinstance(parameters, str):
            parameters = json.loads(parameters)
        if "lookback_days" in parameters:
            return f"momentum_{int(parameters['lookback_days'])}"
        if "buy_rsi" in parameters:
            return f"rsi_{int(parameters['buy_rsi'])}_{int(parameters['sell_rsi'])}"
        if "fast_ma" in parameters:
            return f"ma_{int(parameters['fast_ma'])}_{int(parameters['slow_ma'])}"
        if "window" in parameters:
            return f"bollinger_{int(parameters['window'])}_{str(parameters['standard_deviations']).replace('.', '_')}"
        return str(row.get("strategy", "unknown")).lower().replace(" ", "_")

    def candidate_from_row(self, row: dict[str, object]) -> StrategyCandidate:
        parameters = row.get("parameters", {})
        if isinstance(parameters, str):
            parameters = json.loads(parameters)
        strategy_name = str(row.get("strategy", ""))
        if "lookback_days" in parameters:
            return self.generator.momentum_candidate(int(parameters["lookback_days"]), f"explore around {strategy_name}")
        if "buy_rsi" in parameters:
            return self.generator.rsi_candidate(int(parameters["buy_rsi"]), int(parameters["sell_rsi"]), f"explore around {strategy_name}")
        if "fast_ma" in parameters:
            return self.generator.ma_candidate(int(parameters["fast_ma"]), int(parameters["slow_ma"]), f"explore around {strategy_name}")
        if "window" in parameters:
            return self.generator.bollinger_candidate(int(parameters["window"]), float(parameters["standard_deviations"]), f"explore around {strategy_name}")
        raise ValueError(f"Cannot rebuild candidate from row: {row}")


def filter_years(prices: pd.DataFrame, years: int) -> pd.DataFrame:
    data = prices.sort_values("date").copy()
    if data.empty:
        return data
    end_date = data["date"].max()
    start_date = end_date - pd.DateOffset(years=years)
    return data[data["date"] >= start_date].copy()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the automated quant research agent.")
    parser.add_argument("ticker", help="Ticker to research, for example: AAPL")
    parser.add_argument("--period-years", type=int, default=5, help="Lookback period in years")
    parser.add_argument("--iterations", type=int, default=10, help="Number of strategy candidates to test")
    parser.add_argument("--cash", type=float, default=10_000, help="Starting cash")
    parser.add_argument("--market-db", type=Path, default=DB_PATH, help="SQLite market data path")
    parser.add_argument("--agent-db", type=Path, default=AGENT_DB_PATH, help="SQLite agent database path")
    return parser


if __name__ == "__main__":
    args = build_parser().parse_args()
    agent = ResearchAgent(store=ExperimentStore(args.agent_db))
    run_result = agent.run(args.ticker, args.period_years, args.iterations, args.cash, args.market_db)
    print("Experiment Log")
    print(run_result.experiments.to_string(index=False))
    print("\nBest Strategies")
    print(run_result.rankings.to_string(index=False))

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from main import save_prices
from research_agent import ExperimentStore, ResearchAgent


class ResearchAgentTests(unittest.TestCase):
    def test_agent_runs_and_persists_experiments(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            market_db = Path(tmpdir) / "market.sqlite"
            agent_db = Path(tmpdir) / "agent.sqlite"
            prices = pd.DataFrame(
                {
                    "ticker": ["TEST"] * 80,
                    "date": pd.date_range("2024-01-01", periods=80, freq="D"),
                    "open": [100 + index for index in range(80)],
                    "high": [101 + index for index in range(80)],
                    "low": [99 + index for index in range(80)],
                    "close": [100 + index for index in range(80)],
                    "adj_close": [100 + index for index in range(80)],
                    "volume": [1000] * 80,
                    "daily_return": [0.01] * 80,
                    "ma_5": [100 + index for index in range(80)],
                    "ma_20": [99 + index for index in range(80)],
                    "rsi_14": [25 if index % 20 < 5 else 75 for index in range(80)],
                }
            )
            save_prices(prices, market_db)

            agent = ResearchAgent(store=ExperimentStore(agent_db))
            result = agent.run("TEST", period_years=1, iterations=6, market_db_path=market_db)
            history = agent.store.load_history("TEST")

            self.assertEqual(len(result.experiments), 6)
            self.assertFalse(result.rankings.empty)
            self.assertEqual(len(history), 6)
            self.assertTrue(result.experiments["rationale"].str.contains("explore around").any())


if __name__ == "__main__":
    unittest.main()

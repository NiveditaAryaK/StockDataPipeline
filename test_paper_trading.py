from __future__ import annotations

import unittest

import pandas as pd

from paper_trading import Portfolio, run_paper_trading_simulation
from strategies import RSIStrategy


class PaperTradingTests(unittest.TestCase):
    def test_portfolio_tracks_cash_positions_and_pnl(self) -> None:
        portfolio = Portfolio(1_000)
        portfolio.buy("TEST", 10, 50)
        portfolio.update_prices({"TEST": 55})

        self.assertEqual(portfolio.cash, 500)
        self.assertEqual(portfolio.positions_value, 550)
        self.assertEqual(portfolio.unrealized_pnl, 50)
        self.assertEqual(portfolio.value, 1_050)

        shares, realized = portfolio.sell_all("TEST", 45)

        self.assertEqual(shares, 10)
        self.assertEqual(realized, -50)
        self.assertEqual(portfolio.realized_pnl, -50)
        self.assertEqual(portfolio.unrealized_pnl, 0)
        self.assertEqual(portfolio.value, 950)

    def test_paper_trading_simulation_executes_orders(self) -> None:
        prices = pd.DataFrame(
            {
                "date": pd.date_range("2024-01-01", periods=6, freq="D"),
                "open": [100, 95, 94, 98, 105, 110],
                "close": [100, 94, 95, 100, 108, 111],
                "rsi_14": [50, 25, 20, 45, 75, 50],
            }
        )

        result = run_paper_trading_simulation({"TEST": prices}, RSIStrategy(), initial_cash=1_000)

        self.assertEqual(len(result.trades), 2)
        self.assertIn("portfolio_value", result.equity_curve.columns)
        self.assertGreater(result.equity_curve["portfolio_value"].iloc[-1], 0)


if __name__ == "__main__":
    unittest.main()

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import stock_universe
from capital_flow_analysis.global_macro_loader import FLOW_TICKER_LABELS


class MacroUniverseSeparationTests(unittest.TestCase):
    def test_backward_alias_points_to_rl_macro_universe(self):
        self.assertEqual(stock_universe.MACRO_TICKERS, stock_universe.MACRO_TICKERS_RL)
        self.assertEqual(stock_universe.MACRO_TICKERS_RL, ["^TWII", "^IXIC", "USDTWD=X"])

    def test_flow_macro_universe_is_separate(self):
        flow = set(stock_universe.MACRO_TICKERS_FLOW)
        self.assertIn("BTC-USD", flow)
        self.assertIn("NQ=F", flow)
        self.assertIn("DX-Y.NYB", flow)
        self.assertNotIn("BTC-USD", stock_universe.MACRO_TICKERS_RL)

    def test_flow_loader_labels_are_defined_for_flow_tickers(self):
        missing = [
            ticker
            for ticker in stock_universe.MACRO_TICKERS_FLOW
            if ticker not in FLOW_TICKER_LABELS
        ]
        self.assertEqual(missing, [])

    def test_rl_entrypoints_import_rl_macro_constant(self):
        root = Path(__file__).resolve().parents[1]
        for filename in ["train_portfolio.py", "walk_forward.py", "evaluate_portfolio.py"]:
            text = (root / filename).read_text(encoding="utf-8")
            self.assertIn("MACRO_TICKERS_RL", text)
            self.assertNotIn("from stock_universe import MACRO_TICKERS,", text)


if __name__ == "__main__":
    unittest.main()

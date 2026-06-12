"""Tests for P6 refactor — module extraction from cmoney_rpa.py.

Coverage
--------
- cmoney_client: get_accounts_config env-variable logic (no network)
- signal_validator: re-export facade exposes correct objects
- rebalance_planner: re-export facade exposes correct objects
- backward compat: ``from cmoney_rpa import X`` still works for all P6 symbols
- backward compat: ``cmoney_rpa.get_auto_aids`` and ``get_accounts_config`` still accessible
- trade_guard import chain still resolves correctly
"""

from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


# ---------------------------------------------------------------------------
# cmoney_client — standalone account config (no network required)
# ---------------------------------------------------------------------------

class CMoneyClientTests(unittest.TestCase):
    def test_get_accounts_config_returns_empty_without_cookie(self):
        """No CMONEY_COOKIE → empty list (load_dotenv mocked so .env is ignored)."""
        import rpa_pipeline.cmoney_client as cmoney_client

        with patch("rpa_pipeline.cmoney_client.load_dotenv", autospec=True):
            with patch.dict(os.environ, {}, clear=True):
                result = cmoney_client.get_accounts_config()
        self.assertEqual(result, [])

    def test_get_accounts_config_with_explicit_aid(self):
        """CMONEY_COOKIE + CMONEY_AID → one account entry, no network call."""
        import rpa_pipeline.cmoney_client as cmoney_client

        env = {"CMONEY_COOKIE": "fake_cookie_value", "CMONEY_AID": "99999"}
        with patch("rpa_pipeline.cmoney_client.load_dotenv", autospec=True):
            with patch.dict(os.environ, env, clear=True):
                result = cmoney_client.get_accounts_config()

        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["aid"], "99999")
        self.assertEqual(result[0]["cookie"], "fake_cookie_value")
        self.assertEqual(result[0]["name"], "Account_99999")

    def test_get_accounts_config_with_multiple_aids(self):
        """CMONEY_AID + CMONEY_AID_1 → two account entries."""
        import rpa_pipeline.cmoney_client as cmoney_client

        env = {
            "CMONEY_COOKIE": "fake_cookie",
            "CMONEY_AID": "10001",
            "CMONEY_AID_1": "10002",
        }
        with patch("rpa_pipeline.cmoney_client.load_dotenv", autospec=True):
            with patch.dict(os.environ, env, clear=True):
                result = cmoney_client.get_accounts_config()

        self.assertEqual(len(result), 2)
        aids = {r["aid"] for r in result}
        self.assertEqual(aids, {"10001", "10002"})

    def test_get_accounts_config_fallback_default_account(self):
        """CMONEY_COOKIE without any AID → Default_Account with aid=None."""
        import rpa_pipeline.cmoney_client as cmoney_client

        env = {"CMONEY_COOKIE": "fake_cookie"}
        with patch("rpa_pipeline.cmoney_client.load_dotenv", autospec=True):
            with patch.dict(os.environ, env, clear=True):
                with patch.object(cmoney_client, "get_auto_aids", return_value=[]):
                    result = cmoney_client.get_accounts_config()

        self.assertEqual(len(result), 1)
        self.assertIsNone(result[0]["aid"])
        self.assertEqual(result[0]["name"], "Default_Account")

    def test_all_symbols_importable(self):
        """signal_validator exposes all expected public symbols."""
        import rpa_pipeline.signal_validator as signal_validator

        for sym in ["SignalError", "EXECUTION_LOG_FILE", "load_signal",
                    "record_signal", "signal_was_executed"]:
            self.assertTrue(
                hasattr(signal_validator, sym),
                msg=f"Missing symbol: signal_validator.{sym}",
            )

    def test_objects_are_same_as_cmoney_rpa(self):
        """Symbols in signal_validator are the same objects as in cmoney_rpa."""
        import rpa_pipeline.cmoney_rpa as cmoney_rpa
        import rpa_pipeline.signal_validator as signal_validator

        for sym in ["SignalError", "load_signal", "record_signal", "signal_was_executed"]:
            self.assertIs(
                getattr(signal_validator, sym),
                getattr(cmoney_rpa, sym),
                msg=f"signal_validator.{sym} is not cmoney_rpa.{sym}",
            )

    def test_signal_error_is_value_error_subclass(self):
        """SignalError from signal_validator is a ValueError subclass."""
        from rpa_pipeline.signal_validator import SignalError

        self.assertTrue(issubclass(SignalError, ValueError))


# ---------------------------------------------------------------------------
# rebalance_planner — facade correctness
# ---------------------------------------------------------------------------

class RebalancePlannerFacadeTests(unittest.TestCase):
    def test_all_symbols_importable(self):
        """rebalance_planner exposes all expected public symbols."""
        import rpa_pipeline.rebalance_planner as rebalance_planner

        for sym in ["build_rebalance_plan", "build_dry_run_diff",
                    "write_dry_run_diff", "run_signal_file"]:
            self.assertTrue(
                hasattr(rebalance_planner, sym),
                msg=f"Missing symbol: rebalance_planner.{sym}",
            )

    def test_objects_are_same_as_cmoney_rpa(self):
        """Symbols in rebalance_planner are the same objects as in cmoney_rpa."""
        import rpa_pipeline.cmoney_rpa as cmoney_rpa
        import rpa_pipeline.rebalance_planner as rebalance_planner

        for sym in ["build_rebalance_plan", "build_dry_run_diff", "run_signal_file"]:
            self.assertIs(
                getattr(rebalance_planner, sym),
                getattr(cmoney_rpa, sym),
                msg=f"rebalance_planner.{sym} is not cmoney_rpa.{sym}",
            )

    def test_build_rebalance_plan_via_facade(self):
        """build_rebalance_plan works correctly when called via rebalance_planner."""
        from rpa_pipeline.rebalance_planner import build_rebalance_plan

        plan = build_rebalance_plan({"2330": 1, "2317": 3}, {"2330": 2, "2317": 0})
        self.assertEqual(plan["buys"], {"2330": 1})
        self.assertEqual(plan["sells"], {"2317": 3})


# ---------------------------------------------------------------------------
# Backward compatibility — cmoney_rpa still re-exports everything
# ---------------------------------------------------------------------------

class BackwardCompatTests(unittest.TestCase):
    def test_cmoney_rpa_signal_symbols_still_accessible(self):
        """All signal-related symbols remain accessible via cmoney_rpa."""
        import rpa_pipeline.cmoney_rpa as cmoney_rpa

        for sym in ["SignalError", "EXECUTION_LOG_FILE", "load_signal",
                    "record_signal", "signal_was_executed"]:
            self.assertTrue(
                hasattr(cmoney_rpa, sym),
                msg=f"Backward compat broken: cmoney_rpa.{sym} missing",
            )

    def test_cmoney_rpa_rebalance_symbols_still_accessible(self):
        """All rebalance-related symbols remain accessible via cmoney_rpa."""
        import rpa_pipeline.cmoney_rpa as cmoney_rpa

        for sym in ["build_rebalance_plan", "build_dry_run_diff",
                    "write_dry_run_diff", "run_signal_file"]:
            self.assertTrue(
                hasattr(cmoney_rpa, sym),
                msg=f"Backward compat broken: cmoney_rpa.{sym} missing",
            )

    def test_cmoney_rpa_account_symbols_still_accessible(self):
        """get_auto_aids and get_accounts_config still accessible via cmoney_rpa."""
        import rpa_pipeline.cmoney_rpa as cmoney_rpa

        self.assertTrue(callable(cmoney_rpa.get_auto_aids))
        self.assertTrue(callable(cmoney_rpa.get_accounts_config))

    def test_trade_guard_imports_still_resolve(self):
        """trade_guard can import CMoneyRPA, build_dry_run_diff, load_signal from cmoney_rpa."""
        import importlib

        # Re-import rpa_pipeline.trade_guard as trade_guard fresh to check its import chain
        import rpa_pipeline.trade_guard as trade_guard
        importlib.reload(trade_guard)

        # trade_guard should expose generate_diff (its public API)
        self.assertTrue(callable(trade_guard.generate_diff))


if __name__ == "__main__":
    unittest.main()
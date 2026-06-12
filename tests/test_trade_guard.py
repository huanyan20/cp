import json
import os
import sys
import tempfile
import unittest
from datetime import UTC, datetime, timedelta
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import rpa_pipeline.daily_trade_runner as daily_trade_runner
import rpa_pipeline.trade_guard as trade_guard


class AccountStatusOnlyRPA:
    def get_account_status(self):
        return {
            "total_assets": 1_000_000,
            "available_cash": 200_000,
            "inventory": {"2330": 1},
        }

    def close(self):
        pass


class TradeGuardTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.tmp_path = Path(self.tmp.name)
        self.old_guard_rpa = trade_guard.CMoneyRPA
        self.old_daily_aid = daily_trade_runner.LIVE_AID
        self.old_daily_ttl = daily_trade_runner.LIVE_TTL_SECONDS
        self.old_daily_diff_path = daily_trade_runner.DRY_RUN_DIFF_PATH

    def tearDown(self):
        trade_guard.CMoneyRPA = self.old_guard_rpa
        daily_trade_runner.LIVE_AID = self.old_daily_aid
        daily_trade_runner.LIVE_TTL_SECONDS = self.old_daily_ttl
        daily_trade_runner.DRY_RUN_DIFF_PATH = self.old_daily_diff_path
        self.tmp.cleanup()

    def write_json(self, name, payload):
        path = self.tmp_path / name
        path.write_text(json.dumps(payload), encoding="utf-8")
        return path

    def signal(self, **overrides):
        payload = {
            "signal_id": "sig-guard",
            "created_at": datetime.now(UTC).isoformat(),
            "aid": "2249294",
            "target_lots": {"2330": 2},
        }
        payload.update(overrides)
        return payload

    def macro_guard(self, level="OK"):
        return {
            "level": level,
            "critical_reasons": ["critical reason"] if level == "CRITICAL" else [],
            "warn_reasons": ["warn reason"] if level == "WARN" else [],
        }

    def test_live_flag_false_blocks_live_execution(self):
        daily_trade_runner.LIVE_AID = "2249294"
        with patch.dict(os.environ, {"ENABLE_LIVE_TRADING": "false"}, clear=False):
            with self.assertRaisesRegex(RuntimeError, "ENABLE_LIVE_TRADING"):
                daily_trade_runner._require_live_execution_context()

    def test_aid_mismatch_blocks_signal_guard(self):
        daily_trade_runner.LIVE_AID = "2249294"
        signal_path = self.write_json("signal.json", self.signal(aid="999"))
        guard_path = self.write_json("guard.json", self.macro_guard("OK"))

        with self.assertRaisesRegex(RuntimeError, "signal aid mismatch"):
            daily_trade_runner._require_signal_guard(str(signal_path), str(guard_path))

    def test_expired_signal_blocks_signal_guard(self):
        daily_trade_runner.LIVE_AID = "2249294"
        daily_trade_runner.LIVE_TTL_SECONDS = 60
        signal_path = self.write_json(
            "signal.json",
            self.signal(created_at=(datetime.now(UTC) - timedelta(minutes=5)).isoformat()),
        )
        guard_path = self.write_json("guard.json", self.macro_guard("OK"))

        with self.assertRaisesRegex(RuntimeError, "signal expired"):
            daily_trade_runner._require_signal_guard(str(signal_path), str(guard_path))

    def test_macro_warn_blocks_live_signal_guard(self):
        daily_trade_runner.LIVE_AID = "2249294"
        signal_path = self.write_json("signal.json", self.signal())
        guard_path = self.write_json("guard.json", self.macro_guard("WARN"))

        with self.assertRaisesRegex(RuntimeError, "macro guard is not OK"):
            daily_trade_runner._require_signal_guard(str(signal_path), str(guard_path))

    def test_dry_run_diff_missing_blocks_live(self):
        daily_trade_runner.LIVE_AID = "2249294"
        daily_trade_runner.DRY_RUN_DIFF_PATH = str(self.tmp_path / "missing_diff.json")

        with patch.object(daily_trade_runner, "run_command", return_value=(True, "ok")):
            with self.assertRaisesRegex(RuntimeError, "dry-run diff file was not created"):
                daily_trade_runner._require_dry_run_diff("signal.json")

    def test_dry_run_diff_failed_risk_checks_block_live(self):
        daily_trade_runner.LIVE_AID = "2249294"
        diff_path = self.write_json(
            "diff.json",
            {"risk_checks": {"passed": False, "reasons": ["total exposure too high"]}},
        )
        daily_trade_runner.DRY_RUN_DIFF_PATH = str(diff_path)

        with patch.object(daily_trade_runner, "run_command", return_value=(True, "ok")):
            with self.assertRaisesRegex(RuntimeError, "risk checks failed"):
                daily_trade_runner._require_dry_run_diff("signal.json")

    def test_trade_guard_blocks_target_weight_risk_limit_breach(self):
        trade_guard.CMoneyRPA = lambda aid: AccountStatusOnlyRPA()
        signal_path = self.write_json(
            "signal.json",
            self.signal(target_weights={"2330": 0.8}, target_lots={"2330": 2}),
        )

        with self.assertRaisesRegex(RuntimeError, "max single weight"):
            trade_guard.generate_diff(str(signal_path), "2249294", str(self.tmp_path / "diff.json"))


if __name__ == "__main__":
    unittest.main()

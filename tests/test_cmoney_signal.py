import json
import sys
import tempfile
import unittest
from datetime import UTC, datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cmoney_rpa
import trade_guard


class FakeRPA:
    def __init__(self, aid, headless=True):
        self.aid = aid
        self.actions = []

    def inventory(self):
        return [{"Id": "2330", "IQty": 1}, {"Id": "2317", "IQty": 3}]

    def orders(self):
        return [{"CanDel": "1", "CNo": "A001"}, {"CanDel": "0", "CNo": "B002"}]

    def cancel_all(self):
        self.actions.append("cancel_all")
        return [{"CNo": "A001", "result": {"Success": True}}]

    def buy(self, sid, qty, price="漲停"):
        self.actions.append(("buy", sid, qty))
        return {"Success": True}

    def sell(self, sid, qty, price="跌停"):
        self.actions.append(("sell", sid, qty))
        return {"Success": True}

    def close(self):
        self.actions.append("close")


class AccountStatusOnlyRPA:
    def __init__(self, aid):
        self.aid = aid
        self.closed = False

    def get_account_status(self):
        return {
            "total_assets": 1_000_000,
            "available_cash": 200_000,
            "inventory": {"2330": 1, "2317": 3},
        }

    def close(self):
        self.closed = True


class CMoneySignalTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.tmp_path = Path(self.tmp.name)
        self.old_log = cmoney_rpa.EXECUTION_LOG_FILE
        self.old_rpa = cmoney_rpa.CMoneyRPA
        self.old_guard_rpa = trade_guard.CMoneyRPA
        cmoney_rpa.EXECUTION_LOG_FILE = self.tmp_path / "execution_log.json"

    def tearDown(self):
        cmoney_rpa.EXECUTION_LOG_FILE = self.old_log
        cmoney_rpa.CMoneyRPA = self.old_rpa
        trade_guard.CMoneyRPA = self.old_guard_rpa
        self.tmp.cleanup()

    def write_signal(self, payload):
        path = self.tmp_path / "signal.json"
        path.write_text(json.dumps(payload), encoding="utf-8")
        return path

    def base_signal(self):
        return {
            "signal_id": "sig-1",
            "created_at": datetime.now(UTC).isoformat(),
            "aid": "2249294",
            "target_lots": {"2330": 2, "2317.TW": 0},
        }

    def test_load_signal_accepts_target_lots(self):
        signal = cmoney_rpa.load_signal(str(self.write_signal(self.base_signal())), "2249294")
        self.assertEqual(signal["target_lots"], {"2330": 2, "2317": 0})

    def test_target_lots_takes_priority_when_both_are_present(self):
        payload = self.base_signal()
        payload["target_weights"] = {"2330.TW": 0.5}
        signal = cmoney_rpa.load_signal(str(self.write_signal(payload)), "2249294")
        self.assertEqual(signal["target_lots"], {"2330": 2, "2317": 0})
        self.assertEqual(signal["target_weights"], {"2330": 0.5})

    def test_rejects_missing_required_field(self):
        payload = self.base_signal()
        del payload["signal_id"]
        with self.assertRaises(cmoney_rpa.SignalError):
            cmoney_rpa.load_signal(str(self.write_signal(payload)), "2249294")

    def test_rejects_negative_or_fractional_lots(self):
        payload = self.base_signal()
        payload["target_lots"] = {"2330": -1}
        with self.assertRaises(cmoney_rpa.SignalError):
            cmoney_rpa.load_signal(str(self.write_signal(payload)), "2249294")

        payload["target_lots"] = {"2330": 1.5}
        with self.assertRaises(cmoney_rpa.SignalError):
            cmoney_rpa.load_signal(str(self.write_signal(payload)), "2249294")

    def test_rejects_negative_weights(self):
        payload = self.base_signal()
        del payload["target_lots"]
        payload["target_weights"] = {"2330.TW": -0.1}
        with self.assertRaises(cmoney_rpa.SignalError):
            cmoney_rpa.load_signal(str(self.write_signal(payload)), "2249294")

    def test_rejects_expired_signal_and_aid_mismatch(self):
        payload = self.base_signal()
        payload["created_at"] = (datetime.now(UTC) - timedelta(hours=2)).isoformat()
        with self.assertRaises(cmoney_rpa.SignalError):
            cmoney_rpa.load_signal(str(self.write_signal(payload)), "2249294", ttl_seconds=3600)

        payload = self.base_signal()
        payload["aid"] = "999"
        with self.assertRaises(cmoney_rpa.SignalError):
            cmoney_rpa.load_signal(str(self.write_signal(payload)), "2249294")

    def test_build_rebalance_plan(self):
        plan = cmoney_rpa.build_rebalance_plan({"2330": 1, "2317": 3}, {"2330": 2, "2317": 0})
        self.assertEqual(plan["buys"], {"2330": 1})
        self.assertEqual(plan["sells"], {"2317": 3})

    def test_dry_run_does_not_cancel_or_order(self):
        fake = FakeRPA("2249294")
        cmoney_rpa.CMoneyRPA = lambda aid, headless=True: fake

        cmoney_rpa.run_signal_file(
            aid="2249294",
            signal_path=str(self.write_signal(self.base_signal())),
            execute=False,
            cancel_first=True,
        )

        self.assertNotIn("cancel_all", fake.actions)
        self.assertNotIn(("buy", "2330", 1), fake.actions)
        self.assertIn("close", fake.actions)

    def test_successful_signal_is_idempotent(self):
        cmoney_rpa.record_signal("sig-1", "success", {"ok": True})
        self.assertTrue(cmoney_rpa.signal_was_executed("sig-1"))

    def test_trade_guard_generates_diff_from_account_status_inventory(self):
        fake = AccountStatusOnlyRPA("2249294")
        trade_guard.CMoneyRPA = lambda aid: fake
        output_path = self.tmp_path / "trade_guard_diff.json"

        written = trade_guard.generate_diff(
            str(self.write_signal(self.base_signal())),
            "2249294",
            str(output_path),
        )

        diff = json.loads(written.read_text(encoding="utf-8"))
        self.assertTrue(fake.closed)
        self.assertEqual(diff["current_lots"], {"2330": 1, "2317": 3})
        self.assertEqual(diff["plan"]["buys"], {"2330": 1})
        self.assertEqual(diff["plan"]["sells"], {"2317": 3})


if __name__ == "__main__":
    unittest.main()

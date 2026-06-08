import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import experiment_report


def metric_payload(seed=42, sortino=1.1):
    return {
        "algo": "ppo",
        "seed": seed,
        "cash_mode": "enabled",
        "enable_cash_action": True,
        "enable_margin_short": False,
        "overall": {
            "sortino": sortino,
            "max_drawdown": -0.10,
            "total_return": 0.22,
            "avg_cash_weight": 0.06,
            "cash_weight_std": 0.04,
            "cash_corr_next_return": -0.10,
            "turnover": 0.04,
            "win_rate": 0.55,
        },
        "periods": {
            "2024H2": {
                "sortino": sortino,
                "max_drawdown": -0.08,
                "total_return": 0.08,
                "avg_cash_weight": 0.05,
                "cash_weight_std": 0.03,
                "cash_corr_next_return": -0.05,
                "turnover": 0.03,
                "win_rate": 0.56,
                "test_start": "2024-07-01",
                "test_end": "2024-12-31",
                "was_clamped": False,
            }
        },
    }


class ExperimentReportTests(unittest.TestCase):
    def write_json(self, directory, name, payload):
        path = Path(directory) / name
        path.write_text(json.dumps(payload), encoding="utf-8")
        return path

    def test_metric_reader_includes_with_features_results(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.write_json(
                tmp,
                "metrics_ppo_enabled_with_features_wf_seed42.json",
                metric_payload(),
            )

            records = experiment_report._read_metric_files(tmp)

        self.assertEqual(len(records), 1)
        self.assertTrue(records[0]["path"].endswith("_with_features_wf_seed42.json"))
        self.assertEqual(records[0]["variant"], "with_features")

    def test_summary_groups_base_and_with_features_separately(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.write_json(
                tmp,
                "metrics_ppo_enabled_wf_seed42.json",
                metric_payload(seed=42, sortino=2.0),
            )
            self.write_json(
                tmp,
                "metrics_ppo_enabled_with_features_wf_seed42.json",
                metric_payload(seed=42, sortino=1.0),
            )

            records = experiment_report._read_metric_files(tmp)
            overall_df = experiment_report._overall_dataframe(records)
            period_df = experiment_report._period_dataframe(records)
            summary_df, raw_summary, period_summary_df = experiment_report._summary_tables(
                overall_df, period_df
            )

        self.assertEqual(set(summary_df["Variant"]), {"base", "with_features"})
        self.assertEqual({row["variant"] for row in raw_summary}, {"base", "with_features"})
        self.assertEqual(set(period_summary_df["Variant"]), {"base", "with_features"})

    def test_generate_report_includes_promotion_gate_and_optional_summaries(self):
        with tempfile.TemporaryDirectory() as tmp:
            for seed, sortino in [(42, 1.1), (43, 1.2), (44, 1.0)]:
                self.write_json(
                    tmp,
                    f"metrics_ppo_enabled_with_features_wf_seed{seed}.json",
                    metric_payload(seed=seed, sortino=sortino),
                )
            self.write_json(
                tmp,
                "baseline_summary.json",
                {
                    "buy_and_hold": {"total_return": 0.10, "sharpe": 0.5},
                    "^TWII": {"total_return": 0.08, "sharpe": 0.4},
                    "0050": {"total_return": 0.09, "sharpe": 0.45},
                },
            )
            self.write_json(
                tmp,
                "ablation_summary.json",
                {
                    "overnight_features": {
                        "with_feature": {"sortino": 1.1},
                        "without_feature": {"sortino": 0.9},
                    }
                },
            )
            self.write_json(
                tmp,
                "stress_summary.json",
                {
                    "tests": {
                        "fee_1bp": {"total_return": 0.20},
                        "slippage_1bp": {"total_return": 0.19},
                        "spread_2bp": {"total_return": 0.18},
                    }
                },
            )
            output_md = Path(tmp) / "report.md"
            output_json = Path(tmp) / "summary.json"

            experiment_report.generate_report(
                results_dir=tmp,
                output_md=str(output_md),
                output_json=str(output_json),
            )

            md = output_md.read_text(encoding="utf-8")
            summary = json.loads(output_json.read_text(encoding="utf-8"))

        self.assertIn("## 0. Promotion Decision", md)
        self.assertIn("Baseline summary loaded", md)
        self.assertIn("Feature ablation summary loaded", md)
        self.assertIn("Stress test summary loaded", md)
        self.assertIn("promotion_gate", summary)
        period_gate = next(
            gate
            for gate in summary["promotion_gate"]["gates"]
            if gate["name"] == "Period Consistency"
        )
        self.assertTrue(period_gate["passed"])
        self.assertTrue(summary["baselines"])
        self.assertTrue(summary["ablations"])
        self.assertTrue(summary["stress_tests"])


if __name__ == "__main__":
    unittest.main()

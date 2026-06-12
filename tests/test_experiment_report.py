import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import scripts.experiment_report as experiment_report
from env_config import build_env_config_snapshot


def metric_payload(seed=42, sortino=1.1, env_tagged=True):
    payload = {
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
    if env_tagged:
        env_config = build_env_config_snapshot()
        payload["env_config_version"] = env_config["version"]
        payload["env_config_hash"] = env_config["hash"]
        payload["env_config"] = env_config
    return payload


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

    def test_base_variant_outranks_with_features_even_if_lower_sortino(self):
        # O3: with_features is a risk overlay and must never outrank base, even when
        # its Sortino is higher.
        with tempfile.TemporaryDirectory() as tmp:
            self.write_json(
                tmp,
                "metrics_ppo_enabled_wf_seed42.json",
                metric_payload(seed=42, sortino=1.0),
            )
            self.write_json(
                tmp,
                "metrics_ppo_enabled_with_features_wf_seed42.json",
                metric_payload(seed=42, sortino=9.0),
            )

            records = experiment_report._read_metric_files(tmp)
            overall_df = experiment_report._overall_dataframe(records)
            period_df = experiment_report._period_dataframe(records)
            _, raw_summary, _ = experiment_report._summary_tables(overall_df, period_df)

        self.assertEqual(raw_summary[0]["variant"], "base")

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
                    "Semi_2x": {"total_return": 0.08, "sharpe": 0.4},
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

    def test_filter_excludes_legacy_when_current_tagged_metrics_exist(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.write_json(
                tmp,
                "metrics_ppo_enabled_wf_seed42.json",
                metric_payload(seed=42, sortino=2.0, env_tagged=True),
            )
            self.write_json(
                tmp,
                "metrics_ppo_enabled_wf_seed43.json",
                metric_payload(seed=43, sortino=1.0, env_tagged=False),
            )

            records, notes = experiment_report._filter_records_by_env_config(
                experiment_report._read_metric_files(tmp),
                current_env_only=True,
            )

        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["seed"], 42)
        self.assertTrue(any("Excluded" in note for note in notes))

    def test_filter_by_env_config_version(self):
        with tempfile.TemporaryDirectory() as tmp:
            tagged = metric_payload(seed=42, sortino=2.0, env_tagged=True)
            tagged["algo"] = "sac"
            tagged["env_config_version"] = "legacy_test"
            self.write_json(tmp, "metrics_sac_enabled_wf_seed42.json", tagged)
            self.write_json(
                tmp,
                "metrics_ppo_enabled_wf_seed43.json",
                metric_payload(seed=43, sortino=1.0, env_tagged=True),
            )

            records, notes = experiment_report._filter_records_by_env_config(
                experiment_report._read_metric_files(tmp),
                env_config_version="legacy_test",
            )

        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["algo"], "sac")
        self.assertTrue(any("legacy_test" in note for note in notes))


    def test_generate_report_includes_sl_vs_rl_comparison(self):
        with tempfile.TemporaryDirectory() as tmp:
            tagged = metric_payload(seed=42, sortino=2.0)
            tagged["algo"] = "sac"
            self.write_json(tmp, "metrics_sac_enabled_wf_seed42.json", tagged)
            sl_payload = {
                "strategy": "sl_rule",
                "allocator": "rule",
                "algo": "sl_lightgbm",
                "horizon": 5,
                "seed": 42,
                "cash_mode": "enabled",
                "overall": {
                    "sortino": 1.6,
                    "max_drawdown": 0.10,
                    "total_return": 0.20,
                    "avg_cash_weight": 0.15,
                    "cash_weight_std": 0.05,
                    "cash_corr_next_return": -0.05,
                    "turnover": 0.02,
                    "win_rate": 0.54,
                },
                "periods": {
                    "2024H2": {
                        "sortino": 1.6,
                        "max_drawdown": 0.08,
                        "total_return": 0.07,
                        "avg_cash_weight": 0.14,
                        "cash_weight_std": 0.04,
                        "turnover": 0.02,
                        "win_rate": 0.55,
                        "test_start": "2024-07-01",
                        "test_end": "2024-12-31",
                    }
                },
            }
            self.write_json(tmp, "metrics_sl_rule_h5_seed42.json", sl_payload)
            md_path = Path(tmp) / "report.md"
            json_path = Path(tmp) / "summary.json"
            experiment_report.generate_report(
                results_dir=tmp,
                output_md=str(md_path),
                output_json=str(json_path),
            )
            md_text = md_path.read_text(encoding="utf-8")
            summary = json.loads(json_path.read_text(encoding="utf-8"))

        self.assertIn("### 8d. SL vs RL Comparison", md_text)
        self.assertIn("#### Overall Metrics", md_text)
        self.assertIn("#### Verdict", md_text)
        comparison = summary.get("sl_vs_rl_comparison")
        self.assertIsNotNone(comparison)
        self.assertIn("overall", comparison)
        self.assertIn("verdict", comparison)
        self.assertTrue(len(comparison["verdict"]) > 0)
        sortino_row = next(r for r in comparison["overall"] if r["metric"] == "sortino")
        self.assertEqual(sortino_row["winner"], "RL")

    def test_generate_report_includes_sl_baseline_section(self):
        with tempfile.TemporaryDirectory() as tmp:
            sl_payload = {
                "strategy": "sl_rule",
                "allocator": "rule",
                "algo": "sl_lightgbm",
                "horizon": 5,
                "seed": 42,
                "cash_mode": "enabled",
                "overall": {
                    "sortino": 1.4,
                    "max_drawdown": 0.11,
                    "total_return": 0.18,
                    "avg_cash_weight": 0.15,
                    "cash_weight_std": 0.05,
                    "cash_corr_next_return": -0.05,
                    "turnover": 0.03,
                    "win_rate": 0.54,
                },
                "periods": {
                    "2025H1": {
                        "sortino": 1.4,
                        "max_drawdown": 0.09,
                        "total_return": 0.08,
                        "avg_cash_weight": 0.14,
                        "cash_weight_std": 0.04,
                        "turnover": 0.02,
                        "win_rate": 0.55,
                        "test_start": "2025-01-01",
                        "test_end": "2025-06-30",
                    }
                },
            }
            self.write_json(tmp, "metrics_sl_rule_h5_seed42.json", sl_payload)
            md_path = Path(tmp) / "report.md"
            json_path = Path(tmp) / "summary.json"
            experiment_report.generate_report(
                results_dir=tmp,
                output_md=str(md_path),
                output_json=str(json_path),
                current_env_only=False,
            )
            md_text = md_path.read_text(encoding="utf-8")
            summary = json.loads(json_path.read_text(encoding="utf-8"))

        self.assertIn("## 8. SL Baseline", md_text)
        self.assertIn("SL Promotion Gate", md_text)
        self.assertIn("sl_baseline", summary)
        self.assertIsNotNone(summary["sl_baseline"]["promotion_gate"])


if __name__ == "__main__":
    unittest.main()

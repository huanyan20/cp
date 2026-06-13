"""
Integration test to verify walk_forward refactored code can initialize and parse arguments
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from research_pipeline import build_artifact_paths, build_period_plan
from walk_forward import cash_mode_name, cash_modes_from_arg, parse_seeds


def test_walk_forward_initialization():
    """Test that walk_forward components can initialize correctly."""

    # Test parse_seeds
    seeds = parse_seeds("42,43,44")
    assert seeds == [42, 43, 44], f"Expected [42,43,44], got {seeds}"
    print("✓ parse_seeds works")

    # Test cash_modes_from_arg
    modes_enabled = cash_modes_from_arg("enabled")
    assert modes_enabled == [True], f"Expected [True], got {modes_enabled}"
    modes_disabled = cash_modes_from_arg("disabled")
    assert modes_disabled == [False], f"Expected [False], got {modes_disabled}"
    modes_both = cash_modes_from_arg("both")
    assert modes_both == [True, False], f"Expected [True, False], got {modes_both}"
    print("✓ cash_modes_from_arg works")

    # Test cash_mode_name
    assert cash_mode_name(True) == "enabled"
    assert cash_mode_name(False) == "disabled"
    print("✓ cash_mode_name works")

    # Test period planning
    periods = build_period_plan(today="2026-06-07")
    assert len(periods) == 5, f"Expected 5 periods, got {len(periods)}"
    assert periods[0]["name"] == "2022_BEAR"
    assert "effective_test_end" in periods[0]
    assert "effective_train_end" in periods[0]
    print("✓ build_period_plan works")

    # Test artifact paths
    paths = build_artifact_paths(
        algo="ppo",
        cash_mode="enabled",
        seed=42,
        feature_suffix="_with_features",
        results_dir="results_dir",
        period_name="2024H2",
    )
    assert "metrics" in paths
    assert "model" in paths
    assert "chart" in paths
    assert paths["metrics"].endswith("_wf_seed42.json")
    print("✓ build_artifact_paths works")

    print("\n✅ All walk_forward integration tests passed!")


if __name__ == "__main__":
    test_walk_forward_initialization()

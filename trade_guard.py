"""Pre-trade guard for validating signals and generating dry-run diffs."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from cmoney_rpa import CMoneyRPA
from rebalance_planner import build_dry_run_diff
from signal_validator import load_signal
from settings import load_settings

SETTINGS = load_settings()


def _resolve_aid(explicit_aid: str | None = None) -> str:
    aid = explicit_aid or os.getenv("CMONEY_AID")
    if aid:
        return str(aid)
    raise ValueError("Missing aid. Set CMONEY_AID or pass --aid.")


def _inventory_rows_from_account_status(account_status: dict) -> list[dict]:
    inventory = account_status.get("inventory", {})
    if isinstance(inventory, dict):
        return [{"Id": sid, "IQty": qty} for sid, qty in inventory.items()]
    return inventory


def _load_inventory_rows(rpa: CMoneyRPA) -> list[dict]:
    if hasattr(rpa, "get_account_status"):
        return _inventory_rows_from_account_status(rpa.get_account_status())
    if hasattr(rpa, "inventory"):
        return rpa.inventory()
    raise AttributeError("CMoneyRPA must provide get_account_status() or inventory().")


def evaluate_risk_limits(signal: dict) -> dict:
    target_weights = signal.get("target_weights") or {}
    if not target_weights:
        return {
            "checked": False,
            "passed": True,
            "reasons": [],
            "max_single_weight": SETTINGS.risk_limits.max_single_weight,
            "max_total_exposure": SETTINGS.risk_limits.max_total_exposure,
        }

    single_exposure = {sid: abs(float(weight)) for sid, weight in target_weights.items()}
    max_single = max(single_exposure.values(), default=0.0)
    total_exposure = sum(single_exposure.values())
    reasons = []
    if max_single > SETTINGS.risk_limits.max_single_weight:
        reasons.append(
            f"max single weight {max_single:.4f} exceeds {SETTINGS.risk_limits.max_single_weight:.4f}"
        )
    if total_exposure > SETTINGS.risk_limits.max_total_exposure:
        reasons.append(
            f"total exposure {total_exposure:.4f} exceeds {SETTINGS.risk_limits.max_total_exposure:.4f}"
        )

    return {
        "checked": True,
        "passed": not reasons,
        "reasons": reasons,
        "max_single_weight": SETTINGS.risk_limits.max_single_weight,
        "max_total_exposure": SETTINGS.risk_limits.max_total_exposure,
        "observed_max_single_weight": max_single,
        "observed_total_exposure": total_exposure,
    }


def generate_diff(signal_path: str, aid: str, output_path: str | None = None) -> Path:
    signal = load_signal(signal_path, aid, ttl_seconds=SETTINGS.live.signal_ttl_seconds)
    rpa = CMoneyRPA(aid=aid)
    try:
        inventory = _load_inventory_rows(rpa)
        diff = build_dry_run_diff(signal, inventory)
        risk_checks = evaluate_risk_limits(signal)
        diff["risk_checks"] = risk_checks
        if not risk_checks["passed"]:
            raise RuntimeError("; ".join(risk_checks["reasons"]))
        output = Path(output_path or SETTINGS.live.dry_run_diff_path)
        output.write_text(json.dumps(diff, indent=4, ensure_ascii=False), encoding="utf-8")
        return output
    finally:
        rpa.close()


def main():
    parser = argparse.ArgumentParser(description="Generate and validate a pre-trade dry-run diff.")
    parser.add_argument("--signal", required=True, help="Path to signal.json")
    parser.add_argument("--aid", default=None, help="CMoney aid; defaults to CMONEY_AID")
    parser.add_argument("--output", default=None, help="Optional diff output path")
    args = parser.parse_args()

    aid = _resolve_aid(args.aid)
    output = generate_diff(args.signal, aid, args.output)
    print(output)


if __name__ == "__main__":
    main()

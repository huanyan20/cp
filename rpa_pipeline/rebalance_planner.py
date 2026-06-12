import sys, os; sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
"""Rebalance planning: compute buy/sell diffs and orchestrate signal execution.

Exported symbols
----------------
- ``build_rebalance_plan``  ¡X compute buy/sell lot diffs from current vs target
- ``build_dry_run_diff``    ¡X produce a dry-run diff JSON payload
- ``write_dry_run_diff``    ¡X high-level helper: load signal, query RPA, write JSON
- ``run_signal_file``       ¡X orchestrate a full signal file execution / dry-run
- ``_current_lots_from_rpa`` ¡X extract current lot counts from an RPA instance
"""

from __future__ import annotations

import json

from signal_validator import _normalize_sid, load_signal, record_signal


def build_dry_run_diff(signal: dict, inventory: list[dict]) -> dict:
    target_lots = signal.get("target_lots", {})
    current_lots = {
        _normalize_sid(item["Id"]): int(item.get("IQty", 0)) for item in inventory
    }
    plan = build_rebalance_plan(current_lots, target_lots)
    return {
        "signal_id": signal["signal_id"],
        "aid": signal["aid"],
        "created_at": signal["created_at"],
        "current_lots": current_lots,
        "target_lots": target_lots,
        "plan": plan,
        "net_buy_lots": int(sum(plan["buys"].values())),
        "net_sell_lots": int(sum(plan["sells"].values())),
    }

def write_dry_run_diff(signal_path: str, aid: str, output_path: str | None = None) -> str:
    signal = load_signal(signal_path, aid)
    # Late import to avoid circular dependency
    from cmoney_rpa import CMoneyRPA
    try:
        rpa = CMoneyRPA(aid=aid)
    except TypeError:
        rpa = CMoneyRPA(aid=aid)
    try:
        inventory = rpa.get_account_status()["inventory"]
        diff = build_dry_run_diff(signal, [{"Id": sid, "IQty": qty} for sid, qty in inventory.items()])
        output_path = output_path or f"dry_run_diff_{signal['signal_id']}_{aid}.json"
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(diff, f, indent=4, ensure_ascii=False)
        return output_path
    finally:
        rpa.close()

def build_rebalance_plan(current_lots: dict, target_lots: dict) -> dict:
    buys = {}
    sells = {}
    all_sids = set(current_lots) | set(target_lots)
    for sid in all_sids:
        current = int(current_lots.get(sid, 0))
        target = int(target_lots.get(sid, 0))
        diff = target - current
        if diff > 0:
            buys[sid] = diff
        elif diff < 0:
            sells[sid] = abs(diff)
    return {"buys": buys, "sells": sells}

def _current_lots_from_rpa(rpa) -> dict:
    if hasattr(rpa, "inventory"):
        inventory = rpa.inventory()
        return {_normalize_sid(i["Id"]): int(i.get("IQty", 0)) for i in inventory}
    account_info = rpa.get_account_status()
    inventory = account_info.get("inventory", {})
    return {str(k): int(v) for k, v in inventory.items()}

def run_signal_file(
    aid: str,
    signal_path: str,
    execute: bool = False,
    cancel_first: bool = False,
    headless: bool = True,
):
    signal = load_signal(signal_path, aid)
    target_lots = signal.get("target_lots", {})

    from cmoney_rpa import CMoneyRPA
    try:
        rpa = CMoneyRPA(aid=aid, headless=headless)
    except TypeError:
        rpa = CMoneyRPA(aid=aid)

    try:
        current_lots = _current_lots_from_rpa(rpa)
        plan = build_rebalance_plan(current_lots, target_lots)

        if execute and cancel_first:
            rpa.cancel_all()
        if execute:
            for sid, qty in plan["sells"].items():
                rpa.sell(sid, qty)
            for sid, qty in plan["buys"].items():
                rpa.buy(sid, qty)
            record_signal(signal["signal_id"], "success", {"plan": plan})
        return plan
    finally:
        rpa.close()

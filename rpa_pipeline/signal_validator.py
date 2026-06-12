import sys, os; sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
"""Signal validation and execution log management.

Exported symbols
----------------
- ``SignalError``           ¡X custom exception for malformed / expired signals
- ``EXECUTION_LOG_FILE``    ¡X path to the execution log (module-level constant)
- ``load_signal``           ¡X parse, validate, and normalise a signal JSON file
- ``record_signal``         ¡X append a signal ID to the execution log
- ``signal_was_executed``   ¡X check whether a signal ID has been logged
"""

from __future__ import annotations

import json
import os
from datetime import UTC, datetime

from settings import load_settings

SETTINGS = load_settings()
EXECUTION_LOG_FILE = str(SETTINGS.paths.execution_log_path)

class SignalError(ValueError):
    pass

def _normalize_sid(sid: str) -> str:
    return str(sid).split(".")[0]

def _read_execution_log():
    if not os.path.exists(EXECUTION_LOG_FILE):
        return []
    with open(EXECUTION_LOG_FILE, encoding="utf-8") as f:
        try:
            data = json.load(f)
        except json.JSONDecodeError:
            return []
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        return data.get("signals", [])
    return []

def _write_execution_log(entries):
    with open(EXECUTION_LOG_FILE, "w", encoding="utf-8") as f:
        json.dump(entries, f, indent=4, ensure_ascii=False)

def record_signal(signal_id: str, status: str = "success", details: dict | None = None):
    entries = _read_execution_log()
    if signal_id not in entries:
        entries.append(signal_id)
    _write_execution_log(entries)

def signal_was_executed(signal_id: str) -> bool:
    return signal_id in _read_execution_log()

def _validate_created_at(created_at: str, ttl_seconds: int | None):
    if ttl_seconds is None:
        return

    try:
        created = datetime.fromisoformat(created_at.replace("Z", "+00:00"))
    except Exception as exc:
        raise SignalError("Invalid created_at") from exc
    if created.tzinfo is None:
        created = created.replace(tzinfo=UTC)
    age = datetime.now(UTC) - created.astimezone(UTC)
    if age.total_seconds() > ttl_seconds:
        raise SignalError("Signal expired")

def load_signal(signal_path: str, aid: str, ttl_seconds: int | None = None) -> dict:
    with open(signal_path, encoding="utf-8") as f:
        signal = json.load(f)

    for field in ["signal_id", "created_at", "aid"]:
        if field not in signal:
            raise SignalError(f"Missing required field: {field}")
    if str(signal["aid"]) != str(aid):
        raise SignalError("AID mismatch")

    _validate_created_at(signal["created_at"], ttl_seconds)

    if "target_lots" not in signal and "target_weights" not in signal:
        raise SignalError("Missing target_lots or target_weights")

    if "target_lots" in signal:
        lots = {}
        for sid, qty in signal["target_lots"].items():
            if not isinstance(qty, int) or isinstance(qty, bool) or qty < 0:
                raise SignalError("target_lots must be non-negative integers")
            lots[_normalize_sid(sid)] = qty
        signal["target_lots"] = lots

    if "target_weights" in signal:
        weights = {}
        for sid, weight in signal["target_weights"].items():
            if float(weight) < 0:
                raise SignalError("target_weights must be non-negative")
            weights[_normalize_sid(sid)] = float(weight)
        signal["target_weights"] = weights

    return signal

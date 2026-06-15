"""Current SL candidate metadata helpers.

The active h10 candidate is intentionally named and tracked so stale gate
artifacts cannot be mistaken for the deployable state.
"""

from __future__ import annotations

import subprocess
from dataclasses import asdict, is_dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


ROOT_DIR = Path(__file__).resolve().parents[1]
CURRENT_CANDIDATE_ID = "sl_rule_h10_top20_equal_no_voltarget"
CURRENT_LABEL_MODE = "top20_positive_cross_demean"
CURRENT_PROMOTION_STATE = "blocked_until_gate_approved"


def utc_now_iso() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def _run_git(args: list[str]) -> str | None:
    try:
        result = subprocess.run(
            ["git", *args],
            cwd=ROOT_DIR,
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except Exception:
        return None
    if result.returncode != 0:
        return None
    return result.stdout.strip()


def git_snapshot() -> dict[str, Any]:
    status = _run_git(["status", "--short"])
    commit = _run_git(["rev-parse", "--short", "HEAD"])
    branch = _run_git(["branch", "--show-current"])
    status_lines = status.splitlines() if status else []
    return {
        "commit": commit,
        "branch": branch,
        "dirty": bool(status_lines),
        "status_short": status_lines,
    }


def config_snapshot(config: Any) -> dict[str, Any]:
    if config is None:
        return {}
    if is_dataclass(config):
        return asdict(config)
    if isinstance(config, dict):
        return dict(config)
    return {
        key: value
        for key, value in vars(config).items()
        if not key.startswith("_")
    }


def current_candidate_metadata(
    *,
    horizon: int,
    allocator: str,
    seed: int | None = None,
    allocator_config: Any = None,
) -> dict[str, Any]:
    return {
        "candidate_id": CURRENT_CANDIDATE_ID,
        "label_mode": CURRENT_LABEL_MODE,
        "promotion_state": CURRENT_PROMOTION_STATE,
        "horizon": horizon,
        "allocator": allocator,
        "seed": seed,
        "allocator_config": config_snapshot(allocator_config),
        "generated_at": utc_now_iso(),
        "git": git_snapshot(),
    }

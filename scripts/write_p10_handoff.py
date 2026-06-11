#!/usr/bin/env python3
"""Write P10.json handoff from completed ablation JSON (if ablation predates auto-handoff)."""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.ppo_efficiency_ablation import _write_p10_handoff  # noqa: E402


def main() -> int:
    path = Path(sys.argv[1] if len(sys.argv) > 1 else "results_dir/ppo_efficiency_ablation.json")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("smoke"):
        raise SystemExit("Refusing handoff from smoke run; use full ablation JSON.")
    _write_p10_handoff(payload, path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

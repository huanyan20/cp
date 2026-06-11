"""CP research loop orchestrator — one round, disk-backed state.

Default: dry-run (print next task + suggested agent prompt).
Use --execute only for safe actions (pytest smoke, experiment_report).

Does NOT auto-start 300K walk_forward.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STATE_PATH = ROOT / ".research" / "research_state.json"
LEDGER_PATH = ROOT / ".research" / "experiment_ledger.jsonl"
RESULTS_DIR = ROOT / "results_dir"

# R6 candidate set: SAC enabled + PPO disabled × seeds 42, 43, 44
R6_METRIC_FILES = [
    f"metrics_{algo}_{cash}_wf_seed{seed}.json"
    for algo, cash in (("sac", "enabled"), ("ppo", "disabled"))
    for seed in (42, 43, 44)
]


def load_state() -> dict:
    return json.loads(STATE_PATH.read_text(encoding="utf-8"))


def save_state(state: dict) -> None:
    state["updated_at"] = datetime.now(UTC).astimezone().isoformat(timespec="seconds")
    STATE_PATH.write_text(json.dumps(state, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def append_ledger(entry: dict) -> None:
    LEDGER_PATH.parent.mkdir(parents=True, exist_ok=True)
    with LEDGER_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def walk_forward_running() -> bool:
    if sys.platform != "win32":
        return False
    try:
        out = subprocess.run(
            [
                "powershell",
                "-NoProfile",
                "-Command",
                "(Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | "
                "Where-Object { $_.CommandLine -match 'walk_forward' }).Count",
            ],
            capture_output=True,
            text=True,
            timeout=15,
            cwd=ROOT,
        )
        return out.stdout.strip() not in ("", "0")
    except (subprocess.TimeoutExpired, OSError):
        return False


def r6_metrics_on_disk() -> bool:
    return all((RESULTS_DIR / name).exists() for name in R6_METRIC_FILES)


def sync_r6_status(state: dict) -> None:
    """Align r6_status with metrics files on disk (source of truth)."""
    if not r6_metrics_on_disk():
        return
    r6 = state.setdefault("r6_status", {})
    for seed in ("42", "43", "44"):
        r6.setdefault("sac_enabled_seeds", {})[seed] = "done"
        r6.setdefault("ppo_disabled_seeds", {})[seed] = "done"


def r6_complete(state: dict) -> bool:
    if r6_metrics_on_disk():
        return True
    r6 = state.get("r6_status", {})
    sac = r6.get("sac_enabled_seeds", {})
    ppo = r6.get("ppo_disabled_seeds", {})
    sac_done = all(v == "done" for v in sac.values()) if sac else False
    ppo_done = all(v == "done" for v in ppo.values()) if ppo else False
    return sac_done and ppo_done


def queue_done_ids(state: dict) -> set[str]:
    return {t["id"] for t in state.get("queue", []) if t.get("status") == "done"}


def pick_next_task(state: dict) -> dict | None:
    done = queue_done_ids(state)
    handoff_done = {t["id"] for t in state.get("queue", []) if t.get("status") in ("handoff", "done")}
    done |= handoff_done
    for task in state.get("queue", []):
        if task.get("status") == "in_progress":
            return task
    for task in state.get("queue", []):
        if task.get("status") not in ("pending", "blocked"):
            continue
        blocked_by = task.get("blocked_by") or []
        if task.get("id") == "freeze_r6" and r6_complete(state):
            blocked_by = [b for b in blocked_by if b != "r6_complete"]
        if all(dep in done or dep == "r6_complete" for dep in blocked_by):
            return task
    return None


def build_agent_prompt(state: dict, task: dict | None) -> str:
    if state.get("phase") == "r6_ppo_running":
        if walk_forward_running():
            return (
                "Monitor R6 walk_forward (PPO arm). Do not change train_portfolio or start R7/P8. "
                "Report progress from results_dir metrics and terminal."
            )
        return (
            "R6 phase says running but no walk_forward process found. "
            "Check if PPO needs resume: same promotion command, Resume by period."
        )
    if task is None:
        return (
            "Run experiment_report.py, update .research/research_state.json gate fields, "
            "refill queue if needed. Read cp-promotion-gate skill."
        )
    wt = task.get("worktree") or "(main repo)"
    skill = task.get("skill", "cp-research-loop")
    return (
        f"Execute queue task {task['id']}: {task.get('title', '')}. "
        f"Worktree: {wt}. Read .cursor/skills/{skill}/SKILL.md. "
        f"One variable only; append experiment_ledger.jsonl when done."
    )


def maybe_execute(state: dict, task: dict | None, execute: bool) -> None:
    if not execute:
        return
    sync_r6_status(state)
    if task and task.get("id") == "freeze_r6" and not r6_complete(state):
        print("[execute] Skipped: R6 metrics not complete on disk.")
        return
    if task and task.get("id") == "freeze_r6":
        report = subprocess.run(
            [sys.executable, "experiment_report.py"],
            cwd=ROOT,
            check=False,
        )
        baselines = ROOT / ".research" / "baselines"
        baselines.mkdir(parents=True, exist_ok=True)
        for p in RESULTS_DIR.glob("metrics_*_wf_*.json"):
            dest = baselines / p.name
            if not dest.exists():
                dest.write_text(p.read_text(encoding="utf-8"), encoding="utf-8")
        task["status"] = "done"
        state["phase"] = "post_r6"
        state["train_slot"] = {"status": "free", "owner": None, "note": None}
        save_state(state)
        append_ledger(
            {
                "ts": datetime.now(UTC).astimezone().isoformat(timespec="seconds"),
                "agent": "orchestrator",
                "action": "freeze_r6",
                "decision": f"experiment_report exit={report.returncode}",
            }
        )
        print("[execute] freeze_r6 done; phase=post_r6")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true", help="Run safe automated steps only")
    args = parser.parse_args()

    if not STATE_PATH.exists():
        print(f"Missing {STATE_PATH}")
        return 1

    state = load_state()
    sync_r6_status(state)
    task = pick_next_task(state)

    print(f"phase={state.get('phase')}")
    print(f"walk_forward_running={walk_forward_running()}")
    print(f"r6_complete={r6_complete(state)}")
    if task:
        print(f"next_task={task['id']} status={task.get('status')} title={task.get('title', '')}")
        if task.get("worktree"):
            print(f"  worktree: git worktree add {task['worktree']} -b {task.get('branch', task['id'])}")
    else:
        print("next_task=None (run triage / experiment_report)")

    print("\n--- suggested agent prompt ---")
    print(build_agent_prompt(state, task))

    if args.execute:
        maybe_execute(state, task, True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

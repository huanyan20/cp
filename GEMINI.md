# Antigravity-specific rules (P8 worktree)

- **Project root**: open `cp-p8-buffer` worktree, not main `cp` repo
- **Terminal**: allow auto-run for `pytest` and `ruff`; ask before `git push` or deleting files
- **Autonomy**: Agent-assisted mode — safe to run tests without asking each time
- **Python**: always `..\cp\env\Scripts\python.exe`, not system Python
- **Merge**: never merge to main — human integrates after cross-review memos exist

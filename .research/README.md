# CP 外部研究記憶

規格：[`docs/RESEARCH_LOOP.md`](../docs/RESEARCH_LOOP.md)

## 現況（2026-06-11）

| 項目 | 狀態 |
|------|------|
| **R6** | ✅ 24 WF 模型 + 6 metrics；Gate **BLOCKED**（MDD 44.41%） |
| **P8**（Antigravity） | ✅ handoff → [`handoffs/P8.json`](handoffs/P8.json) · branch `fe831c4` |
| **P10**（Cursor） | 🔄 worktree 程式 + smoke ablation；**缺** `handoffs/P10.json` |
| **下一步** | P10 完整 ablation → P10 handoff → cross_review → merge → R7 |

## 檔案

| 路徑 | 用途 |
|------|------|
| `research_state.json` | phase、queue、artifacts |
| `baselines/metrics_*_wf_*.json` | R6 凍結基線（6 份） |
| `handoffs/P8.json` | Antigravity P8 交接 |
| `handoffs/P10.json` | （待 Cursor 寫入） |
| `EXTERNAL_AGENT_BRIEF.md` | Antigravity 操作說明 |

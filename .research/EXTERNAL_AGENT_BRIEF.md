# External Agent Brief — Antigravity IDE（P8）

> **工具**：Google **Antigravity IDE**
> **Cursor 負責**：P10、審查 P8、R7
> **通訊**：Git branch + `.research/handoffs/` + `.research/reviews/` — 無即時對話

---

## 0. 你的角色

| 項目 | 值 |
|------|-----|
| **負責任務** | **P8** IndexedReplayBuffer → **cross_review P10** |
| **工具** | Antigravity IDE |
| **分支** | `feat/p8-indexed-replay-buffer` |
| **Worktree** | `C:\Users\ggini\Desktop\cp-p8-buffer` |
| **Agent 規則** | worktree 根目錄 `AGENTS.md` + `GEMINI.md`（由主 repo 複製） |

---

## 1. Antigravity 設定（一次性）

### 1.1 Worktree + 規則檔

```powershell
cd C:\Users\ggini\Desktop\cp
git worktree add ..\cp-p8-buffer -b feat/p8-indexed-replay-buffer
Copy-Item .research\antigravity\AGENTS.md ..\cp-p8-buffer\AGENTS.md
Copy-Item .research\antigravity\GEMINI.md ..\cp-p8-buffer\GEMINI.md
```

Antigravity 會自動讀取 worktree 根目錄的 `AGENTS.md`（跨工具）與 `GEMINI.md`（Antigravity 專用）。

### 1.2 開啟 IDE

1. 安裝：[antigravity.google/download](https://antigravity.google/download)
2. 登入個人 Gmail；建議 **Agent-assisted**（pytest 可自動跑，push 前再確認）
3. **Open Folder** → `C:\Users\ggini\Desktop\cp-p8-buffer`（⚠️ 不是 `cp` 主 repo）
4. Agent 面板首則 prompt（若 `AGENTS.md` 未自動載入可貼）：

```text
Implement P8 IndexedReplayBuffer per AGENTS.md and docs/SAC_BUFFER_PLAN.md.
When done, write handoff to C:\Users\ggini\Desktop\cp\.research\handoffs\P8.json with tool antigravity-ide.
```

### 1.3 驗證環境

```powershell
cd C:\Users\ggini\Desktop\cp-p8-buffer
..\cp\env\Scripts\python.exe -m pytest tests/ -q --tb=short
```

---

## 2. 開工前必讀

1. `docs/SAC_BUFFER_PLAN.md` — P8 規格
2. `../cp/.research/research_state.json` — queue / gate
3. `../cp/.research/baselines/metrics_sac_enabled_wf_seed*.json` — R6 基線
4. `../cp/experiment_report.md` — Gate BLOCKED（worst MDD 44.41%）

**R6 已結束**。SAC buffer 2,805 → 全歷史是首要槓桿。

---

## 3. P8 實作要點

- `IndexedReplayBuffer`：只存 `(t, account_block, action, reward, done)`；obs 由 `trading_env._market_data[t]` 重建
- 整合：`train_portfolio.py` SAC buffer 建構處
- 驗收：`pytest` 全綠 + wrap / sample 邊界測試
- **禁止**：改 PPO、VecEnv、walk_forward tier（Cursor P10 範圍）

詳見 `docs/SAC_BUFFER_PLAN.md` §2–3。

---

## 4. Handoff

寫入主 repo：

`C:\Users\ggini\Desktop\cp\.research\handoffs\P8.json`

```json
{
  "task_id": "P8",
  "phase": "handoff",
  "agent": "external",
  "tool": "antigravity-ide",
  "branch": "feat/p8-indexed-replay-buffer",
  "commit": "<git rev-parse HEAD>",
  "files_touched": ["train_portfolio.py"],
  "pytest": "pass",
  "commands_run": ["..\\cp\\env\\Scripts\\python.exe -m pytest tests/ -q"],
  "summary": "一句話摘要",
  "open_questions": ["請 Cursor review 時看 ..."],
  "ready_for_cross_review": true
}
```

---

## 5. Cross-review P10

**觸發**：`../cp/.research/handoffs/P10.json` 出現。

1. 讀 `P10.json` + `docs/SAC_BUFFER_PLAN.md` §4
2. 審 branch `feat/p10-ppo-vecenv`（唯讀；最多改 3 檔）
3. 寫 `../cp/.research/reviews/P10-reviewed-by-external.md`

Verdict：`pass` | `pass_with_nits` | `needs_changes`

---

## 6. Ledger（可選）

```json
{"ts":"2026-06-11T12:00:00+08:00","agent":"antigravity","action":"P8_handoff","task_id":"P8","decision":"pytest pass"}
```

Append 至 `../cp/.research/experiment_ledger.jsonl`。

Cursor 讀磁碟狀態即可 — **不必通知 Cursor**。

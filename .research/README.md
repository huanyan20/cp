# CP 研究狀態（精簡）

## 三個檔案就夠

| 優先 | 檔案 | 用途 |
|------|------|------|
| 1 | [`docs/RESEARCH_STRATEGY_V3.md`](../docs/RESEARCH_STRATEGY_V3.md) | **活躍路線圖** |
| 2 | [`research_state.json`](research_state.json) | queue · phase · gate · train_slot |
| 3 | [`experiment_ledger.jsonl`](experiment_ledger.jsonl) | 事件審計（讀最後 5 行） |

## 目錄一覽

| 目錄 | 用途 |
|------|------|
| [`baselines/`](baselines/README.md) | R6 凍結指標（env **r4**，勿與 r5.1 混比） |
| [`runs/`](runs/README.md) | 短跑 / 非 canonical 產物（30K 方向性） |
| [`decisions/`](decisions/README.md) | 戰略決策紀錄（v3.3 action decode） |
| [`reviews/`](reviews/V3-STRATEGY-REVIEW-BRIEF.md) | v3 外部審核 |
| [`archive/`](archive/README.md) | handoffs · 舊 reviews · WF logs（本地） |
| [`handoffs/`](handoffs/README.md) | tombstone → `archive/handoffs/` |
| [`antigravity/`](antigravity/AGENTS.md) | 外部 agent 設定 |

## 決策與審核

| 檔案 | 用途 |
|------|------|
| [`decisions/decision_algorithm_review.md`](decisions/decision_algorithm_review.md) | v3.3 action decode 修訂依據 |
| [`reviews/V3-STRATEGY-REVIEW-BRIEF.md`](reviews/V3-STRATEGY-REVIEW-BRIEF.md) | 審核 checklist |
| [`EXTERNAL_AGENT_BRIEF.md`](EXTERNAL_AGENT_BRIEF.md) | 外部 agent 設定 |

## 現況（2026-06-12）

- **phase**: `rl_rebuild_v3`
- **env**: `r5.1`（M1a+M1b+M1d+M1c done）
- **train_slot**: `free` — **M2-smoke 300K 需人類確認後啟動**
- **短跑參考**: [`runs/m2-smoke-quick-metrics.json`](runs/m2-smoke-quick-metrics.json)（2025H1 · 30K · MDD 31.9% · 非 promotion）
- **基線**: `baselines/`（r4 · worst MDD 44.41%）

```powershell
# 完整 smoke（確認後）
.\env\Scripts\python.exe walk_forward.py --tier smoke --algo sac --cash-mode enabled
```

# CP 研究狀態（精簡）

## 三個檔案就夠

| 優先 | 檔案 | 用途 |
|------|------|------|
| 1 | [`docs/RESEARCH_STRATEGY_V3.md`](../docs/RESEARCH_STRATEGY_V3.md) | **活躍路線圖** |
| 2 | [`research_state.json`](research_state.json) | queue · phase · gate |
| 3 | [`experiment_ledger.jsonl`](experiment_ledger.jsonl) | 事件審計（讀最後 5 行） |

## 再審核（round 2）

| 檔案 | 用途 |
|------|------|
| [`reviews/V3-STRATEGY-REVIEW-BRIEF.md`](reviews/V3-STRATEGY-REVIEW-BRIEF.md) | 審核 checklist |
| [`EXTERNAL_AGENT_BRIEF.md`](EXTERNAL_AGENT_BRIEF.md) | 外部 agent 設定 |
| [`reviews/V3-reviewed-by-codex.md`](reviews/V3-reviewed-by-codex.md) | 上一輪 verdict + remediation |

輸出：`.research/reviews/V3-reviewed-by-<agent>-r2.md`

## 決策紀錄

| 檔案 | 用途 |
|------|------|
| [`decisions/decision_algorithm_review.md`](decisions/decision_algorithm_review.md) | v3.3 action decode 修訂 |

## 現況

- **phase**: `rl_rebuild_v3`
- **env**: `r5.1`（M1a+M1b+M1d+M1c）
- **下一步**: M2-smoke 300K（需人類確認 train_slot）
- **短跑**: [`runs/`](runs/README.md)（30K 方向性）
- **基線**: `baselines/`（R6 凍結 · r4）
- **封存**: [`archive/`](archive/README.md)（含 abort logs）

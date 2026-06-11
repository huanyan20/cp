# External Agent Brief — v3 再審核（round 2）

> **Repo**：`C:\Users\ggini\Desktop\cp`（主 repo）  
> **任務**：審核 v3 計畫 + **精簡後目錄結構** + 程式一致性

---

## 1. 必讀（僅 4 個）

1. [`reviews/V3-STRATEGY-REVIEW-BRIEF.md`](reviews/V3-STRATEGY-REVIEW-BRIEF.md)
2. [`docs/RESEARCH_STRATEGY_V3.md`](../docs/RESEARCH_STRATEGY_V3.md)
3. [`research_state.json`](research_state.json)
4. [`.research/README.md`](README.md) — 精簡索引

上一輪：[`reviews/V3-reviewed-by-codex.md`](reviews/V3-reviewed-by-codex.md)（含 remediation）

---

## 2. 精簡後布局（應為真）

```text
docs/
  RESEARCH_STRATEGY_V3.md     ← 唯一活躍路線圖
  SAC_BUFFER_PLAN.md          ← 短 stub（指向 archive）
  archive/SAC_BUFFER_PLAN_v2.md

.research/
  research_state.json
  experiment_ledger.jsonl
  baselines/
  reviews/
    V3-STRATEGY-REVIEW-BRIEF.md
    V3-reviewed-by-codex.md
  archive/                    ← P8/P10/SAC-R handoffs & 舊 reviews
  README.md
```

**不應**在 `.research/handoffs/` 或 `.research/reviews/` 根目錄看到 `P8.json`、`P10.json`、`SAC-R-PLAN-REVIEW-BRIEF.md`。

---

## 3. 程式檢查

```powershell
cd C:\Users\ggini\Desktop\cp
Test-Path scripts\sac_gradient_ablation.py          # False
rg "SAC_GRADIENT_STEPS|SAC_BATCH_SIZE" train_portfolio.py  # 無匹配
.\env\Scripts\python.exe -m pytest tests/test_indexed_replay_buffer.py -q
```

---

## 4. 輸出

`reviews/V3-reviewed-by-<agent>-r2.md`

Verdict：`pass` | `pass_with_nits` | `needs_changes` | `block`

---

## 5. Prompt

```text
Review CP v3 strategy round 2 per .research/reviews/V3-STRATEGY-REVIEW-BRIEF.md.
Verify archive layout and no stale active-plan docs.
Write reviews/V3-reviewed-by-<agent>-r2.md.
```

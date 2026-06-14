> Superseded by 2026-06-14 SL-first strategy.
> Retained for review/history only; current implementation priority is SL-first, SAC research-only.
# v3 戰略審核 Brief — round 2（精簡後）

> **Audience**：審核 agent
> **Date**：2026-06-11
> **Repo**：`C:\Users\ggini\Desktop\cp` · `main`
> **Round 1**：[`V3-reviewed-by-codex.md`](V3-reviewed-by-codex.md)（block → remediated）

---

## 0. 審核任務

1. **v3 戰略**仍對準 Drawdown Gate（M1→M2 queue）
2. **目錄精簡**：活躍 vs `archive/` 分離是否清晰
3. **程式一致**：R7b 清理仍為真（腳本刪、無 `SAC_GRADIENT_STEPS` / `SAC_BATCH_SIZE`）
4. **易讀性**：新 agent 能否只靠 4 個檔案理解現況

**輸出**：`.research/reviews/V3-reviewed-by-<agent>-r2.md`

```text
verdict: pass | pass_with_nits | needs_changes | block
blockers: [...]
nits: [...]
stale_docs_found: [...]
m1_recommendation: GO | GO with changes | BLOCK
```

---

## 1. 活躍文件（僅這些算「現行計畫」）

| # | 路徑 |
|---|------|
| 1 | `docs/RESEARCH_STRATEGY_V3.md` |
| 2 | `.research/research_state.json` |
| 3 | `專案總覽.md` §5–6 |
| 4 | `docs/RESEARCH_PLAYBOOK.md` §1（O2） |

索引：`.research/README.md` · `docs/README.md`

---

## 2. 封存（不應出現在「下一步」）

| 位置 | 內容 |
|------|------|
| `.research/archive/` | P8/P10/SAC-R handoffs、舊 reviews、SAC-R brief |
| `docs/archive/SAC_BUFFER_PLAN_v2.md` | v2 完整排程 |
| `docs/SAC_BUFFER_PLAN.md` | **stub only**（< 15 行） |

---

## 3. 北極星與 queue（不變）

- worst MDD ≤ 35% · SAC cash=enabled · 3 seeds
- **M1a** obs POMDP → **M1b** r5 → **M2** smoke/candidate/promotion
- R7/R7b/R8/R9/SAC-R active：**已砍/封存**

---

## 4. 驗證命令

```powershell
cd C:\Users\ggini\Desktop\cp

# 布局
Get-ChildItem .research -Name
Get-ChildItem .research\archive -Recurse -Name
Test-Path scripts\sac_gradient_ablation.py
rg "SAC_GRADIENT_STEPS|SAC_BATCH_SIZE" train_portfolio.py

# 一致性
rg -n "R7 訓練中|R7 running|next.*R7b" --glob "*.md" --glob "!**/archive/**"

# 測試
.\env\Scripts\python.exe -m pytest tests/test_indexed_replay_buffer.py -q
```

---

## 5. Round 1 已修項目（應仍為真）

- [ ] `sac_gradient_ablation.py` 不存在
- [ ] `train_portfolio.py` SAC 固定 `gradient_steps=1`, `batch_size=256`
- [ ] `train_slot.status=free`
- [ ] handoffs 已移至 `archive/handoffs/`

---

## 6. 修訂紀錄

| 日期 | 變更 |
|------|------|
| 2026-06-11 | round 1 brief |
| 2026-06-11 | round 2 — 精簡布局 + 封存目錄 |

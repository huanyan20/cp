# Capital Flow 研究與狀態

> **定位**：CP 輔助研究與風控層，不取代主交易系統。打勾見 [`../../專案總覽.md`](../../專案總覽.md) §4.5。  
> **日常操作**：見 [`../README.md`](../README.md)（英文 CLI）。

---

## 能力矩陣

| 能力 | 狀態 |
|------|------|
| Preopen Guard（OK / WARN / CRITICAL） | ✅ |
| Top 3 overnight features | ✅ |
| Dry-run RPA 驗證 | ✅ |
| RL signal 產生 | ✅ |
| Top 8 macro features | ⬜ 研究中 |
| Gap model 穩定 alpha | ⬜ 不可宣稱 |
| Live trading | ⬜ 禁止（Gate BLOCKED） |

---

## 正式特徵（Top 3）

- `tsm_adr_premium_chg`
- `tsm_adr_premium`
- `TSM_ret`

Top 8 或更多 macro 須通過 walk-forward + ablation 才可升級。**已決策**：不進 RL 預設 observation（O6/R5），僅 risk overlay。

---

## Signal vs Guard

| 層 | 說明 |
|----|------|
| **Signal** | 模型輸出 `signal.json` 目標權重 |
| **Guard** | 執行端風控；`CRITICAL` = 跳過 pending buy，**不改寫** signal |

Guard 等級：`OK` 正常 · `WARN` pending buys 減半 · `CRITICAL` 停買、允許 sell-only。

---

## 資料健康門檻

`overnight_gap_features_1d.csv`：

- **< 60 rows**：僅 smoke test
- **< 250 rows**：不可宣稱穩定 alpha
- 高缺值 / stale data：不得進 promotion

---

## 研究原則

1. 先證明資料可靠 → walk-forward → ablation → 先小後大
2. OOS 下降時退回 Top 3
3. full-stock 模型 `cash_weight=0` 為預期；動態現金需 cash-enabled RL 模型

---

## 待做研究

- Top 8 ablation（Sortino / MDD / turnover）
- Guard impact 長期回測（WARN 減半 vs 全跳）
- Intraday 資料（需先解時區與穩定性）

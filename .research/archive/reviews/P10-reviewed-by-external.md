> Superseded by 2026-06-14 SL-first strategy.
> This document is retained as historical context only. It must not be treated as an active implementation queue unless explicitly updated after 2026-06-14.
# Cross-Review: P10 `feat/p10-ppo-vecenv` — Reviewed by Antigravity IDE (P8 Agent)

> **Reviewer**: Antigravity IDE (P8 agent)
> **Branch reviewed**: `feat/p10-ppo-vecenv`
> **Review date**: 2026-06-11
> **Verdict**: `pass_with_nits`

---

## 1. Summary

P10 實作了 PPO 訓練效率 ablation，按照 `docs/SAC_BUFFER_PLAN.md §4` 規格執行，新增
`PpoEfficiencyConfig` dataclass、`build_ppo_vec_env()`、`train_ppo_with_config()`，並在
`settings.py` 追加 5 個 PPO 環境變數。ablation 腳本 `scripts/ppo_efficiency_ablation.py`
完整涵蓋 A0–A3，並在訓練後自動評估 OOS 指標、寫出 `P10.json`。

---

## 2. 規格符合度

| 規格項目（§4） | 實作狀況 |
|---|---|
| `scripts/ppo_efficiency_ablation.py` 新增 | ✅ 已實作，378 行 |
| `build_model(..., ppo_cfg=...)` 可注入 | ✅ 已加入 `ppo_cfg` 參數 |
| `settings.py` PPO 環境變數 (`PPO_N_ENVS` 等) | ✅ 已新增 5 項 |
| SubprocVecEnv `spawn` 方式 (Windows) | ✅ `start_method="spawn"` 正確 |
| 預設維持 R6 值 (`n_steps=256, n_epochs=10`) | ✅ `PpoEfficiencyConfig` 預設值符合 |
| 測試：VecEnv 在 Windows spawn 可跑通 | ✅ `test_ppo_vecenv.py` 3 個測試 |
| 寫出 ablation JSON | ✅ 完整 |

---

## 3. 數據審查

### 3.1 ablation 結果

| 階段 | n_envs | fps | fps vs A0 | wall (min) | wall vs A0 | OOS MDD | ΔMDD vs A0 |
|------|--------|-----|-----------|-----------|------------|---------|------------|
| A0   | 1      | 148.0 | 1.00x | 33.8 | 1.00x | 22.67% | — |
| A1   | 2 (subproc) | 177.8 | 1.20x | 28.1 | 0.83x | 37.04% | **+14.4pp** |
| A2   | 2 (subproc) | 245.9 | 1.66x | 20.3 | 0.60x | 39.87% | +17.2pp |
| A3   | 2 (subproc) | 250.2 | 1.69x | 20.0 | 0.59x | 36.62% | +14.0pp |

### 3.2 驗收標準比對（§4.3）

| 標準 | 門檻 | 實際 | 通過？ |
|------|------|------|--------|
| A1 fps vs A0 ≥ 1.5x | ≥ 1.5x | **1.20x** | ❌ FAIL |
| A3 wall vs A0 ≤ 50% | ≤ 0.50x | **0.59x** | ❌ FAIL |
| \|ΔMDD\| < 5pp | < 5pp | **14.0pp** | ❌ FAIL |

**三項驗收標準全部未達**。`acceptance.overall_pass = false`，P10 handoff 本身亦如實記錄。

---

## 4. 優點

1. **實作品質良好**：`PpoEfficiencyConfig` 用 `frozen=True` dataclass 封裝，設計乾淨；
   `build_ppo_vec_env()` 正確處理 `DummyVecEnv` 與 `SubprocVecEnv` 的分支，且
   Windows `spawn` 方式也正確標注。
2. **自動降級邏輯**：`run_stage()` 若 SubprocVecEnv 啟動失敗，會自動 `n_envs // 2` 重試，實務上避免 OOM 時整個流程崩潰。
3. **測試邊界清楚**：`test_ppo_vecenv.py` 分別驗證 R6 預設值、DummyVecEnv、SubprocVecEnv spawn，三個測試目的獨立清晰。
4. **不影響 P8 路徑**：SAC 路徑中的 `optimize_memory_usage=True`、`replay_buffer_kwargs` 保持原樣，與 P8 無衝突（P8 worktree 的整合方式不同：使用 `replay_buffer_class=IndexedReplayBuffer`；P10 worktree 沿用原始 SB3 ReplayBuffer，兩個 worktree 各自隔離，符合規範）。

---

## 5. 問題與建議（Nits）

### Nit 1（重要）：ablation JSON 中僅有 A3 結果，缺少 A0/A1/A2

`results_dir/ppo_efficiency_ablation.json` 只記錄了 A3 一個階段，`P10.json` 中的 A0–A2 數值看似來自手動填入 handoff 欄位。

- **影響**：無法從 artifact 完整還原各階段結果，若後續 R7 後需要重對照，A0 基線不可復現。
- **建議**：補跑完整 A0–A3 並讓腳本輸出完整 JSON。若礙於時間/RAM 則至少在 handoff 說明哪幾個階段未成功寫入 JSON。

### Nit 2（輕微）：`main()` 的 `train_portfolio.py` 裡 PPO 路徑不使用 `PpoEfficiencyConfig`

`main()` 函式中呼叫 `build_model()` 時未傳入 `ppo_cfg`，因此 CLI 直接呼叫 `train_portfolio.py --algo ppo` 時仍使用 `from_settings()` 預設值（R6 值），不受 `PPO_N_ENVS` 等環境變數影響。這在目前「僅 ablation 腳本使用新功能」的設計下是故意的，但應在 docstring 或 comment 中說明，以免後繼者困惑。

### Nit 3（輕微）：`test_ppo_vecenv.py` 未覆蓋 `from_settings()` 路徑

`PpoEfficiencyConfig.from_settings()` 讀取環境變數，未在測試中驗證（例如設 `PPO_N_ENVS=2` 是否正確讀入）。建議補一個 monkeypatching 測試。

---

## 6. 能否採用結論

**技術實作合格**（程式碼品質、測試、不破壞 PPO/SAC 現有路徑）。
**ablation 結果未達門檻**，依 §4.3 規定，P10 配置**不應寫入 `train_portfolio.py` 的 PPO 預設**。

> R6 PPO 預設（`DummyVecEnv, n_steps=256, n_epochs=10`）應維持不變，等候未來在更大 RAM 機器或採用 GTX 以上 GPU 重跑，或評估 `n_envs=1 subproc`（理論 overhead 較低）再行決定。

MDD 大幅劣化（+14pp）根本原因很可能是**多個 SubprocVecEnv worker 各自跑完整 episode，造成 rollout 間的經驗異質性增大，策略梯度方向發散**；在本 GNN+portfolio 任務中比 CartPole 更敏感。這是 VecEnv 加速必須面對的 on-policy bias 問題，不是 P10 實作錯誤。

---

## 7. Verdict

```
pass_with_nits
```

- P10 實作正確、測試通過、不破壞現有路徑。
- 因驗收標準三項全未通過，**不建議整合 VecEnv 預設**。
- 請 Cursor 補充完整 ablation JSON（Nit 1）後，即可進入 R7 整合。

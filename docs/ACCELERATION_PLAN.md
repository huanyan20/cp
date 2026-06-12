# 訓練效能加速計畫 (Acceleration Plan)

> **目標**：在不改變模型研究品質的前提下，將每次 Walk-Forward 訓練時間盡量壓縮，讓 500K 步訓練能在最短時間完成。  
> **更新**：2026-06-12

---

## 各方案總覽

| 方案 | 工程成本 | 預估速度提升 | 狀態 |
|------|---------|------------|------|
| A. 超參數 Batch Size 最大化 | 低 | 1.5x | ✅ 已實作 |
| B. `torch.compile` JIT 加速 | 低 | N/A | ❌ GTX 1060 不支援 |
| C. Numba 環境核心加速 | 中 | 2–4x | ✅ 已實作 |
| D. 多 Worker 並行訓練 | 低 | N Seeds 倍 | ✅ 已實作（`--workers 6`）|
| E. GPU 張量化環境 | 高 | 30–100x | 📋 M4 預留 |

---

## 方案 A：超參數最佳化（已完成）

**原理**：增大 `batch_size` 讓 GPU 每次做更多矩陣乘法，降低閒置等待。

| 參數 | 原來 | 現在 |
|------|------|------|
| PPO `n_steps` | 256 | 512 (為避免 RAM OOM 而下調) |
| PPO `batch_size` | 64 | 128 |
| SAC `batch_size` | 256 | 1024 (有特製 Buffer 故無影響) |

**檔案**：`train_portfolio.py`

---

## 方案 B：`torch.compile` JIT 編譯（已放棄）

**原理**：PyTorch 2.0 新增的 `torch.compile()` 會把 Policy 網路的前向/反向傳播融合成更有效率的 CUDA 核心指令，省去 Python overhead。

**狀態**：❌ GTX 1060（Compute Capability 6.1）太過老舊，不支援 PyTorch Triton 編譯器（最低需求為 Volta 7.0）。嘗試強行編譯會導致 CUDA 崩潰，因此已將此段程式碼移除。

---

## 方案 C：Numba JIT 環境核心（進行中）

**原理**：`trading_env.py` 的 `_execute_trades()`、`_update_portfolio()`、`_compute_reward()` 等熱路徑（hot path）在每個 timestep 都會被呼叫數十萬次，但目前用的是標準 Python 迴圈。改用 `@numba.jit(nopython=True)` 編譯這些核心函數，可以讓 CPU 計算速度提升 3–5 倍。

**實作策略**：
- 將環境的核心計算（不含 Python 物件操作）抽離成純 NumPy 的靜態函數。
- 用 `@numba.jit(nopython=True, cache=True)` 裝飾，第一次呼叫時 JIT 編譯，之後呼叫無額外開銷。

**限制**：Numba 無法加速含 Python 物件（如 `deque`、`dict`）的程式碼，這部分維持原樣。

**檔案**：`trading_env_kernels.py`（新增），`trading_env.py`（引用 kernel 函數）。

---

## 方案 D：多進程平行訓練（已完成）

**原理**：每個 `--seed` 是完全獨立的 Walk-Forward，可用 Python `ProcessPoolExecutor` 並行。

**啟動指令**：
```powershell
.\env\Scripts\python.exe walk_forward.py --tier promotion --cash-mode enabled --workers 6 --overwrite
```

---

## 方案 E：純 GPU 張量化環境（M4 長期計畫）

**原理**：將整個 `TaiwanStockEnv` 用 PyTorch 張量重寫，讓 `step()` 的所有計算在 GPU 內部完成，避免 CPU-GPU 資料搬運的瓶頸。

**框架選擇**：可參考 [torchrl](https://github.com/pytorch/rl) 的 `EnvBase` 或手工實作。

**預估效果**：可同時在 GPU 上跑 1,000+ 個並行環境，訓練速度從數十分鐘壓縮至數十秒。

**工程成本**：重寫整個環境（約 2–3 週工程量），待 SL 實盤穩定後再啟動。

---

## 方案 F：R8 Bug 修復 + 系統最佳化（已完成）

**更新**：2026-06-12

### F1. `_compute_pomdp_features()` 重複計算（已修復）

**問題**：每個 `step()` 中 `_compute_reward()` 和 `_get_observation()` 各呼叫一次
`_compute_pomdp_features()`，造成 `np.std` / `np.mean` 對 deque 做兩次計算。

**修復**：在 `_compute_reward()` 中於 `return_history.append(log_r)` **之後**
計算並存入 `self._pomdp_cache`；`_get_observation()` 直接讀快取。每步節省一次
O(SHARPE_WINDOW) 統計計算。

**檔案**：`trading_env.py`

### F2. `IndexedReplayBuffer._reconstruct_obs()` chunk 迴圈（已修復）

**問題**：64-chunk for 迴圈讓 batch_size=1024 需要 16 次迭代。

**修復**：完整向量化 — 一次 NumPy gather + `np.clip` 邊界保護，
速度提升 ~16×；同時修復 `done=True` 最後一步 `t+1` 越界問題
（`optimize_memory_usage=True` 時會寫入 `t+1`，現已 clamp 至 `max_steps-1`）。

**檔案**：`indexed_replay_buffer.py`

### F3. PPO `target_kl` 過鬆（已修復）

**問題**：`target_kl=0.08` 讓每次 rollout（40 次更新）policy 偏移過大。

**修復**：`target_kl=0.04`，更穩定的訓練曲線，降低 policy collapse 風險。

**檔案**：`train_portfolio.py`

### F4. SAC `gradient_steps=1` GPU 閒置（已修復）

**問題**：`train_freq=10, gradient_steps=1` 導致每 10 步環境互動只做 1 次梯度更新，
GPU 利用率極低。

**修復**：`gradient_steps=-1`（SB3 自動設等於 `train_freq=10`），GPU 利用率
大幅提升。

**檔案**：`train_portfolio.py`

### F5. Windows 虛擬記憶體不足（已新增設定腳本）

**問題**：實體 RAM 15.9 GB；6 workers × SAC buffer 4 GB = 24 GB，加上 Python
overhead 峰值約 34 GB，超出實體記憶體約 18 GB。System Managed page file 無法
保證足夠虛擬記憶體。

**修復**：新增 `scripts/set_virtual_memory.ps1`，自動計算並設定：
- InitialSize = 24 GB（預先分配，避免動態擴展開銷）
- MaximumSize = 48 GB（3× RAM 上限）
- 需以**系統管理員**身份執行，執行後需重啟。

**或** 降低每 worker 的 buffer 上限（減少 RAM 使用）：
```powershell
$env:SAC_BUFFER_RAM_GB = "2"  # 2 GB/worker × 6 = 12 GB total buffer
```

---

## 執行路線圖

```
已完成：A（Batch Size）→ D（Multi-worker）→ B（torch.compile）→ F（R8 修復）
進行中：C（Numba kernels）
未來：E（GPU Env，M4）
```

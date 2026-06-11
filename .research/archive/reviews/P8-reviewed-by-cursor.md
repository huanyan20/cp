# Cross-Review: P8 IndexedReplayBuffer — Reviewed by Cursor

> **Note**（2026-06-11）：§「後續建議」中 R7/R7b 已随 v3 **取消**。P8 保留為基礎設施。  
> **Reviewer**: Cursor IDE Agent  
> **Scope**: `main`（P8 merged + post-merge fixes）  
> **Handoff**: [`.research/handoffs/P8.json`](../handoffs/P8.json)（Antigravity @ `fe831c4`）  
> **Review date**: 2026-06-11（re-review）  
> **Verdict**: `pass`

---

## 1. Summary

P8 以 `IndexedReplayBuffer` 取代 SB3 預設 buffer：只存 `(t, account_block, action, reward, done)`，抽樣時由 `env._market_data` / `_sl_data` 重建完整 obs。  
**main 現況**已含 merge 後修正（buffer_size RAM 公式、float16、`estimated_bytes_per_transition`、向量化 `_reconstruct_obs`）。

本輪重審：**173/173 pytest 全過**（含 5 項 buffer 測試），回覆 handoff open question「完整 suite 副作用」。

---

## 2. 規格符合度（`docs/SAC_BUFFER_PLAN.md` §1.1 / §3）

| 規格項目 | 狀態 | 備註 |
|---|---|---|
| `indexed_replay_buffer.py` | ✅ | ~191 行；含 `estimated_bytes_per_transition()` |
| SAC 注入 `replay_buffer_class=IndexedReplayBuffer` | ✅ | `train_portfolio.py` L164 |
| `env` 經 `replay_buffer_kwargs` | ✅ | `handle_timeout_termination=False`（與 `optimize_memory_usage=True` 相容） |
| float16 account 儲存 | ✅ | `storage_dtype=np.float16`；sample cast float32 |
| 向量化 `_reconstruct_obs` | ✅ | fancy-index batch；`test_reconstruct_obs_vectorized_matches_loop` |
| buffer_size → 300K @ 預設 RAM | ✅ | 見 §3 |
| obs / reward / network 語意不變 | ✅ | 重建測試 rtol 1e-3 |
| PPO 路徑不受影響 | ✅ | 僅 `algo == "sac"` 分支 |
| Line B 隔離 | ✅ | SAC-R 用獨立 `DailySequenceReplayBuffer`，未改 P8 |

---

## 3. Merge 後修正（已確認落地）

### 3.1 buffer_size RAM 公式 ✅

```python
bytes_per_transition = estimated_bytes_per_transition(env, optimize_memory=True, storage_dtype=np.float16)
max_buffer_by_ram = int((ram_gb * 1024**3) / bytes_per_transition)
buffer_size = min(timesteps, max_buffer_by_ram)
```

- 預設 `SAC_BUFFER_RAM_GB=4`、`timesteps=300_000` → **buffer_size = 300,000**（生產 env ~2.4KB/transition → ~0.7GB RAM）
- `SAC_BUFFER_RAM_GB=1` 保留給 R6 重現（~2,805 cap）；註解已標 **勿用於 R7+**

### 3.2 首輪 review nit 狀態

| # | 項目 | 狀態 |
|---|------|------|
| 1 | buffer_size 公式 | ✅ 已修 |
| 2 | 未使用 import（psutil/warnings） | ✅ 已移除 |
| 3 | 完整 pytest suite | ✅ **173 passed**（本輪） |

---

## 4. 測試覆蓋

```
tests/test_indexed_replay_buffer.py — 5/5 passed
  · reconstruction（含 wrap）
  · optimize_memory on/off
  · float16 storage 逐元素
  · vectorized vs loop 重建 bitwise 一致
```

**完整 suite**：`pytest tests/ -q` → **173 passed** in ~15s（2026-06-11 本機）。

---

## 5. 程式碼審閱

### 5.1 優點

1. **`_reconstruct_obs`**：market window `[t-w, t)` 與 account block 拼接正確；向量化與 loop 參考實作一致。
2. **`add()` / `sample()`**：episode 結束時 `current_t ← window_size` 與 env 對齊；ring buffer wrap 有測試。
3. **`estimated_bytes_per_transition`**：容量規劃與 buffer 實作解耦，train 路徑可讀。
4. **SB3 相容**：實作 `_get_samples`；`optimize_memory_usage` + `handle_timeout_termination=False` 組合合法。

### 5.2 殘餘 nits（非 blocker）

| # | 嚴重度 | 說明 |
|---|--------|------|
| 1 | 低 | 未測 `enable_sl_features=True` 路徑（`_sl_data[t]` 重建） |
| 2 | 低 | 未測 `n_envs > 1`（SubprocVecEnv / 多 env 並行） |
| 3 | 低 | Buffer 持有 `env._market_data` 引用 — env 重建後 buffer 失效（SB3 慣例，文件已隱含） |
| 4 | 資訊 | 生產 obs ~96K/transition → Indexed ~0.7KB；**GPU forward 仍大**（P10 / R8 範疇，非 P8 缺陷） |

---

## 6. Handoff open questions — 回覆

| 問題 | 回覆 |
|------|------|
| 記憶體充足環境重跑完整 pytest | ✅ **173/173 passed**；無 P8 相關 regression |
| buffer_size 能否達 300K | ✅ 公式已修；預設 4GB RAM cap 下 min(timesteps, cap) = 300K |

---

## 7. 採用結論（v2 戰略）

| 決策 | 說明 |
|------|------|
| **P8 維持 merged** | 工程驗收：測試 + review ✅；**不以 ΔR6 MDD 為 merge 條件** |
| R7 WF | ~~可選~~ → **v3 已取消** |
| R7b gradient_steps | ~~可平行~~ → **v3 已砍** |
| SAC-R | 獨立線；`DailySequenceReplayBuffer` 參考 P8 模式，不回流改 P8 |

---

## 8. Verdict

```
pass
```

**Blockers**: 無  
**建議後續**（可選）：SL features 與 multi-env 各加 1 測；sample 端 profiling（plan §1.1 待做）。

---

## 9. 重現命令

```powershell
cd C:\Users\ggini\Desktop\cp
.\env\Scripts\python.exe -m pytest tests/test_indexed_replay_buffer.py -v
.\env\Scripts\python.exe -m pytest tests/ -q
```

# SAC 優勢完全釋放計畫：Replay Buffer 根本性解法評估

> **狀態**：詳細評估（2026-06-10）· 實施排程見文末
> **範圍**：`trading_env.py`、`train_portfolio.py`、`sl_pipeline/`、SB3 ReplayBuffer
> **前提**：P7 env NumPy 化已完成（`_market_data [T,N,F]` 預堆疊陣列是本計畫的基礎）
> **對應計畫**：[`../專案總覽.md`](../專案總覽.md) · [`ALGORITHM_REVIEW.md`](ALGORITHM_REVIEW.md)

---

## 0. 問題量化：SAC 的優勢目前被沒收到什麼程度

SAC 相對 PPO 的核心優勢 = off-policy 經驗重用。但 observation 巨大
（45 檔 × 2,126 維 = 95,670 float32 ≈ **374KB/筆**，`optimize_memory_usage=True`
下 obs/next_obs 已共用），buffer 容量被 RAM 上限壓死：

| 版本 | buffer | ≈ episodes（1,030 步/集） | 一筆暴跌經驗存活時間 |
|------|--------|--------------------------|---------------------|
| R6（進行中） | 2,805 | 2.7 | 2,805 步 ≈ 訓練全程的 **0.9%** |
| R7 起（4GB 上限 + 公式修正） | 11,223 | 10.9 | ≈ 3.7% |
| 理想（buffer ≥ timesteps） | 300,000 | 全歷史 | **永不淘汰** |

R6 的 SAC 實際上是「準 on-policy」：股災 transition 進 buffer 不到 3 個
episode 就被 FIFO 淘汰，後期訓練完全無法重溫熊市——這正是 MDD 控制不佳
的頭號結構性嫌疑（[`ALGORITHM_REVIEW.md`](ALGORITHM_REVIEW.md) §二）。

---

## 1. 提案逐條評估

### 提案一：時間視窗移出 Buffer + RecurrentPolicy — 目標正確，手段不可行，**改用索引式 Buffer**

**目標完全正確**：obs 中 20 天視窗是純冗餘——相鄰兩步的 obs 有 19/20 重疊，
且市場資料是**靜態資料集**，`obs(t)` 的市場部分由 t 唯一決定，根本不需要存。

**但提案的手段在本專案不可行**：

- SB3 **沒有 recurrent SAC**。`RecurrentPPO` 只存在於 sb3-contrib（本環境未安裝，
  且僅支援 PPO）。已實測確認 SB3 2.8.0 buffers 僅有
  `ReplayBuffer / DictReplayBuffer / NStepReplayBuffer`。自行實作 recurrent SAC
  + sequence sampling 是高風險研究工程（hidden state burn-in、序列抽樣正確性），
  投入產出比極差。
- 改成「env 只輸出當天截面 + GRU 記憶」**會改變 MDP 的資訊結構**，與 R6 結果
  不可比，等於開一個全新研究線。

**修正方案（P8 採用）：IndexedReplayBuffer——存索引，抽樣時重建 obs**

P7 已把市場資料堆成 `_market_data [T,N,F]`、SL 特徵堆成 `_sl_data [T,N,3]`。
obs 三段中只有 account 區塊（45×6）依賴交易狀態，必須實際儲存；其餘兩段
由 t 重建即可：

| 儲存內容 | 大小/筆 |
|----------|---------|
| t（int32） | 4B |
| account 區塊 obs/next_obs 各一份（45×6 float32 × 2） | 2,160B |
| action（46 × float32）+ reward + done | ~192B |
| **合計** | **~2.4KB**（vs 現行 374KB，**156x 縮減**） |

- `sample()` 時以 `sliding_window_view(_market_data)` + fancy indexing 重建
  batch obs，數值與現行 obs **逐位元一致**（同一份陣列、同一種切片）。
- 抽樣端成本與現行相同量級（每次 update 仍是 ~98MB 進 GPU，重建只是把
  「從 buffer 讀」換成「從共享陣列 gather」）。訓練速度不變。
- 整合零侵入：SB3 `SAC(replay_buffer_class=IndexedReplayBuffer,
  replay_buffer_kwargs=...)` 原生支援注入，不碰 SAC 內部。
- t 可由 buffer 自行追蹤（add 序列 + done 旗標即可推回 t），env 不用改。
- **結果：buffer = 全部 300K 步只需 ~0.7GB RAM**，SAC 拿回完整 off-policy
  能力，而 obs 語意、reward、網路結構全部不變 → 與 R6 嚴格可比，
  唯一變因是 replay 容量。

**判定：✅ 採用（P8，最高優先）。工作量 ~1 天（buffer ~120 行 + 等價測試 + smoke）。**

### 提案二：兩階段架構（SL 信號取代原始市場特徵）— 合理的研究線，但不是「免費優化」

對齊 [`ALGORITHM_REVIEW.md`](ALGORITHM_REVIEW.md) §四中期 #5。數字核對正確：
45 × (5 SL + 6 帳戶) = 495 維 ≈ 2KB/筆。

**真正的吸引力其實在訓練吞吐**：obs 從 95,670 維 → 495 維，
`node_embedder` 第一層從 2,126→128 縮到 11→128，batch 傳輸 98MB → 0.5MB。
在 GTX 1060 3GB 上（現為 GPU-bound，36 it/s）這可能是所有提案中
**牆鐘時間收益最大**的一項，估 3–5x。

**但有三個不可忽視的前提**：

1. **資訊瓶頸風險**：RL 只能看到 SL 敢說的話。SL 自己 Gate BLOCKED
   （MDD 38.55%），把 alpha 上限綁在 SL 上是實質假設，不是工程優化，
   必須走完整 walk-forward 對照才能採信。
2. 現行 `sl_features` 是 **3 維**（非提案說的 5 維）；補「勝率/波動/籌碼」
   等信號是 `sl_pipeline` 的延伸工作。
3. 需要新增 env `obs_mode="sl_only"`（現行 `enable_sl_features` 是
   **附加**而非取代）——P7 之後實作成本低。

**判定：⬜ 排入 R8 候選實驗（在 P8 之後、與 base obs 並排比較）。
若 P8 解決容量後 MDD 仍超標，這是下一個最有力的桿子。**

### 提案三：工程級壓縮 — 一半已完成，一半被 P8 取代

| 項 | 評估 |
|----|------|
| 3-1 `optimize_memory_usage=True` | **早已啟用**（N1 起就在 `train_portfolio.py`）。提案說「開啟後 2,800→5,600」是誤判——真相是舊的容量公式把 obs+next_obs 重複計算了一倍，這已於 2026-06-10 修正（連同上限 2GB→4GB，得 11,223）。**此項無事可做。** |
| 3-2 float16 儲存 | 技術上可行（特徵已標準化），但只能再 ×2（11K→22K），距全歷史 300K 仍差一個數量級；且 SB3 buffer dtype 跟著 observation_space 走，要嘛改 space dtype（影響網路輸入路徑）要嘛客製 buffer——**既然都要客製 buffer，不如直接做 P8 索引式（×156）**。判定：❌ 被 P8 取代。 |

### 提案四：PER（優先經驗回放）— 假說合理，排條件式 backlog

「讓 SAC 反覆重溫暴跌日」的方向與 MDD 痛點對得上，且機制成立：
暴跌日吃 `LAMBDA_DRAWDOWN` 大懲罰 → TD-error 大 → 被優先抽樣。

但三個現實約束：

1. **SB3 2.8 核心與本環境都沒有 PER**；且 PER 不只是換 buffer——SAC 的
   `train()` 必須加 importance-sampling 權重與 priority 回寫，等於要
   subclass SAC。工作量中高（2–3 天），且動到演算法核心、需自驗正確性。
2. **P8 已先吃掉 PER 一半的紅利**：全歷史 buffer 下暴跌經驗永不淘汰，
   uniform sampling 也抽得到（現行 R6 是「根本抽不到」，那才是主要矛盾）。
   PER 的增量價值只剩「加權重溫」。
3. reward clip ±1 壓縮了 TD-error 的動態範圍，優先級訊號會被鈍化，
   實際效果存疑、需實驗證明。

**判定：⬜ 條件式 backlog——僅在 R7（P8 重訓）之後 worst-case MDD 仍 >35%
時實作。先讓「抽得到」發揮作用，再決定要不要「加權抽」。**

---

## 2. 事實核對（提案 vs 程式碼/SB3 實況）

| 提案敘述 | 實況 |
|----------|------|
| 「每筆 transition 765KB」 | 374KB（`optimize_memory_usage=True` 下 obs/next_obs 共用，提案用了 ×2 的舊公式） |
| 「啟用 optimize_memory_usage 可翻倍」 | 一直都是啟用狀態；翻倍來自容量公式修正（2026-06-10 已完成） |
| 「SB3 用 RecurrentPolicy」 | SB3/sb3-contrib 皆無 recurrent **SAC**（只有 RecurrentPPO，且未安裝） |
| 「SL 信號 5 維」 | 現行 3 維；5 維需擴充 `sl_pipeline` |
| 「視窗移出 buffer 容量 ×20」 | 索引式重建可達 ×156，且不需要改網路架構 |

---

## 3. 執行排程（全部在 R6 跑完後生效，嚴禁中途插入）

| 編號 | 內容 | 工作量 | 風險 | 條件 |
|------|------|--------|------|------|
| **P8** | `IndexedReplayBuffer`（存 t + account 區塊，抽樣時由 `_market_data`/`_sl_data` 重建）+ 逐位元等價測試 + smoke | ~1 天 | 低（不改 obs 語意/網路/reward） | R6 完成後 |
| **R7** | SAC promotion tier 重訓，唯一變因 = buffer 容量（2,805 → 300K 全歷史），直接檢驗 off-policy 完整化對 MDD/Sortino 的效果 | ~1 天 GPU | 低 | P8 完成 |
| **R8** | 兩階段架構實驗：`obs_mode="sl_only"` + `sl_pipeline` 信號擴 3→5 維，與 base obs 並排 walk-forward | 3–5 天 | 中（資訊瓶頸） | R7 結果出爐後決定 |
| **R9** | PER（custom buffer + SAC.train IS 權重） | 2–3 天 | 中高 | 僅當 R7 後 MDD 仍 >35% |
| **P10** | PPO 訓練效率 ablation（VecEnv + n_steps/n_epochs，見 §4） | ~0.5 天 | 低 | R6 完成後；與 P8 可平行 |
| ❌ | recurrent SAC（SB3 不支援）、float16 buffer（被 P8 取代） | — | — | 否決 |

**R7 的判讀準則**：若全歷史 buffer 使 worst-case MDD 明顯下降（向 35% 收斂），
證明「容量」是主因，續推 R8 放大優勢；若幾乎無感，則問題在 reward 設計
而非記憶，應回頭走 [`ALGORITHM_REVIEW.md`](ALGORITHM_REVIEW.md) §四的
CVaR constraint 路線，PER 同步降級。

**可選微調（R7 時一併考慮，但一次只動一個變因）**：buffer 變大後
`gradient_steps=1 / train_freq=10`（全程僅 30K 次更新）偏保守，可另開
ablation 測 `gradient_steps=2–4` 提高樣本利用率。

---

## 4. P10 — PPO 訓練效率 Ablation（R6 完成後實施，現階段僅計畫）

> **觸發條件**：R6 promotion 跑完、`experiment_report.py` 產出後。  
> **目的**：在**不換算法、不改 obs/reward** 的前提下，量測 PPO 吞吐能否接近 2–4x，
> 供後續 PPO 臂（含 R6 對照後的新實驗）縮短牆鐘時間。  
> **實作入口（R6 後新增）**：`scripts/ppo_efficiency_ablation.py`

### 4.1 現況基線（R6 PPO/disabled）

| 參數 | 現值 | 效率含義 |
|------|------|----------|
| 環境 | `DummyVecEnv`（1 env） | rollout 循序收集，GPU 空等 |
| `n_steps` | 256 | 每輪僅 256 步即觸發更新，同步開銷高 |
| `n_epochs` | 10 | 同一份 rollout 重訓 10 遍（~40 次 backward/輪） |
| `batch_size` | 64 | — |
| 牆鐘 | ~2h / 300K / GTX 1060 | 與 SAC 同級，瓶頸在 GNN forward |

PPO **沒有** SAC 的 replay buffer 問題；加速槓桿是 **VecEnv 平行 rollout** 與 **減少重複 epoch**。

### 4.2 實驗設計（一次只動一個變因）

固定：`algo=ppo`、`cash=disabled`、`seed=42`、`timesteps=300_000`、
`train_portfolio.py` 現行 lr/clip/target_kl、**單期 2025H1**（熊市壓力期，與 R4 驗證對齊）。

| 階段 | 變因 | 設定 | 量測 |
|------|------|------|------|
| **A0** | 基線 | 現行 hyperparams + DummyVecEnv | fps、分鐘數、Return/MDD/Sortino |
| **A1** | VecEnv | `SubprocVecEnv`, `n_envs=4`（VRAM 不足則 2） | 同上 |
| **A2** | n_epochs | A1 + `n_epochs=5` | 同上 |
| **A3** | n_steps | A2 + `n_steps=512` | 同上 |

**不納入本輪**：改 obs 維度（留 R8）、RecurrentPPO、改 reward/env（破壞與 R6 可比性）。

### 4.3 驗收標準

1. **吞吐**：A1 相對 A0 fps ≥ **1.5x**；A3 相對 A0 總訓練時間 ≤ **50%** 為採用門檻。
2. **品質**：OOS（2025H1 test）Return/MDD/Sortino 與 A0 差異在 R6 seed 分散範圍內（|ΔMDD| < 5pp 且 Sortino 符號不反轉）——效率換來品質崩潰則回退。
3. **產出**：`results_dir/ppo_efficiency_ablation.json` 記錄各階段 fps、elapsed_s、metrics；採用組合寫入 `train_portfolio.py` 的 PPO 預設（或 `settings.research` 可配置項）。

### 4.4 R6 後實作清單（屆時才改程式）

1. 新增 `scripts/ppo_efficiency_ablation.py`（呼叫 `build_model` + 單期 `research_pipeline` 訓練/評估）。
2. `train_portfolio.py`：`build_model(..., n_envs=1)` 可選注入 `SubprocVecEnv`；PPO 區塊支援 `n_steps`/`n_epochs` 覆寫（預設維持 R6 值直至 ablation 通過）。
3. `settings.py`（可選）：`ppo_n_envs`、`ppo_n_steps`、`ppo_n_epochs` 環境變數。
4. 測試：smoke 確認 VecEnv 在 Windows `spawn` 下可跑通 1K steps。

### 4.5 預期指令（R6 後）

```powershell
# 基線 vs 最佳組合（腳本一次跑 A0–A3 並寫 JSON）
.\env\Scripts\python.exe scripts\ppo_efficiency_ablation.py --period 2025H1 --seed 42

# 採用後單期驗證
.\env\Scripts\python.exe scripts\validate_period.py --period 2025H1 --algo ppo --cash disabled --timesteps 300000
```

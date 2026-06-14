> Superseded by 2026-06-14 SL-first strategy.
> This document is retained as historical context only. It must not be treated as an active implementation queue unless explicitly updated after 2026-06-14.
# Turnover Dampening & Reward Enhancement Walkthrough

本次更新成功地將「懲罰過大」的問題根除，讓模型的 Reward 曲線（梯度）重新恢復生機。以下是本次完成的主要機制與成果：

## 1. Action Deadband (動作死區)
- **實作位置**：`trading_env.py`
- **邏輯**：當神經網路輸出的目標權重變化小於 `ACTION_DEADBAND = 0.02` (2%) 時，強制不執行操作。
- **效果**：這如同為操作加上一個硬體級別的低通濾波器，濾掉了 PPO 常見的無意義微調震盪。

## 2. Piecewise Convex Turnover Penalty (分段非線性換手懲罰)
- **實作位置**：`trading_env_kernels.py` & `env_core/reward_calculator.py`
- **邏輯**：
  - `Turnover <= 30%`：套用極度平滑的 `2.0 * (Turnover)^3`。
  - `Turnover > 30%`：改用斜率為 0.54 的切線**線性外推**。
- **效果**：在 0~30% 區間提供優雅的凸函數懲罰（微幅換手幾乎無感，大量換手明顯痛感），同時避免了單日 100% 大爆發時的數值崩壞。

## 3. Holding Period Whipsaw Penalty (雙巴衰減懲罰)
- **實作位置**：`trading_env.py` (計算), `reward_calculator.py` (套用)
- **邏輯**：使用 `whipsaw_penalty += 賣出比例 * max(0.0, 3.0 - 持有天數)`。
- **效果**：當日買當日賣會承受最大懲罰，持有超過 3 天則不再受此特別懲罰，完美模擬短線進出的額外滑價與風險。

## 4. Reward Clipping Relaxation (放寬環境 Reward)
- **實作位置**：`trading_env_kernels.py` & `env_core/reward_calculator.py`
- **邏輯**：將最終的環境回報從 `np.clip(..., -1.0, 1.0)` 改為 `max(-5.0, min(5.0, raw))`。
- **效果**：解決了模型初期因為懲罰值超越 -1.0，導致每一步都拿到死死 `-1.0` 的**梯度消失**問題。

## 5. TensorBoard Logging Integration
- **實作位置**：`core/model_trainer.py`
- **邏輯**：為 `PPO` 與 `SAC` 的初始化參數加入了 `tensorboard_log="logs/tb_logs"`。
- **效果**：現在可以直接使用 `tensorboard --logdir logs/tb_logs` 指令來繪製與監控實驗過程。

---

> [!TIP]
> **Smoke Test 結果分析**
> 經過 50K steps 的快速煙霧測試，我們觀察到 `ep_rew_mean` 穩定落在 `-2300` 左右（平均每步 `-2.9` 分）。
> 這證實了 **Reward 的封印已經解除**，模型確實在感受超額隨機交易與虧損帶來的痛覺！後續只需啟動百萬步級別的正式訓練（讓 PPO 的 `entropy_loss` 下降並收斂變異數），即可期待一條漂亮往上爬升的 Reward 學習曲線。

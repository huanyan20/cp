"""
gnn_extractor.py — 時空雙重注意力 GNN 特徵提取器 (v2)
════════════════════════════════════════════════════════
升級摘要（相較 v1）：

  A. 時間軸 (Temporal)：
     每檔股票在進入 Graph Attention 前，先通過輕量級 LSTM，
     讓節點攜帶的資訊是「波段趨勢向量」而非壓扁的原始數字。

  B. 空間軸 (Spatial)：
     Multi-Head Self-Attention 現在接收的是含時序語意的節點嵌入，
     股票間交換的資訊品質大幅提升。

  C. 注意力矩陣可視化：
     forward() 可選擇性回傳 attn_weights (shape: batch × N × N)，
     用於繪製「AI 市場關聯性熱力圖」。

架構流程：
  Input  : (batch, num_stocks, window × features_per_step)
              ↓ reshape
  Reshape: (batch × num_stocks, window, features_per_step)
              ↓ TemporalEncoder (LSTM)
  LSTM   : 取最後一個 hidden state → (batch × num_stocks, lstm_hidden)
              ↓ reshape back
  Nodes  : (batch, num_stocks, lstm_hidden)
              ↓ NodeEmbedder (Linear)
  Embed  : (batch, num_stocks, embed_dim)
              ↓ Multi-Head Self-Attention
  Attend : (batch, num_stocks, embed_dim)  +  attn_weights (batch, N, N)
              ↓ Residual + LayerNorm
  Output : flatten → Linear → ReLU → (batch, features_dim)
"""

import gymnasium as gym
import torch
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

# ════════════════════════════════════════════════════════════════════
# A. 時間序列編碼器：輕量 LSTM，提取每支股票的波段趨勢特徵
# ════════════════════════════════════════════════════════════════════

class TemporalEncoder(nn.Module):
    """
    對單支股票的 window 時序做 LSTM 編碼。
    實際呼叫時會把 batch × num_stocks 維度合併，達到「所有股票共享同一個 LSTM 權重」
    但各自獨立跑時序，不互相污染。

    參數：
      input_size  : 每個時間步的特徵數 (features_per_step)
      hidden_size : LSTM 隱藏層維度（建議 64，保持輕量）
      num_layers  : LSTM 層數（建議 1~2，避免梯度問題）
      dropout     : 僅在 num_layers > 1 時生效
    """
    def __init__(self, input_size: int, hidden_size: int = 64,
                 num_layers: int = 1, dropout: float = 0.1):
        super().__init__()
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x shape: (batch_all_stocks, window, input_size)
              其中 batch_all_stocks = original_batch × num_stocks

        回傳: (batch_all_stocks, hidden_size) — 取最後一步的 hidden state
        """
        _, (h_n, _) = self.lstm(x)   # h_n: (num_layers, batch_all, hidden)
        return h_n[-1]               # 只取最後一層: (batch_all, hidden_size)


# ════════════════════════════════════════════════════════════════════
# B. 主提取器：時空雙重注意力
# ════════════════════════════════════════════════════════════════════

class GnnFeatureExtractor(BaseFeaturesExtractor):
    """
    Spatio-Temporal GNN 特徵提取器 v2

    Observation Space 形狀假設：
      shape == (num_stocks, window_size * features_per_step)
      ─ 環境端先把 (window, features) flatten 成一個向量後才存入 obs space。
      ─ 本類別會在內部 reshape 回 (window, features_per_step) 再做 LSTM。

    若環境直接輸出 (num_stocks, features_per_stock) 的二維 obs（v1 格式），
    可將 window_size=1 使用，此時 LSTM 退化為單步線性映射，向後相容。

    參數：
      features_dim      : 輸出給 PPO 策略網路的向量維度（預設 256）
      window_size       : 時間窗口長度（必須與 TaiwanStockEnv 一致）
      lstm_hidden       : LSTM 隱藏層維度
      embed_dim         : Graph Attention 的節點嵌入維度
      num_heads         : Multi-Head Attention 頭數（須能整除 embed_dim）
      attn_dropout      : Attention dropout（訓練時抑制過擬合）
    """

    def __init__(
        self,
        observation_space: gym.spaces.Box,
        features_dim: int = 256,
        window_size: int = 20,
        lstm_hidden: int = 64,
        embed_dim: int = 64,
        num_heads: int = 4,
        attn_dropout: float = 0.1,
    ):
        super().__init__(observation_space, features_dim)

        assert len(observation_space.shape) == 2, (
            f"GnnFeatureExtractor 需要 2D Observation Space "
            f"(num_stocks, window×features)，但收到 shape={observation_space.shape}"
        )

        self.num_stocks        = observation_space.shape[0]
        self.flattened_features = observation_space.shape[1]   # window × features_per_step
        self.window_size       = window_size

        # 每個時間步的特徵數（還原 flatten 之前）
        assert self.flattened_features % window_size == 0, (
            f"flattened_features ({self.flattened_features}) 無法被 "
            f"window_size ({window_size}) 整除，請確認 obs shape。"
        )
        self.features_per_step = self.flattened_features // window_size

        # ── A. 時間序列編碼器（共享，所有股票共用同一組 LSTM 權重）───────
        self.temporal_encoder = TemporalEncoder(
            input_size=self.features_per_step,
            hidden_size=lstm_hidden,
            num_layers=1,
            dropout=0.0,   # 單層 LSTM 不使用 dropout
        )

        # ── B-1. 節點嵌入：LSTM 輸出 → Graph Attention 輸入維度 ──────────
        self.node_embedder = nn.Sequential(
            nn.Linear(lstm_hidden, embed_dim),
            nn.ReLU(),
        )

        # ── B-2. Multi-Head Self-Attention（空間軸）────────────────────────
        # need_weights=True 預設開啟；實際是否回傳由 forward() 的 return_attn 控制
        self.attention = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            batch_first=True,
            dropout=attn_dropout,
        )
        self.norm = nn.LayerNorm(embed_dim)

        # ── C. 輸出整合層 ─────────────────────────────────────────────────
        self.output_net = nn.Sequential(
            nn.Linear(self.num_stocks * embed_dim, features_dim),
            nn.ReLU(),
        )

        # 供 evaluate_portfolio.py 在推論時擷取最新 attn_weights
        self._last_attn_weights: torch.Tensor | None = None

    # ────────────────────────────────────────────────────────────────
    # forward：SB3 PPO 主訓練路徑呼叫此函式
    # ────────────────────────────────────────────────────────────────
    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        """
        observations shape: (batch_size, num_stocks, window×features_per_step)
        回傳            : (batch_size, features_dim)
        """
        batch_size = observations.shape[0]

        # ── Step 1：reshape for LSTM ────────────────────────────────
        # (batch, num_stocks, window×feat) → (batch×num_stocks, window, feat)
        x = observations.view(
            batch_size * self.num_stocks,
            self.window_size,
            self.features_per_step,
        )

        # ── Step 2：時間序列編碼（每股獨立，LSTM 共享權重）────────────
        # → (batch×num_stocks, lstm_hidden)
        temporal_feat = self.temporal_encoder(x)

        # reshape 回 (batch, num_stocks, lstm_hidden)
        temporal_feat = temporal_feat.view(batch_size, self.num_stocks, -1)

        # ── Step 3：節點嵌入 → (batch, num_stocks, embed_dim) ────────
        x_embed = self.node_embedder(temporal_feat)

        # ── Step 4：Multi-Head Self-Attention（空間 GNN）───────────────
        # attn_weights: (batch, num_stocks, num_stocks)
        attn_out, attn_weights = self.attention(
            x_embed, x_embed, x_embed,
            need_weights=True,
            average_attn_weights=True,   # 對所有 head 取平均 → 可解釋性更強
        )

        # 快取最新 attn_weights，供外部視覺化使用（不影響梯度）
        self._last_attn_weights = attn_weights.detach()

        # ── Step 5：殘差 + LayerNorm ─────────────────────────────────
        x_out = self.norm(x_embed + attn_out)

        # ── Step 6：攤平 → 輸出 ──────────────────────────────────────
        x_flat = x_out.view(batch_size, -1)
        return self.output_net(x_flat)

    # ────────────────────────────────────────────────────────────────
    # 公開工具：取得最後一次 forward 的注意力矩陣（用於視覺化）
    # ────────────────────────────────────────────────────────────────
    def get_last_attn_weights(self) -> torch.Tensor | None:
        """
        回傳最近一次 forward() 計算的注意力矩陣。
        shape: (batch, num_stocks, num_stocks) 或 None（尚未執行過 forward）

        典型用法（評估階段）：
            extractor = model.policy.features_extractor
            weights   = extractor.get_last_attn_weights()
            # weights[0] → 第一個樣本的 N×N 關聯矩陣
        """
        return self._last_attn_weights

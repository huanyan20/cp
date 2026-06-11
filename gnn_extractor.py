"""
gnn_extractor.py — 純 PyTorch Self-Attention GNN 特徵提取器
將 N 檔股票視為圖中的節點 (Node)，透過 Multi-Head Attention 讓股票間交換資訊。
"""

import gymnasium as gym
import torch
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class GnnFeatureExtractor(BaseFeaturesExtractor):
    """
    Self-Attention 圖特徵提取器 (模擬全連接 Graph Attention Network)

    架構：
      1. Node Embedder   : 每檔股票獨立壓縮 (Linear → Softsign → Linear → Softsign)
      2. Graph Attention : Multi-Head Self-Attention，讓所有節點互相交換資訊
      3. Residual        : 殘差連接，防止梯度消失
      4. Output Linear   : 攤平後映射到 PPO 政策網路的輸入維度 (Linear → Softsign)

    v8.1 激活函數改為 Softsign（取代 ReLU）：
      - 輸出有界 (-1, 1)：防止特徵層梯度爆炸
      - 梯度永不為零：消除 Dead Neuron 問題
      - 適合長訓練中政策網路的穩定性需求
    """

    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 256):
        super().__init__(observation_space, features_dim)

        # observation_space.shape == (num_stocks, features_per_stock)
        assert len(observation_space.shape) == 2, (
            f"GnnFeatureExtractor 需要 2D Observation Space，"
            f"但收到 shape={observation_space.shape}"
        )
        self.num_stocks = observation_space.shape[0]
        self.features_per_stock = observation_space.shape[1]

        embed_dim = 64
        self.input_norm = nn.LayerNorm(self.features_per_stock)

        # 1. 節點嵌入：每檔股票獨立提取特徵
        self.node_embedder = nn.Sequential(
            nn.Linear(self.features_per_stock, 128),
            nn.Softsign(),
            nn.Linear(128, embed_dim),
            nn.Softsign(),
        )

        # 2. Self-Attention (模擬全連接 GNN 邊) ─ batch_first=True
        self.attention = nn.MultiheadAttention(
            embed_dim=embed_dim, num_heads=4, batch_first=True, dropout=0.0
        )
        self.norm = nn.LayerNorm(embed_dim)

        # 3. 輸出整合層
        self.output_net = nn.Sequential(
            nn.Linear(self.num_stocks * embed_dim, features_dim), nn.Softsign()
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        # observations shape: (batch_size, num_stocks, features_per_stock)
        batch_size = observations.shape[0]
        x = observations.view(batch_size, self.num_stocks, self.features_per_stock)
        x = self.input_norm(x)

        # 節點嵌入 → (batch, num_stocks, embed_dim)
        x_embed = self.node_embedder(x)

        # Self-Attention 讓每一節點收集所有其他節點的資訊
        attn_out, _ = self.attention(x_embed, x_embed, x_embed)

        # 殘差連接 + Layer Norm
        x_out = self.norm(x_embed + attn_out)

        # 攤平後輸出給 PPO 決策層
        x_flat = x_out.view(batch_size, -1)
        return self.output_net(x_flat)


class TemporalGnnFeatureExtractor(BaseFeaturesExtractor):
    """GRU-over-window encoder followed by stock-level self-attention.

    This extractor is opt-in for experiments that want the policy to observe
    whether macro/ADR stress is rising or fading across the recent window.
    The legacy ``GnnFeatureExtractor`` remains the default.
    """

    def __init__(
        self,
        observation_space: gym.spaces.Box,
        features_dim: int = 256,
        window_size: int = 20,
        account_features: int = 9,
    ):
        super().__init__(observation_space, features_dim)
        assert len(observation_space.shape) == 2, (
            f"TemporalGnnFeatureExtractor 需要 2D Observation Space，"
            f"但收到 shape={observation_space.shape}"
        )
        self.num_stocks = observation_space.shape[0]
        self.features_per_stock = observation_space.shape[1]
        self.window_size = window_size
        self.account_features = account_features

        sequence_features = self.features_per_stock - self.account_features
        if sequence_features <= 0 or sequence_features % self.window_size != 0:
            raise ValueError(
                "features_per_stock must equal window_size * market_features + account_features"
            )
        self.market_features = sequence_features // self.window_size

        embed_dim = 64
        hidden_dim = 64
        self.input_norm = nn.LayerNorm(self.market_features)
        self.account_norm = nn.LayerNorm(self.account_features)
        self.temporal_encoder = nn.GRU(
            input_size=self.market_features,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True,
        )
        self.node_embedder = nn.Sequential(
            nn.Linear(hidden_dim + self.account_features, embed_dim),
            nn.Softsign(),
        )
        self.attention = nn.MultiheadAttention(
            embed_dim=embed_dim, num_heads=4, batch_first=True, dropout=0.0
        )
        self.norm = nn.LayerNorm(embed_dim)
        self.output_net = nn.Sequential(
            nn.Linear(self.num_stocks * embed_dim, features_dim),
            nn.Softsign(),
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        batch_size = observations.shape[0]
        x = observations.view(batch_size, self.num_stocks, self.features_per_stock)

        seq_flat = x[:, :, : self.window_size * self.market_features]
        account = x[:, :, -self.account_features :]
        seq = seq_flat.view(
            batch_size * self.num_stocks,
            self.window_size,
            self.market_features,
        )
        seq = self.input_norm(seq)
        _, hidden = self.temporal_encoder(seq)
        hidden = hidden[-1].view(batch_size, self.num_stocks, -1)
        account = self.account_norm(account)

        x_embed = self.node_embedder(torch.cat([hidden, account], dim=-1))
        attn_out, _ = self.attention(x_embed, x_embed, x_embed)
        x_out = self.norm(x_embed + attn_out)
        return self.output_net(x_out.reshape(batch_size, -1))

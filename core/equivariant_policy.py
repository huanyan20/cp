import torch
import torch.nn as nn
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.distributions import Distribution

class EquivariantNetwork(nn.Module):
    """
    Processes the flattened GNN features in an equivariant manner.
    Assumes features_dim = num_stocks * embed_dim.
    """
    def __init__(self, num_stocks: int, embed_dim: int, enable_cash_action: bool):
        super().__init__()
        self.num_stocks = num_stocks
        self.embed_dim = embed_dim
        self.enable_cash_action = enable_cash_action

        # Shared layer for each stock
        self.stock_net = nn.Sequential(
            nn.Linear(embed_dim, 64),
            nn.Softsign(),
            nn.Linear(64, 1)
        )

        if self.enable_cash_action:
            # Global pooling for cash action
            self.cash_net = nn.Sequential(
                nn.Linear(embed_dim, 64),
                nn.Softsign(),
                nn.Linear(64, 1)
            )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        batch_size = features.shape[0]
        # Reshape to (batch, num_stocks, embed_dim)
        x = features.view(batch_size, self.num_stocks, self.embed_dim)
        
        # Apply shared network across all stocks
        # (batch, num_stocks, 1) -> (batch, num_stocks)
        stock_actions = self.stock_net(x).squeeze(-1)

        if self.enable_cash_action:
            # Global average pooling
            x_pool = x.mean(dim=1)
            cash_action = self.cash_net(x_pool)
            return torch.cat([stock_actions, cash_action], dim=-1)
        
        return stock_actions


class EquivariantActorCriticPolicy(ActorCriticPolicy):
    """
    Custom SB3 Policy that preserves Permutation Equivariance for multi-asset RL.
    It expects the feature extractor to output num_stocks * embed_dim.
    """
    def __init__(self, *args, num_stocks: int = 45, embed_dim: int = 256, enable_cash_action: bool = False, **kwargs):
        self.num_stocks = num_stocks
        self.embed_dim = embed_dim
        self.enable_cash_action = enable_cash_action
        super().__init__(*args, **kwargs)

    def _build(self, lr_schedule) -> None:
        """
        Override the default build method to replace the action_net and value_net.
        """
        super()._build(lr_schedule)
        
        # Replace the default action_net (which is a dense layer) with our Equivariant network
        self.action_net = EquivariantNetwork(self.num_stocks, self.embed_dim, self.enable_cash_action)
        
        # Value net can be standard MLP, but we use a custom one over pooled features for efficiency
        # We can just keep the default value_net which connects to the flattened features.
        # But for parameter efficiency, we can build a pooled value net.
        # We'll just rely on the standard value net for now.

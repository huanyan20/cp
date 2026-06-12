import numpy as np

# Account feature indices (copied from trading_env.py)
NUM_ACCOUNT_FEATURES = 9
IDX_CASH = 0
IDX_TOTAL_RETURN = 1
IDX_MAX_DRAWDOWN = 2
IDX_POSITION = 3
IDX_TRADE_RETURN = 4
IDX_HOLDING_PERIOD = 5
IDX_ROLLING_VOL = 6
IDX_ROLLING_SORTINO = 7
IDX_CURRENT_DRAWDOWN = 8

class ObservationBuilder:
    def __init__(
        self,
        num_stocks: int,
        window_size: int,
        num_market_features: int,
        enable_cash_action: bool = False,
        enable_margin_short: bool = False,
        enable_sl_features: bool = False,
    ):
        self.num_stocks = num_stocks
        self.window_size = window_size
        self.num_market_features = num_market_features
        self.enable_cash_action = enable_cash_action
        self.enable_margin_short = enable_margin_short
        self.enable_sl_features = enable_sl_features

        self.market_dim = self.window_size * self.num_market_features
        self.obs_dim_per_stock = self.market_dim + NUM_ACCOUNT_FEATURES

    def build(
        self,
        current_step: int,
        positions: np.ndarray,
        cash_weight: float,
        portfolio_value: float,
        initial_balance: float,
        max_drawdown: float,
        trade_returns: np.ndarray,
        holding_periods: np.ndarray,
        pomdp_features: tuple[float, float, float],
        market_data: np.ndarray,
        sl_data: np.ndarray | None = None,
    ) -> np.ndarray:
        """
        Builds the observation matrix [num_stocks, obs_dim_per_stock]
        """
        start = current_step - self.window_size
        stock_exposure = float(np.sum(np.abs(positions)))
        
        if self.enable_margin_short or self.enable_cash_action:
            cash_ratio = float(cash_weight)
        else:
            cash_ratio = max(0.0, 1.0 - stock_exposure)

        total_ret = (portfolio_value - initial_balance) / max(initial_balance, 1e-8)
        dd_norm = float(np.clip(max_drawdown, 0.0, 1.0))
        
        rolling_vol, sortino_proxy, current_dd = pomdp_features

        # Pre-allocate observation matrix
        obs = np.empty((self.num_stocks, self.obs_dim_per_stock), dtype=np.float32)
        
        # [window, num_stocks, F] -> [num_stocks, window * F]
        obs[:, :self.market_dim] = (
            market_data[start : current_step]
            .transpose(1, 0, 2)
            .reshape(self.num_stocks, self.market_dim)
        )

        account = obs[:, self.market_dim : self.market_dim + NUM_ACCOUNT_FEATURES]
        account[:, IDX_CASH] = cash_ratio
        account[:, IDX_TOTAL_RETURN] = np.clip(total_ret, -1.0, 1.0)
        account[:, IDX_MAX_DRAWDOWN] = dd_norm
        account[:, IDX_POSITION] = positions
        account[:, IDX_TRADE_RETURN] = np.clip(trade_returns, -1.0, 1.0)
        account[:, IDX_HOLDING_PERIOD] = np.clip(holding_periods / 100.0, 0.0, 1.0)
        account[:, IDX_ROLLING_VOL] = rolling_vol
        account[:, IDX_ROLLING_SORTINO] = sortino_proxy
        account[:, IDX_CURRENT_DRAWDOWN] = current_dd

        if self.enable_sl_features and sl_data is not None:
            # We assume the caller concatenated SL data feature dim at the end
            obs[:, self.market_dim + NUM_ACCOUNT_FEATURES :] = sl_data[current_step]
            
        return obs

import numpy as np
import torch as th
from gymnasium import spaces
from typing import Any, Dict, List, Optional, Union

from stable_baselines3.common.buffers import BaseBuffer
from stable_baselines3.common.vec_env import VecNormalize
from stable_baselines3.common.type_aliases import ReplayBufferSamples


class IndexedReplayBuffer(BaseBuffer):
    """
    Memory-efficient ReplayBuffer for TaiwanStockEnv.
    Stores time index (t) + account block; market window rebuilt on sample().
    Account values default to float16 (features are normalized); network still sees float32.
    """

    def __init__(
        self,
        buffer_size: int,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        device: Union[th.device, str] = "auto",
        n_envs: int = 1,
        optimize_memory_usage: bool = False,
        handle_timeout_termination: bool = True,
        env: Any = None,
        storage_dtype: np.dtype = np.float16,
    ):
        super().__init__(buffer_size, observation_space, action_space, device, n_envs=n_envs)

        self.buffer_size = max(buffer_size // n_envs, 1)

        if optimize_memory_usage and handle_timeout_termination:
            raise ValueError(
                "ReplayBuffer does not support optimize_memory_usage = True "
                "and handle_timeout_termination = True simultaneously."
            )
        self.optimize_memory_usage = optimize_memory_usage
        self.handle_timeout_termination = handle_timeout_termination
        self.storage_dtype = np.dtype(storage_dtype)

        if env is None:
            raise ValueError("IndexedReplayBuffer requires `env` to be passed in replay_buffer_kwargs.")

        unwrapped_env = env.envs[0] if hasattr(env, "envs") else env
        self.window_size = unwrapped_env.window_size
        self.num_stocks = unwrapped_env.num_stocks
        self.num_market_features = unwrapped_env.num_market_features
        self.market_dim = self.window_size * self.num_market_features
        self.enable_sl_features = unwrapped_env.enable_sl_features
        self._NUM_ACCOUNT_FEATURES = unwrapped_env._NUM_ACCOUNT_FEATURES
        self.obs_dim_per_stock = self.market_dim + self._NUM_ACCOUNT_FEATURES + (3 if self.enable_sl_features else 0)

        self.market_data = unwrapped_env._market_data
        self.sl_data = unwrapped_env._sl_data if self.enable_sl_features else None

        self.t_memory = np.zeros((self.buffer_size, self.n_envs), dtype=np.int32)

        self.account_observations = np.zeros(
            (self.buffer_size, self.n_envs, self.num_stocks, self._NUM_ACCOUNT_FEATURES),
            dtype=self.storage_dtype,
        )

        if not optimize_memory_usage:
            self.account_next_observations = np.zeros(
                (self.buffer_size, self.n_envs, self.num_stocks, self._NUM_ACCOUNT_FEATURES),
                dtype=self.storage_dtype,
            )

        self.actions = np.zeros(
            (self.buffer_size, self.n_envs, self.action_dim), dtype=self._maybe_cast_dtype(action_space.dtype)
        )

        self.rewards = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.dones = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.timeouts = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)

        self.current_t = np.full((self.n_envs,), self.window_size, dtype=np.int32)

    def add(
        self,
        obs: np.ndarray,
        next_obs: np.ndarray,
        action: np.ndarray,
        reward: np.ndarray,
        done: np.ndarray,
        infos: List[Dict[str, Any]],
    ) -> None:
        action = action.reshape((self.n_envs, self.action_dim))

        acc_obs = obs[:, :, self.market_dim : self.market_dim + self._NUM_ACCOUNT_FEATURES].astype(
            self.storage_dtype, copy=False
        )
        acc_next_obs = next_obs[:, :, self.market_dim : self.market_dim + self._NUM_ACCOUNT_FEATURES].astype(
            self.storage_dtype, copy=False
        )

        self.t_memory[self.pos] = self.current_t
        self.account_observations[self.pos] = acc_obs

        if self.optimize_memory_usage:
            # Clamp t+1 to prevent out-of-bounds on the final episode step
            # (when done=True, current_t is at max_steps-1 and t+1 would overflow).
            next_t = np.minimum(self.current_t + 1, self.market_data.shape[0] - 1)
            self.account_observations[(self.pos + 1) % self.buffer_size] = acc_next_obs
            self.t_memory[(self.pos + 1) % self.buffer_size] = next_t
        else:
            self.account_next_observations[self.pos] = acc_next_obs

        self.actions[self.pos] = np.array(action)
        self.rewards[self.pos] = np.array(reward)
        self.dones[self.pos] = np.array(done)

        if self.handle_timeout_termination:
            self.timeouts[self.pos] = np.array([info.get("TimeLimit.truncated", False) for info in infos])

        self.pos += 1
        if self.pos == self.buffer_size:
            self.full = True
            self.pos = 0

        self.current_t += 1
        for i in range(self.n_envs):
            if done[i]:
                self.current_t[i] = self.window_size

    def sample(self, batch_size: int, env: Optional[VecNormalize] = None) -> ReplayBufferSamples:
        if self.full:
            batch_inds = (np.random.randint(1, self.buffer_size, size=batch_size) + self.pos) % self.buffer_size
        else:
            batch_inds = np.random.randint(0, self.pos, size=batch_size)
        return self._get_samples(batch_inds, env=env)

    def _reconstruct_obs(self, t_array: np.ndarray, acc_array: np.ndarray) -> np.ndarray:
        """Fully vectorized observation reconstruction from stored time indices.

        Replaces the original chunk-loop (chunk=64) with a single batched
        NumPy gather.  For batch_size=1024 this is ~16x faster; peak memory
        is bounded by (batch * window * num_stocks * F * 4 bytes) which is
        well under 100 MB for typical settings.

        Boundary clamp: t_array values are clamped so that the oldest window
        index (t - window_size) never goes below 0, preventing negative
        index wrapping that would silently read stale data.
        """
        t_array = np.asarray(t_array, dtype=np.int32)
        batch_size = len(t_array)
        acc_f32 = np.asarray(acc_array, dtype=np.float32)
        obs = np.empty((batch_size, self.num_stocks, self.obs_dim_per_stock), dtype=np.float32)

        w = self.window_size
        max_t = self.market_data.shape[0] - 1
        # Build gather indices: (batch, window); clamp to [0, max_t]
        idx = t_array[:, None] + np.arange(-w, 0, dtype=np.int32)   # (batch, window)
        idx = np.clip(idx, 0, max_t)
        # market_data: [max_steps, num_stocks, F]  →  gather: (batch, window, num_stocks, F)
        windows = self.market_data[idx]
        # Reshape to (batch, num_stocks, window*F)
        obs[:, :, :self.market_dim] = windows.transpose(0, 2, 1, 3).reshape(
            batch_size, self.num_stocks, self.market_dim
        )
        obs[:, :, self.market_dim:self.market_dim + self._NUM_ACCOUNT_FEATURES] = acc_f32
        if self.enable_sl_features:
            obs[:, :, self.market_dim + self._NUM_ACCOUNT_FEATURES:] = self.sl_data[t_array]

        return obs

    def _get_samples(self, batch_inds: np.ndarray, env: Optional[VecNormalize] = None) -> ReplayBufferSamples:
        env_indices = np.random.randint(0, high=self.n_envs, size=(len(batch_inds),))

        t_obs = self.t_memory[batch_inds, env_indices]
        acc_obs = self.account_observations[batch_inds, env_indices, :]
        obs = self._reconstruct_obs(t_obs, acc_obs)

        if self.optimize_memory_usage:
            next_inds = (batch_inds + 1) % self.buffer_size
            t_next = self.t_memory[next_inds, env_indices]
            acc_next = self.account_observations[next_inds, env_indices, :]
            next_obs = self._reconstruct_obs(t_next, acc_next)
        else:
            t_next = t_obs + 1
            acc_next = self.account_next_observations[batch_inds, env_indices, :]
            next_obs = self._reconstruct_obs(t_next, acc_next)

        data = (
            self._normalize_obs(obs, env),
            self.actions[batch_inds, env_indices, :],
            self._normalize_obs(next_obs, env),
            (self.dones[batch_inds, env_indices] * (1 - self.timeouts[batch_inds, env_indices])).reshape(-1, 1),
            self._normalize_reward(self.rewards[batch_inds, env_indices].reshape(-1, 1), env),
        )
        return ReplayBufferSamples(*tuple(map(self.to_torch, data)))

    @staticmethod
    def _maybe_cast_dtype(dtype: Any) -> Any:
        if dtype == np.float64:
            return np.float32
        return dtype


def estimated_bytes_per_transition(env, *, optimize_memory: bool, storage_dtype: np.dtype = np.float16) -> int:
    """RAM estimate for buffer capacity planning in train_portfolio."""
    unwrapped = env.envs[0] if hasattr(env, "envs") else env
    acc_item = unwrapped.num_stocks * unwrapped._NUM_ACCOUNT_FEATURES * np.dtype(storage_dtype).itemsize
    action_bytes = int(np.prod(env.action_space.shape)) * 4
    acc_bytes = acc_item if optimize_memory else acc_item * 2
    return 4 + acc_bytes + action_bytes + 16

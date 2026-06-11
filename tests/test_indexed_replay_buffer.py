import numpy as np
import pandas as pd
import pytest
from stable_baselines3.common.vec_env import DummyVecEnv

from indexed_replay_buffer import IndexedReplayBuffer
from trading_env import TaiwanStockEnv

@pytest.fixture
def dummy_env():
    # Create small dummy environment
    dates = pd.date_range("2020-01-01", periods=100)
    df1 = pd.DataFrame({"close": np.random.rand(100), "log_return": np.random.rand(100)}, index=dates)
    df2 = pd.DataFrame({"close": np.random.rand(100), "log_return": np.random.rand(100)}, index=dates)
    
    df_dict = {"2330": df1, "2317": df2}
    env = TaiwanStockEnv(df_dict, window_size=5, initial_balance=1e6)
    return env

def test_indexed_replay_buffer_reconstruction(dummy_env):
    vec_env = DummyVecEnv([lambda: dummy_env])
    obs_space = vec_env.observation_space
    act_space = vec_env.action_space
    
    buffer = IndexedReplayBuffer(
        buffer_size=10,
        observation_space=obs_space,
        action_space=act_space,
        n_envs=1,
        optimize_memory_usage=True,
        handle_timeout_termination=False,
        env=vec_env
    )
    
    obs = vec_env.reset()
    
    for i in range(15):
        action = np.array([vec_env.action_space.sample()])
        next_obs, reward, done, infos = vec_env.step(action)
        
        buffer.add(obs, next_obs, action, reward, done, infos)
        
        # Test exact match
        if i < 10:
            pos = i
        else:
            pos = i % 10
            
        samples = buffer._get_samples(np.array([pos]))

        np.testing.assert_allclose(
            samples.observations.cpu().numpy(), obs, rtol=1e-3, atol=1e-4
        )

        if not done[0]:
            np.testing.assert_allclose(
                samples.next_observations.cpu().numpy(), next_obs, rtol=1e-3, atol=1e-4
            )
            
        obs = next_obs
        
def test_indexed_replay_buffer_wrap_around(dummy_env):
    vec_env = DummyVecEnv([lambda: dummy_env])
    obs_space = vec_env.observation_space
    act_space = vec_env.action_space
    
    buffer = IndexedReplayBuffer(
        buffer_size=5,
        observation_space=obs_space,
        action_space=act_space,
        n_envs=1,
        optimize_memory_usage=True,
        handle_timeout_termination=False,
        env=vec_env
    )
    
    assert buffer.pos == 0
    obs = vec_env.reset()
    for _ in range(6):
        action = np.array([vec_env.action_space.sample()])
        next_obs, reward, done, infos = vec_env.step(action)
        buffer.add(obs, next_obs, action, reward, done, infos)
        obs = next_obs
        
    assert buffer.pos == 1
    assert buffer.full

def test_indexed_replay_buffer_without_optimize_memory(dummy_env):
    vec_env = DummyVecEnv([lambda: dummy_env])
    obs_space = vec_env.observation_space
    act_space = vec_env.action_space
    
    buffer = IndexedReplayBuffer(
        buffer_size=5,
        observation_space=obs_space,
        action_space=act_space,
        n_envs=1,
        optimize_memory_usage=False,
        env=vec_env
    )
    
    obs = vec_env.reset()
    action = np.array([vec_env.action_space.sample()])
    next_obs, reward, done, infos = vec_env.step(action)
    
    buffer.add(obs, next_obs, action, reward, done, infos)
    
    samples = buffer._get_samples(np.array([0]))
    np.testing.assert_allclose(samples.observations.cpu().numpy(), obs, rtol=1e-3, atol=1e-4)
    np.testing.assert_allclose(samples.next_observations.cpu().numpy(), next_obs, rtol=1e-3, atol=1e-4)


def test_indexed_replay_buffer_float16_storage(dummy_env):
    vec_env = DummyVecEnv([lambda: dummy_env])
    buffer = IndexedReplayBuffer(
        buffer_size=5,
        observation_space=vec_env.observation_space,
        action_space=vec_env.action_space,
        n_envs=1,
        optimize_memory_usage=False,
        env=vec_env,
        storage_dtype=np.float16,
    )
    assert buffer.account_observations.dtype == np.float16
    obs = vec_env.reset()
    action = np.array([vec_env.action_space.sample()])
    next_obs, reward, done, infos = vec_env.step(action)
    buffer.add(obs, next_obs, action, reward, done, infos)
    acc_block = obs[0, :, buffer.market_dim : buffer.market_dim + buffer._NUM_ACCOUNT_FEATURES]
    stored = buffer.account_observations[0, 0]
    np.testing.assert_array_equal(stored, acc_block.astype(np.float16))


def _reconstruct_obs_loop(buffer: IndexedReplayBuffer, t_array: np.ndarray, acc_array: np.ndarray) -> np.ndarray:
    """Reference loop implementation for vectorization equivalence checks."""
    batch_size = len(t_array)
    obs = np.empty((batch_size, buffer.num_stocks, buffer.obs_dim_per_stock), dtype=np.float32)
    acc_f32 = np.asarray(acc_array, dtype=np.float32)
    for i in range(batch_size):
        t = int(t_array[i])
        start = t - buffer.window_size
        market_slice = buffer.market_data[start:t].transpose(1, 0, 2).reshape(buffer.num_stocks, buffer.market_dim)
        obs[i, :, : buffer.market_dim] = market_slice
        obs[i, :, buffer.market_dim : buffer.market_dim + buffer._NUM_ACCOUNT_FEATURES] = acc_f32[i]
        if buffer.enable_sl_features:
            obs[i, :, buffer.market_dim + buffer._NUM_ACCOUNT_FEATURES :] = buffer.sl_data[t]
    return obs


def test_reconstruct_obs_vectorized_matches_loop(dummy_env):
    vec_env = DummyVecEnv([lambda: dummy_env])
    buffer = IndexedReplayBuffer(
        buffer_size=20,
        observation_space=vec_env.observation_space,
        action_space=vec_env.action_space,
        n_envs=1,
        optimize_memory_usage=True,
        handle_timeout_termination=False,
        env=vec_env,
    )
    obs = vec_env.reset()
    for _ in range(25):
        action = np.array([vec_env.action_space.sample()])
        next_obs, reward, done, infos = vec_env.step(action)
        buffer.add(obs, next_obs, action, reward, done, infos)
        obs = next_obs

    t_array = np.array([10, 15, 20, 25], dtype=np.int32)
    acc_array = buffer.account_observations[t_array % buffer.buffer_size, 0]
    vec_out = buffer._reconstruct_obs(t_array, acc_array)
    loop_out = _reconstruct_obs_loop(buffer, t_array, acc_array)
    np.testing.assert_allclose(vec_out, loop_out, rtol=0, atol=0)

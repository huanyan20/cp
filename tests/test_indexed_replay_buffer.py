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
        
        # Test bit-for-bit equivalence
        np.testing.assert_array_equal(samples.observations.cpu().numpy(), obs)
        
        if not done[0]:
            # Because optimize_memory_usage=True, the next_obs of a terminal state
            # might get overwritten by the initial obs of the next episode in the buffer.
            # But for non-terminal, it should match exactly.
            np.testing.assert_array_equal(samples.next_observations.cpu().numpy(), next_obs)
            
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
    np.testing.assert_array_equal(samples.observations.cpu().numpy(), obs)
    np.testing.assert_array_equal(samples.next_observations.cpu().numpy(), next_obs)

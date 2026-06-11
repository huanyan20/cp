"""P10 VecEnv smoke tests (Windows spawn)."""

import gymnasium as gym

from train_portfolio import PpoEfficiencyConfig, build_ppo_vec_env


def _cartpole_factory():
    return gym.make("CartPole-v1")


def test_ppo_efficiency_config_r6_defaults():
    cfg = PpoEfficiencyConfig()
    assert cfg.n_envs == 1
    assert cfg.vecenv == "dummy"
    assert cfg.n_steps == 256
    assert cfg.n_epochs == 10


def test_build_ppo_vec_env_dummy():
    vec = build_ppo_vec_env(_cartpole_factory, PpoEfficiencyConfig(n_envs=1))
    assert vec.num_envs == 1
    vec.close()


def test_build_ppo_vec_env_subproc_spawn_smoke():
    cfg = PpoEfficiencyConfig(n_envs=2, vecenv="subproc")
    vec = build_ppo_vec_env(_cartpole_factory, cfg)
    assert vec.num_envs == 2
    obs = vec.reset()
    assert len(obs) == 2
    vec.close()

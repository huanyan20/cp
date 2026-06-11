"""Train portfolio RL agents with PPO or SAC."""

import argparse
import os
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np

try:
    import winsound
except ImportError:  # pragma: no cover
    winsound = None

from stable_baselines3 import PPO, SAC  # noqa: E402
from stable_baselines3.common.callbacks import BaseCallback  # noqa: E402
from stable_baselines3.common.monitor import Monitor  # noqa: E402
from stable_baselines3.common.utils import set_random_seed  # noqa: E402
from stable_baselines3.common.vec_env import (  # noqa: E402
    DummyVecEnv,
    SubprocVecEnv,
    VecEnv,
)

from data_loader import fetch_multi_asset_data  # noqa: E402
from gnn_extractor import GnnFeatureExtractor, TemporalGnnFeatureExtractor  # noqa: E402
from settings import (  # noqa: E402
    describe_torch_device,
    load_settings,
    resolve_torch_device,
)
from stock_universe import MACRO_TICKERS_RL, TICKERS_TECH_EXPANDED  # noqa: E402
from trading_env import TaiwanStockEnv  # noqa: E402

SETTINGS = load_settings()
TRAIN_START = SETTINGS.research.train_start
TRAIN_END = SETTINGS.research.train_end
WINDOW_SIZE = SETTINGS.research.window_size
TIMESTEPS = SETTINGS.research.timesteps


@dataclass(frozen=True)
class PpoEfficiencyConfig:
    """P10 ablation knobs. Defaults match R6 (DummyVecEnv, n_steps=256, n_epochs=10)."""

    n_envs: int = 1
    vecenv: str = "dummy"  # dummy | subproc
    n_steps: int = 256
    n_epochs: int = 10
    batch_size: int = 64

    @classmethod
    def from_settings(cls, settings=None) -> "PpoEfficiencyConfig":
        s = settings or SETTINGS
        return cls(
            n_envs=s.research.ppo_n_envs,
            vecenv=s.research.ppo_vecenv,
            n_steps=s.research.ppo_n_steps,
            n_epochs=s.research.ppo_n_epochs,
            batch_size=s.research.ppo_batch_size,
        )


def build_ppo_vec_env(
    env_factory: Callable[[], Any],
    cfg: PpoEfficiencyConfig,
) -> VecEnv:
    """Wrap env factory in DummyVecEnv or SubprocVecEnv (Windows spawn)."""
    n_envs = max(1, cfg.n_envs)

    def _init() -> Monitor:
        return Monitor(env_factory())

    if n_envs == 1:
        return DummyVecEnv([_init])
    env_fns = [_init for _ in range(n_envs)]
    if cfg.vecenv == "subproc":
        return SubprocVecEnv(env_fns, start_method="spawn")
    return DummyVecEnv(env_fns)


class EntCoefScheduleCallback(BaseCallback):
    """Linear entropy coefficient annealing for PPO experiments."""

    def __init__(
        self,
        ent_coef_start: float = 0.05,
        ent_coef_end: float = 0.001,
        total_timesteps: int = 300_000,
        verbose: int = 0,
    ):
        super().__init__(verbose)
        self.ent_coef_start = ent_coef_start
        self.ent_coef_end = ent_coef_end
        self.total_timesteps = total_timesteps

    def _on_step(self) -> bool:
        progress = min(self.num_timesteps / self.total_timesteps, 1.0)
        new_ent_coef = self.ent_coef_start + progress * (
            self.ent_coef_end - self.ent_coef_start
        )
        self.model.ent_coef = float(new_ent_coef)
        if self.verbose > 0 and self.num_timesteps % 10_000 == 0:
            print(f"[EntCoef] step={self.num_timesteps:,} ent_coef={new_ent_coef:.4f}")
        return True


def play_done_sound():
    if winsound is None:
        return
    for _ in range(3):
        winsound.Beep(1000, 300)
        time.sleep(0.2)
    winsound.Beep(1500, 500)


def build_policy_kwargs(
    features_dim: int = 256,
    temporal_extractor: bool = False,
    window_size: int = WINDOW_SIZE,
) -> dict:
    extractor_class = TemporalGnnFeatureExtractor if temporal_extractor else GnnFeatureExtractor
    extractor_kwargs = dict(features_dim=features_dim)
    if temporal_extractor:
        extractor_kwargs.update(dict(window_size=window_size, account_features=6))
    return dict(
        features_extractor_class=extractor_class,
        features_extractor_kwargs=extractor_kwargs,
        net_arch=[256, 256],
    )


def build_model(
    algo: str,
    env,
    timesteps: int,
    temporal_extractor: bool = False,
    device: str | None = None,
    ppo_cfg: PpoEfficiencyConfig | None = None,
):
    if device is None:
        device = resolve_torch_device(SETTINGS.research.torch_device)
    else:
        device = resolve_torch_device(device)
    print(f"[Device] {describe_torch_device(device)}")

    policy_kwargs = build_policy_kwargs(temporal_extractor=temporal_extractor)

    if algo == "ppo":
        cfg = ppo_cfg or PpoEfficiencyConfig.from_settings()
        print(
            f"[PPO] vecenv={cfg.vecenv} n_envs={cfg.n_envs} "
            f"n_steps={cfg.n_steps} n_epochs={cfg.n_epochs} batch_size={cfg.batch_size}"
        )
        model = PPO(
            "MlpPolicy",
            env,
            verbose=1,
            device=device,
            learning_rate=3e-5,
            n_steps=cfg.n_steps,
            batch_size=cfg.batch_size,
            n_epochs=cfg.n_epochs,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.20,
            target_kl=0.08,
            ent_coef=0.05,
            policy_kwargs=policy_kwargs,
        )
        callback = EntCoefScheduleCallback(
            ent_coef_start=0.05,
            ent_coef_end=0.001,
            total_timesteps=timesteps,
            verbose=1,
        )
        return model, callback

    if algo == "sac":
        # 自動調整 buffer_size，避免 OOM (Out of Memory)
        # optimize_memory_usage=True 時 SB3 的 obs 與 next_obs 共用同一個陣列，
        # 每筆 transition 實際只佔一份 float32 obs（舊版公式 *2 多估了一倍）。
        # 預設上限 4GB（16GB RAM 機器，約 11K transitions ≈ 10 個 episode）。
        # SAC_BUFFER_RAM_GB 可覆寫：R6 續跑用 1（重現舊公式的 2,805，保 seed 一致）。
        ram_gb = float(os.environ.get("SAC_BUFFER_RAM_GB", "4"))
        obs_size = np.prod(env.observation_space.shape)
        bytes_per_transition = obs_size * 4
        max_buffer_by_ram = int((ram_gb * 1024**3) / bytes_per_transition)

        # 也不要超過訓練的總 timesteps
        buffer_size = min(timesteps, max_buffer_by_ram)

        print(f"[SAC] 自動調整 buffer_size = {buffer_size:,} (預估佔用記憶體: {buffer_size * bytes_per_transition / 1024**3:.1f} GB)")

        model = SAC(
            "MlpPolicy",
            env,
            verbose=1,
            device=device,
            learning_rate=3e-4,
            buffer_size=buffer_size,
            learning_starts=1_000,
            batch_size=256,
            tau=0.005,
            gamma=0.99,
            train_freq=10,
            gradient_steps=1,
            ent_coef="auto",
            optimize_memory_usage=True,
            policy_kwargs=policy_kwargs,
            replay_buffer_kwargs=dict(handle_timeout_termination=False),
        )
        return model, None

    raise ValueError(f"Unsupported algo: {algo}")


def train_ppo_with_config(
    env_factory: Callable[[], TaiwanStockEnv],
    ppo_cfg: PpoEfficiencyConfig,
    timesteps: int,
    temporal_extractor: bool = False,
    device: str | None = None,
) -> tuple[PPO, float]:
    """Train PPO on vec-wrapped env; return (model, elapsed_seconds)."""
    vec_env = build_ppo_vec_env(env_factory, ppo_cfg)
    model, callback = build_model(
        "ppo",
        vec_env,
        timesteps,
        temporal_extractor=temporal_extractor,
        device=device,
        ppo_cfg=ppo_cfg,
    )
    t_start = time.time()
    model.learn(total_timesteps=timesteps, progress_bar=True, callback=callback)
    elapsed = time.time() - t_start
    vec_env.close()
    return model, elapsed


def main(
    tickers=None,
    timesteps: int = TIMESTEPS,
    model_name: str | None = None,
    algo: str = "ppo",
    enable_cash_action: bool | None = None,
    enable_margin_short: bool = SETTINGS.research.default_enable_margin_short,
    seed: int = SETTINGS.research.default_seed,
    overnight_feature_path: str | None = None,
    temporal_extractor: bool = False,
    train_start: str = TRAIN_START,
    train_end: str = TRAIN_END,
    window_size: int = WINDOW_SIZE,
    topk: int = SETTINGS.research.default_topk,
    softmax_temp: float = SETTINGS.research.default_softmax_temp,
):
    set_random_seed(seed)
    tickers = tickers or TICKERS_TECH_EXPANDED
    algo = algo.lower()
    if enable_cash_action is None:
        enable_cash_action = SETTINGS.research.default_enable_cash_action or algo == "sac"
    if overnight_feature_path is None:
        overnight_feature_path = SETTINGS.research.overnight_feature_path
    if model_name is None:
        suffix = "cash" if enable_cash_action else "full_stock"
        if enable_margin_short:
            suffix += "_ls"
        model_name = f"{algo}_portfolio_{suffix}_seed{seed}"

    print("=== Portfolio Manager Training ===")
    print(f"algo={algo}")
    print(f"tickers={len(tickers)} stocks")
    print(f"macro_tickers={MACRO_TICKERS_RL}")
    print(f"enable_cash_action={enable_cash_action}")
    print(f"enable_margin_short={enable_margin_short}")
    print(f"overnight_feature_path={overnight_feature_path}")
    print(f"temporal_extractor={temporal_extractor}")
    print(f"timesteps={timesteps:,}")
    print(f"seed={seed}")
    print(f"train_range={train_start} ~ {train_end}")

    enriched = fetch_multi_asset_data(
        tickers=tickers,
        start_date=train_start,
        end_date=train_end,
        window_size=window_size,
        macro_tickers=MACRO_TICKERS_RL,
        overnight_feature_path=overnight_feature_path,
    )

    env = TaiwanStockEnv(
        df_dict=enriched,
        window_size=window_size,
        initial_balance=1_000_000.0,
        topk=topk,
        softmax_temp=softmax_temp,
        use_benchmark_reward=True,
        enable_cash_action=enable_cash_action,
        enable_margin_short=enable_margin_short,
        max_leverage=SETTINGS.risk_limits.max_leverage,
    )
    print(f"observation_space={env.observation_space.shape}")
    print(f"action_space={env.action_space.shape}")

    model, callback = build_model(
        algo,
        env,
        timesteps,
        temporal_extractor=temporal_extractor,
    )
    t_start = time.time()
    model.learn(total_timesteps=timesteps, progress_bar=True, callback=callback)
    elapsed = time.time() - t_start

    model.save(model_name)
    print(f"[OK] saved {model_name}.zip")
    print(f"[OK] training elapsed {elapsed / 60:.1f} minutes")
    play_done_sound()


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--algo", choices=["ppo", "sac"], default=SETTINGS.research.default_algo)
    parser.add_argument("--timesteps", type=int, default=TIMESTEPS)
    parser.add_argument("--model-name", default=None)
    parser.add_argument(
        "--enable-cash-action",
        action="store_true",
        help="Add a cash logit to the action space. Defaults on for SAC.",
    )
    parser.add_argument(
        "--disable-cash-action",
        action="store_true",
        help="Force legacy full-stock action space.",
    )
    parser.add_argument(
        "--enable-margin-short",
        action="store_true",
        help="Enable margin trading and short selling (Tanh + leverage normalization).",
    )
    parser.add_argument("--seed", type=int, default=SETTINGS.research.default_seed, help="Random seed for training")
    parser.add_argument(
        "--overnight-feature-path",
        default=None,
        help="Optional overnight_gap_features_1d.csv path for RL observation features.",
    )
    parser.add_argument("--train-start", default=TRAIN_START)
    parser.add_argument("--train-end", default=TRAIN_END)
    parser.add_argument("--window-size", type=int, default=WINDOW_SIZE)
    parser.add_argument("--topk", type=int, default=SETTINGS.research.default_topk)
    parser.add_argument("--softmax-temp", type=float, default=SETTINGS.research.default_softmax_temp)
    parser.add_argument(
        "--temporal-extractor",
        action="store_true",
        help="Use GRU-over-window TemporalGnnFeatureExtractor.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    cash_flag = None
    if args.enable_cash_action:
        cash_flag = True
    if args.disable_cash_action:
        cash_flag = False
    main(
        timesteps=args.timesteps,
        model_name=args.model_name,
        algo=args.algo,
        enable_cash_action=cash_flag,
        enable_margin_short=args.enable_margin_short,
        seed=args.seed,
        overnight_feature_path=args.overnight_feature_path,
        temporal_extractor=args.temporal_extractor,
        train_start=args.train_start,
        train_end=args.train_end,
        window_size=args.window_size,
        topk=args.topk,
        softmax_temp=args.softmax_temp,
    )

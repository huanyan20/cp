import os
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.callbacks import BaseCallback
import numpy as np

from gnn_extractor import GnnFeatureExtractor, TemporalGnnFeatureExtractor
from settings import resolve_torch_device, describe_torch_device, SETTINGS
from trading_env import NUM_ACCOUNT_FEATURES
from indexed_replay_buffer import IndexedReplayBuffer, estimated_bytes_per_transition
from core.equivariant_policy import EquivariantActorCriticPolicy


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


def build_policy_kwargs(
    algo: str = "ppo",
    features_dim: int = 256,  # Legacy, overwritten below
    temporal_extractor: bool = False,
    window_size: int = 20,
    num_stocks: int = 45,
    enable_cash_action: bool = False,
) -> dict:
    embed_dim = 64
    actual_features_dim = num_stocks * embed_dim

    extractor_class = (
        TemporalGnnFeatureExtractor if temporal_extractor else GnnFeatureExtractor
    )
    extractor_kwargs = dict(features_dim=actual_features_dim)
    if temporal_extractor:
        extractor_kwargs.update(
            dict(window_size=window_size, account_features=NUM_ACCOUNT_FEATURES)
        )

    kwargs = dict(
        features_extractor_class=extractor_class,
        features_extractor_kwargs=extractor_kwargs,
        net_arch=[],  # Empty to prevent MlpExtractor from mixing features across stocks
    )

    if algo == "ppo":
        kwargs.update(
            dict(
                num_stocks=num_stocks,
                embed_dim=embed_dim,
                enable_cash_action=enable_cash_action,
            )
        )

    return kwargs


class ModelTrainer:
    def __init__(self, algo: str, device: str = "auto"):
        self.algo = algo.lower()
        if self.algo not in ["ppo", "sac"]:
            raise ValueError(f"Unsupported algo: {self.algo}")
        self.device = resolve_torch_device(device)

    def build_model(
        self,
        env,
        timesteps: int,
        temporal_extractor: bool = False,
        window_size: int = 20,
    ):
        policy_kwargs = build_policy_kwargs(
            algo=self.algo,
            temporal_extractor=temporal_extractor,
            window_size=window_size,
            num_stocks=getattr(env, "num_stocks", 45),
            enable_cash_action=getattr(env, "enable_cash_action", False),
        )

        if self.algo == "ppo":
            model = PPO(
                EquivariantActorCriticPolicy,
                env,
                verbose=1,
                device=self.device,
                learning_rate=3e-5,
                n_steps=8192,
                batch_size=4096,
                n_epochs=10,
                gamma=0.99,
                gae_lambda=0.95,
                clip_range=0.20,
                target_kl=0.04,
                ent_coef=0.05,
                policy_kwargs=policy_kwargs,
                tensorboard_log="logs/tb_logs",
            )
            callback = EntCoefScheduleCallback(
                ent_coef_start=0.05,
                ent_coef_end=0.001,
                total_timesteps=timesteps,
                verbose=1,
            )
            return model, callback

        if self.algo == "sac":
            ram_gb = float(os.environ.get("SAC_BUFFER_RAM_GB", "4"))
            bytes_per_transition = estimated_bytes_per_transition(
                env, optimize_memory=True, storage_dtype=np.float16
            )
            max_buffer_by_ram = int((ram_gb * 1024**3) / bytes_per_transition)
            buffer_size = min(timesteps, max_buffer_by_ram, 100_000)

            print(
                f"[SAC] IndexedReplayBuffer buffer_size = {buffer_size:,} "
                f"(~{bytes_per_transition} B/transition, float16 account, optimize_memory=True, "
                f"est. RAM {buffer_size * bytes_per_transition / 1024**3:.2f} GB)"
            )

            model = SAC(
                "MlpPolicy",
                env,
                verbose=1,
                device=self.device,
                learning_rate=3e-4,
                buffer_size=buffer_size,
                learning_starts=1_000,
                batch_size=256,
                tau=0.005,
                gamma=0.99,
                train_freq=100,
                gradient_steps=5,
                ent_coef=0.005,
                optimize_memory_usage=True,
                policy_kwargs=policy_kwargs,
                replay_buffer_class=IndexedReplayBuffer,
                replay_buffer_kwargs=dict(
                    handle_timeout_termination=False,
                    env=env,
                    storage_dtype=np.float16,
                ),
                tensorboard_log="logs/tb_logs",
            )
            return model, None

    def load_model(self, model_path: str, env):
        model_class = PPO if self.algo == "ppo" else SAC
        return model_class.load(model_path, env=env, device=self.device)

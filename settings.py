"""Centralized project settings for research, evaluation, live trading, and paths."""

import os
from pathlib import Path
from typing import Optional

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT_DIR = Path(__file__).resolve().parent
RESULTS_DIR = ROOT_DIR / "results_dir"
MODELS_DIR = ROOT_DIR / "models_dir"
LOGS_DIR = ROOT_DIR / "logs"
CAPITAL_FLOW_DATA_DIR = ROOT_DIR / "capital_flow_analysis" / "data"

def resolve_torch_device(requested: Optional[str] = None) -> str:
    choice = (requested or os.getenv("RESEARCH_DEVICE", "auto")).strip().lower()
    if choice in {"gpu", "cuda"}:
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError(
                "RESEARCH_DEVICE=cuda but CUDA is unavailable. "
                "Install a CUDA PyTorch wheel, e.g. "
                "pip install torch --index-url https://download.pytorch.org/whl/cu124"
            )
        return "cuda"
    if choice == "cpu":
        return "cpu"
    if choice == "auto":
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    raise ValueError(f"Unknown RESEARCH_DEVICE: {choice!r}")

def describe_torch_device(device: str) -> str:
    if device != "cuda":
        return device
    import torch
    return f"cuda ({torch.cuda.get_device_name(0)})"

TIER_PRESETS: dict[str, dict[str, int]] = {
    "smoke": {"timesteps": 30_000, "seeds": 1},
    "candidate": {"timesteps": 500_000, "seeds": 2},
    "promotion": {"timesteps": 500_000, "seeds": 3},
}

def resolve_tier(tier: str, base_seeds: list[int]) -> tuple[int, list[int]]:
    key = (tier or "").strip().lower()
    if key not in TIER_PRESETS:
        raise ValueError(f"Unknown research tier: {tier!r}. Expected one of {sorted(TIER_PRESETS)}.")
    preset = TIER_PRESETS[key]
    seed_count = max(1, preset["seeds"])
    seeds = base_seeds[:seed_count] if base_seeds else []
    return preset["timesteps"], seeds

class ResearchSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="RESEARCH_", env_file=".env", extra="ignore")
    
    train_start: str = "2020-01-01"
    train_end: str = "2023-12-31"
    test_start: str = "2024-01-01"
    window_size: int = 20
    timesteps: int = 300_000
    algo: str = Field("sac", validation_alias="RESEARCH_ALGO")
    seed: int = Field(42, validation_alias="RESEARCH_SEED")
    seeds: str = Field("42,43,44", validation_alias="RESEARCH_SEEDS")
    tier: str = Field("", validation_alias="RESEARCH_TIER")
    topk: int = Field(5, validation_alias="RESEARCH_TOPK")
    softmax_temp: float = Field(1.0, validation_alias="RESEARCH_SOFTMAX_TEMP")
    enable_cash_action: bool = Field(False, validation_alias="RESEARCH_ENABLE_CASH_ACTION")
    enable_margin_short: bool = Field(False, validation_alias="RESEARCH_ENABLE_MARGIN_SHORT")
    env_config_hash: Optional[str] = Field(None, validation_alias="RESEARCH_ENV_CONFIG_HASH")
    env_config_version: Optional[str] = Field(None, validation_alias="RESEARCH_ENV_CONFIG_VERSION")
    torch_device: str = Field("auto", validation_alias="RESEARCH_DEVICE")

    # Non-prefixed env vars (aliased explicitly)
    walk_forward_timesteps: int = Field(300_000, validation_alias="WALK_FORWARD_TIMESTEPS")
    walk_forward_cash_mode: str = Field("enabled", validation_alias="WALK_FORWARD_CASH_MODE")
    overnight_feature_path: Optional[str] = Field(
        str(CAPITAL_FLOW_DATA_DIR / "overnight_gap_features_1d.csv"), 
        validation_alias="OVERNIGHT_FEATURE_PATH"
    )
    promotion_min_seeds: int = Field(3, validation_alias="PROMOTION_MIN_SEEDS")
    promotion_sortino_threshold: float = Field(0.8, validation_alias="PROMOTION_SORTINO_THRESHOLD")
    promotion_max_drawdown: float = Field(0.35, validation_alias="PROMOTION_MAX_DRAWDOWN")
    promotion_turnover_limit: float = Field(0.10, validation_alias="PROMOTION_TURNOVER_LIMIT")

    @property
    def default_algo(self) -> str: return self.algo
    @property
    def default_seed(self) -> int: return self.seed
    @property
    def default_seeds(self) -> str: return self.seeds
    @property
    def research_tier(self) -> str: return self.tier
    @property
    def default_topk(self) -> int: return self.topk
    @property
    def default_softmax_temp(self) -> float: return self.softmax_temp
    @property
    def default_enable_cash_action(self) -> bool: return self.enable_cash_action
    @property
    def default_enable_margin_short(self) -> bool: return self.enable_margin_short

class EvaluationSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="EVALUATION_", env_file=".env", extra="ignore")
    
    test_start: str = "2024-01-01"
    test_end_strategy: str = "today"
    model_name: str = "ppo_portfolio_full_stock_seed42.zip"
    output_file: str = "portfolio_evaluation_ppo_cash_v9.png"

class LiveSettings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")
    
    enable_live_trading: bool = Field(False, validation_alias="ENABLE_LIVE_TRADING")
    cmoney_aid: Optional[str] = Field(None, validation_alias="CMONEY_AID")
    signal_ttl_seconds: int = Field(900, validation_alias="SIGNAL_TTL_SECONDS")
    macro_guard_path: Path = Field(CAPITAL_FLOW_DATA_DIR / "preopen_macro_check.json")
    dry_run_diff_path: Path = Field(ROOT_DIR / "trade_guard_diff.json")

class RiskLimits(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="MAX_", env_file=".env", extra="ignore")
    
    single_weight: float = Field(0.35, validation_alias="MAX_SINGLE_WEIGHT")
    total_exposure: float = Field(1.0, validation_alias="MAX_TOTAL_EXPOSURE")
    leverage: float = Field(2.0, validation_alias="MAX_LEVERAGE")
    turnover: float = Field(0.30, validation_alias="MAX_TURNOVER")
    
    @property
    def max_single_weight(self) -> float: return self.single_weight
    @property
    def max_total_exposure(self) -> float: return self.total_exposure
    @property
    def max_leverage(self) -> float: return self.leverage
    @property
    def max_turnover(self) -> float: return self.turnover

class StressSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="STRESS_", env_file=".env", extra="ignore")
    
    base_fee_rate: float = 0.001425
    base_tax_rate: float = 0.003
    high_fee_rate: float = 0.003
    high_fee_tax_rate: float = 0.003
    high_slippage_fee_rate: float = 0.005
    high_slippage_tax_rate: float = 0.003
    worst_fee_rate: float = 0.005
    worst_tax_rate: float = 0.004

    @property
    def worst_case_fee_rate(self) -> float: return self.worst_fee_rate
    @property
    def worst_case_tax_rate(self) -> float: return self.worst_tax_rate

class PathSettings(BaseSettings):
    root_dir: Path = ROOT_DIR
    results_dir: Path = RESULTS_DIR
    models_dir: Path = MODELS_DIR
    logs_dir: Path = LOGS_DIR
    signal_path: Path = ROOT_DIR / "signal.json"
    execution_log_path: Path = ROOT_DIR / "executed_signals.json"
    experiment_report_md: Path = RESULTS_DIR / "experiment_report.md"
    experiment_summary_json: Path = ROOT_DIR / "experiment_summary.json"
    baseline_summary_path: Path = RESULTS_DIR / "baseline_summary.json"
    ablation_summary_path: Path = RESULTS_DIR / "ablation_summary.json"
    stress_summary_path: Path = RESULTS_DIR / "stress_summary.json"
    pending_buys_dir: Path = ROOT_DIR / "logs"

class AppSettings(BaseSettings):
    research: ResearchSettings = Field(default_factory=ResearchSettings)
    evaluation: EvaluationSettings = Field(default_factory=EvaluationSettings)
    live: LiveSettings = Field(default_factory=LiveSettings)
    paths: PathSettings = Field(default_factory=PathSettings)
    risk_limits: RiskLimits = Field(default_factory=RiskLimits)
    stress: StressSettings = Field(default_factory=StressSettings)

def load_settings() -> AppSettings:
    return AppSettings()

SETTINGS = load_settings()

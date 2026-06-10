"""Centralized project settings for research, evaluation, live trading, and paths."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent
RESULTS_DIR = ROOT_DIR / "results_dir"
MODELS_DIR = ROOT_DIR / "models_dir"
LOGS_DIR = ROOT_DIR / "logs"
CAPITAL_FLOW_DATA_DIR = ROOT_DIR / "capital_flow_analysis" / "data"


def _env_bool(name: str, default: bool = False) -> bool:
    return os.getenv(name, str(default)).strip().lower() == "true"


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except ValueError:
        return default


def _env_str(name: str, default: str) -> str:
    value = os.getenv(name)
    return default if value is None or value == "" else value


def _env_optional_str(name: str, default: str | None) -> str | None:
    value = os.getenv(name)
    if value is None:
        return default
    return value or None


def resolve_torch_device(requested: str | None = None) -> str:
    """Map RESEARCH_DEVICE (auto|cuda|cpu) to an SB3/PyTorch device string."""
    choice = (requested or _env_str("RESEARCH_DEVICE", "auto")).strip().lower()
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
    raise ValueError(
        f"Unknown RESEARCH_DEVICE: {choice!r}. Expected one of auto, cuda, cpu."
    )


def describe_torch_device(device: str) -> str:
    if device != "cuda":
        return device
    import torch

    return f"cuda ({torch.cuda.get_device_name(0)})"


# O2 — layered training protocol. Tiers differ by seed count; timesteps fixed at 300K.
TIER_PRESETS: dict[str, dict[str, int]] = {
    "smoke": {"timesteps": 300_000, "seeds": 1},
    "candidate": {"timesteps": 300_000, "seeds": 2},
    "promotion": {"timesteps": 300_000, "seeds": 3},
}


def resolve_tier(tier: str, base_seeds: list[int]) -> tuple[int, list[int]]:
    """Map a research tier to (timesteps, seeds) using the first N base seeds.

    Raises ValueError for unknown tiers so CLI typos fail fast.
    """
    key = (tier or "").strip().lower()
    if key not in TIER_PRESETS:
        raise ValueError(
            f"Unknown research tier: {tier!r}. Expected one of {sorted(TIER_PRESETS)}."
        )
    preset = TIER_PRESETS[key]
    seed_count = max(1, preset["seeds"])
    seeds = base_seeds[:seed_count] if base_seeds else []
    return preset["timesteps"], seeds


@dataclass(frozen=True)
class ResearchSettings:
    train_start: str = field(default_factory=lambda: _env_str("RESEARCH_TRAIN_START", "2020-01-01"))
    train_end: str = field(default_factory=lambda: _env_str("RESEARCH_TRAIN_END", "2023-12-31"))
    test_start: str = field(default_factory=lambda: _env_str("RESEARCH_TEST_START", "2024-01-01"))
    window_size: int = field(default_factory=lambda: _env_int("RESEARCH_WINDOW_SIZE", 20))
    timesteps: int = field(default_factory=lambda: _env_int("RESEARCH_TIMESTEPS", 300_000))
    walk_forward_timesteps: int = field(default_factory=lambda: _env_int("WALK_FORWARD_TIMESTEPS", 300_000))
    default_algo: str = field(default_factory=lambda: _env_str("RESEARCH_ALGO", "ppo"))
    default_seed: int = field(default_factory=lambda: _env_int("RESEARCH_SEED", 42))
    default_seeds: str = field(default_factory=lambda: _env_str("RESEARCH_SEEDS", "42,43,44"))
    walk_forward_cash_mode: str = field(default_factory=lambda: _env_str("WALK_FORWARD_CASH_MODE", "enabled"))
    # O2: optional training tier (smoke|candidate|promotion); empty keeps explicit timesteps/seeds.
    research_tier: str = field(default_factory=lambda: _env_str("RESEARCH_TIER", ""))
    default_topk: int = field(default_factory=lambda: _env_int("RESEARCH_TOPK", 5))
    default_softmax_temp: float = field(default_factory=lambda: _env_float("RESEARCH_SOFTMAX_TEMP", 0.5))
    default_enable_cash_action: bool = field(default_factory=lambda: _env_bool("RESEARCH_ENABLE_CASH_ACTION", False))
    default_enable_margin_short: bool = field(default_factory=lambda: _env_bool("RESEARCH_ENABLE_MARGIN_SHORT", False))
    overnight_feature_path: str | None = field(
        default_factory=lambda: _env_optional_str(
            "OVERNIGHT_FEATURE_PATH",
            str(CAPITAL_FLOW_DATA_DIR / "overnight_gap_features_1d.csv"),
        )
    )
    # ── Promotion gate thresholds (N2 — configurable via env var) ──────────────
    promotion_min_seeds: int = field(
        default_factory=lambda: _env_int("PROMOTION_MIN_SEEDS", 3)
    )
    promotion_sortino_threshold: float = field(
        default_factory=lambda: _env_float("PROMOTION_SORTINO_THRESHOLD", 0.8)
    )
    promotion_max_drawdown: float = field(
        default_factory=lambda: _env_float("PROMOTION_MAX_DRAWDOWN", 0.35)
    )
    promotion_turnover_limit: float = field(
        default_factory=lambda: _env_float("PROMOTION_TURNOVER_LIMIT", 0.10)
    )
    # O1: filter experiment_report to a specific env_config_hash (8-char); empty = auto
    env_config_hash: str | None = field(
        default_factory=lambda: _env_optional_str("RESEARCH_ENV_CONFIG_HASH", None)
    )
    env_config_version: str | None = field(
        default_factory=lambda: _env_optional_str("RESEARCH_ENV_CONFIG_VERSION", None)
    )
    # auto: use CUDA when available; cuda: require GPU; cpu: force CPU
    torch_device: str = field(default_factory=lambda: _env_str("RESEARCH_DEVICE", "auto"))


@dataclass(frozen=True)
class EvaluationSettings:
    test_start: str = field(default_factory=lambda: _env_str("EVALUATION_TEST_START", "2024-01-01"))
    test_end_strategy: str = field(default_factory=lambda: _env_str("EVALUATION_TEST_END_STRATEGY", "today"))
    model_name: str = field(default_factory=lambda: _env_str("EVALUATION_MODEL_NAME", "ppo_portfolio_full_stock_seed42.zip"))
    output_file: str = field(default_factory=lambda: _env_str("EVALUATION_OUTPUT_FILE", "portfolio_evaluation_ppo_cash_v9.png"))


@dataclass(frozen=True)
class LiveSettings:
    enable_live_trading: bool = field(default_factory=lambda: _env_bool("ENABLE_LIVE_TRADING", False))
    cmoney_aid: str | None = os.getenv("CMONEY_AID")
    signal_ttl_seconds: int = field(default_factory=lambda: _env_int("SIGNAL_TTL_SECONDS", 900))
    macro_guard_path: Path = field(default_factory=lambda: CAPITAL_FLOW_DATA_DIR / "preopen_macro_check.json")
    dry_run_diff_path: Path = field(default_factory=lambda: ROOT_DIR / "trade_guard_diff.json")


@dataclass(frozen=True)
class RiskLimits:
    max_single_weight: float = field(default_factory=lambda: _env_float("MAX_SINGLE_WEIGHT", 0.35))
    max_total_exposure: float = field(default_factory=lambda: _env_float("MAX_TOTAL_EXPOSURE", 1.0))
    max_leverage: float = field(default_factory=lambda: _env_float("MAX_LEVERAGE", 2.0))
    max_turnover: float = field(default_factory=lambda: _env_float("MAX_TURNOVER", 0.30))


@dataclass(frozen=True)
class StressSettings:
    """Fee/slippage rates (one-way %) for the four cost sensitivity scenarios.

    Override any value via the corresponding env var (all floats, e.g. 0.001425).
    Scenarios
    ---------
    base          — standard Taiwan brokerage discount rate + sell tax
    high_fee      — elevated brokerage fee, same sell tax
    high_slippage — large spread/slippage assumption, same sell tax
    worst_case    — high slippage + higher sell tax (conservative ceiling)
    """

    # base scenario
    base_fee_rate: float = field(
        default_factory=lambda: _env_float("STRESS_BASE_FEE_RATE", 0.001425)
    )
    base_tax_rate: float = field(
        default_factory=lambda: _env_float("STRESS_BASE_TAX_RATE", 0.003)
    )
    # high_fee scenario
    high_fee_rate: float = field(
        default_factory=lambda: _env_float("STRESS_HIGH_FEE_RATE", 0.003)
    )
    high_fee_tax_rate: float = field(
        default_factory=lambda: _env_float("STRESS_HIGH_FEE_TAX_RATE", 0.003)
    )
    # high_slippage scenario
    high_slippage_fee_rate: float = field(
        default_factory=lambda: _env_float("STRESS_HIGH_SLIPPAGE_FEE_RATE", 0.005)
    )
    high_slippage_tax_rate: float = field(
        default_factory=lambda: _env_float("STRESS_HIGH_SLIPPAGE_TAX_RATE", 0.003)
    )
    # worst_case scenario
    worst_case_fee_rate: float = field(
        default_factory=lambda: _env_float("STRESS_WORST_FEE_RATE", 0.005)
    )
    worst_case_tax_rate: float = field(
        default_factory=lambda: _env_float("STRESS_WORST_TAX_RATE", 0.004)
    )


@dataclass(frozen=True)
class PathSettings:
    root_dir: Path = ROOT_DIR
    results_dir: Path = RESULTS_DIR
    models_dir: Path = MODELS_DIR
    logs_dir: Path = LOGS_DIR
    signal_path: Path = ROOT_DIR / "signal.json"
    execution_log_path: Path = ROOT_DIR / "executed_signals.json"
    experiment_report_md: Path = ROOT_DIR / "experiment_report.md"
    experiment_summary_json: Path = ROOT_DIR / "experiment_summary.json"
    # P5 summary files consumed by promotion gate
    baseline_summary_path: Path = RESULTS_DIR / "baseline_summary.json"
    ablation_summary_path: Path = RESULTS_DIR / "ablation_summary.json"
    stress_summary_path: Path = RESULTS_DIR / "stress_summary.json"
    # N4: pending buy orders directory (replaces root-level scatter)
    pending_buys_dir: Path = ROOT_DIR / "logs"


@dataclass(frozen=True)
class AppSettings:
    research: ResearchSettings = field(default_factory=ResearchSettings)
    evaluation: EvaluationSettings = field(default_factory=EvaluationSettings)
    live: LiveSettings = field(default_factory=LiveSettings)
    paths: PathSettings = field(default_factory=PathSettings)
    risk_limits: RiskLimits = field(default_factory=RiskLimits)
    stress: StressSettings = field(default_factory=StressSettings)


def load_settings() -> AppSettings:
    return AppSettings()

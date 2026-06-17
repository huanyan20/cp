"""Supervised-learning pipeline (方案 B): SignalGenerator + PortfolioAllocator."""

from sl_pipeline.allocator import (
    MarketContext,
    PortfolioAllocator,
    PortfolioState,
    TargetPortfolio,
)
from sl_pipeline.backtest import (
    BacktestConfig,
    build_sl_seed_metrics,
    metrics_from_backtest,
    simulate_period,
)
from sl_pipeline.comparison import build_sl_vs_rl_comparison
from sl_pipeline.gate import (
    build_sl_raw_summary,
    read_sl_metric_files,
    run_sl_promotion_gate,
)
from sl_pipeline.labels import (
    HORIZON_DAYS,
    build_cross_demean_frame,
    build_feature_panel,
    build_labeled_panel,
    forward_log_return_t1,
    label_column_name,
)
from sl_pipeline.rl_allocator import RLAllocator, RLAllocatorConfig
from sl_pipeline.rule_based_allocator import (
    RuleBasedAllocator,
    RuleBasedAllocatorConfig,
)
from sl_pipeline.signal_generator import SignalGenerator, SignalGeneratorConfig
from sl_pipeline.sl_features import (
    SL_FEATURE_VERSION,
    SL_FEATURES_PER_STOCK,
    build_sl_feature_arrays,
)

__all__ = [
    "BacktestConfig",
    "HORIZON_DAYS",
    "MarketContext",
    "PortfolioAllocator",
    "PortfolioState",
    "RLAllocator",
    "RLAllocatorConfig",
    "RuleBasedAllocator",
    "RuleBasedAllocatorConfig",
    "SL_FEATURE_VERSION",
    "SL_FEATURES_PER_STOCK",
    "SignalGenerator",
    "SignalGeneratorConfig",
    "TargetPortfolio",
    "build_feature_panel",
    "build_cross_demean_frame",
    "build_labeled_panel",
    "build_sl_raw_summary",
    "build_sl_seed_metrics",
    "build_sl_feature_arrays",
    "build_sl_vs_rl_comparison",
    "forward_log_return_t1",
    "label_column_name",
    "metrics_from_backtest",
    "read_sl_metric_files",
    "run_sl_promotion_gate",
    "simulate_period",
]

# Changelog 2026-06-14

## SL h10 Pipeline Repair and Promotion
The Supervised Learning (SL) pipeline with `h10` (10-day horizon) and `rule` allocator has been officially repaired, passing all 5 critical promotion gates (including Sortino stability, drawdown gate, and turnover limits). The model is now eligible for live dry-run (`trade_guard`).

### Key Fixes
1. **Mean-Reversion Bias Fixed (Target Asymmetry)**:
   - **Problem**: The LightGBM model previously learned to strongly favor short-term mean-reversion, picking the weakest, most volatile stocks (e.g., legacy tech dumping in 2024H2) rather than trend followers (e.g., AI stocks), leading to severe losses in the bull market.
   - **Solution**: Restored `.clip(lower=0.0)` in `labels.py` on the cross-sectionally demeaned targets. This acts as a ReLU activation on the labels, forcing the model to ignore negative return noise and dedicate its capacity entirely to finding right-tail "winners" (trend followers).
2. **Huber Loss for Robustness**:
   - Switched LightGBM `objective` to `huber` in `signal_generator.py` to reduce the sensitivity to extreme noisy outliers during training.
3. **Allocator Constraints Fine-Tuned**:
   - Restored `min_score = 1e-4` in `RuleBasedAllocatorConfig` so the allocator ignores neutral/negative expectation stocks during weak or sideways regimes instead of being forced to buy 5 random stocks.
4. **MDD Death Spiral & Macro Guard Calibrated**:
   - Adjusted `red_mdd` to 35% and `yellow_mdd` to 20% in the allocator to match the user's hard constraint, preventing the model from permanently locking into a 100% cash state prematurely.
   - Tweaked `Macro Guard` thresholds in `backtest.py` to be more sensitive to early downtrends (`-0.02` for CRITICAL, `0.01` for WARN based on TWII 60d momentum) to protect the portfolio dynamically rather than relying solely on internal MDD limits.

### Results
- **Overall Return**: +134.42%
- **Overall MDD**: 32.45% (Compliant with < 35% limit)
- **Overall Sortino**: 1.36
- **Turnover**: Maintained strictly within acceptable bounds.
- **Gate Status**: APPROVED (5/5). Ready for `trade_guard` dry run phase.

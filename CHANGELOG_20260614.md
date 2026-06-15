# Changelog 2026-06-14

> Superseded by 2026-06-16 gate review. The active candidate is
> `sl_rule_h10_top20_equal_no_voltarget`; it is **BLOCKED / high risk** and
> remains dry-run only until a current multiseed gate explicitly approves it.

## SL h10 Pipeline Repair and Promotion
The 2026-06-14 run was an intermediate repair attempt for the Supervised
Learning (SL) `h10` pipeline with the `rule` allocator. Do not treat this file
as promotion approval. Current status must be read from the latest
`sl_gate_result_rule_h10_multiseed.json`, `experiment_report.md`, or
`python scripts/sl_status_check.py`.

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
- **Historical run only**: these figures are no longer the active promotion
  decision.
- **Current policy**: SL h10 is BLOCKED / high risk until the active candidate
  passes the current three-seed h10 gate.
- **Live policy**: `trade_guard` dry-run is allowed; CMoney RPA is blocked
  unless the active candidate gate reports `core` or `full` approval.

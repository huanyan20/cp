# Developer Guide

> Strategy snapshot: 2026-06-14
> Decision: SL-first for production candidates; SAC is research-only.

## 1. Operating Principle

目前開發重心是把監督式學習管線打磨成可上線候選：`sl_pipeline` 產生 signal，rule allocator 做倉位配置，promotion gate 與 trade_guard 負責風險守門。SAC / PPO 不再是預設上線路徑。

SAC 的定位改為快速研究工具：用少 steps、單 seed、少 period 先判斷 reward/action 設計是否有明顯病徵。若出現全現金、單股集中、弱期崩壞、turnover 爆炸，就停止該方向，不進完整多 seed promotion。

## 2. Main Research Flow

```text
prepare data
  -> train/evaluate SL signal
  -> allocate with rule allocator
  -> generate metrics
  -> promotion_gate
  -> trade_guard dry-run
  -> live only after explicit APPROVED
```

Recommended commands:

```powershell
.\env\Scripts\python.exe -m sl_pipeline.walk_forward_sl --allocator rule --gate --seed 42
.\env\Scripts\python.exe experiment_report.py
```

SAC research commands should stay smoke-sized unless a result has already passed the early failure checks:

```powershell
.\env\Scripts\python.exe walk_forward.py --tier smoke --algo sac --cash-mode enabled
```

## 3. Component Roles

| Component | Current role |
|---|---|
| `sl_pipeline/` | Mainline signal, labels, allocator, walk-forward. |
| `promotion_gate.py` | Required gate for any production candidate. |
| `experiment_report.py` | Summarizes metrics; report text may be manually annotated for strategy decisions. |
| `trading_env.py` | RL environment retained for SAC/PPO smoke and ablation. |
| `train_portfolio.py` / `walk_forward.py` | RL research tools; not default production path. |
| `rpa_pipeline/trade_guard.py` | Final live risk guard, dry-run diff, and signal validation. |

## 4. SAC Failure Checks

Stop a SAC experiment early if any of these appear in smoke metrics:

- Avg cash weight near 100% across most OOS periods.
- Long exposure near 100% with weak-period drawdown above gate budget.
- A single holding dominates top holdings.
- 2024H2 or 2025H1 collapses while total return is rescued by 2026H1.
- Turnover is too high after realistic fees/tax/slippage.

Passing smoke does not mean promotion. It only allows a candidate-sized run.

## 5. SL Priorities

1. SL h10 is **BLOCKED / under repair** and does not pass current promotion gates.
2. The current focus is live integration (e.g. `trade_guard.py` dry-run compatibility) and monitoring real-world execution.
3. Keep monitoring the model's feature importance (especially trend vs mean-reversion bias) and the allocator's behavior.
4. Add more interpretable diagnostics if necessary.
5. Live deployment and RPA automation can now be considered following a successful dry-run.

## 6. Documentation Policy

Historical SAC-heavy plans in `docs/archive/`, `.research/archive/`, and `.cursor/plans/` are retained for context only. They are superseded by this 2026-06-14 SL-first decision and must not be treated as active implementation queues.

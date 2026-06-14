> Superseded by 2026-06-14 SL-first strategy.
> Retained for review/history only; current implementation priority is SL-first, SAC research-only.
# R6 凍結基線（env r4）

> **勿與 r5.1 直接比較**。hash=`1bbfe5c2` · `env_config_version=r4`

| 檔案 | algo | cash | seed | worst MDD | 備註 |
|------|------|------|------|-----------|------|
| `metrics_sac_enabled_wf_seed42.json` | sac | enabled | 42 | 37.7% | 2025H1 MDD = smoke 門檻 |
| `metrics_sac_enabled_wf_seed43.json` | sac | enabled | 43 | **46.0%** | 集中度問題（8046 98.9% @ 2024H2） |
| `metrics_sac_enabled_wf_seed44.json` | sac | enabled | 44 | 38.2% | |
| `metrics_ppo_disabled_wf_seed*.json` | ppo | disabled | * | — | 對照線 |

Gate 參考：worst-case MDD **44.41%** → Drawdown FAIL（limit 35%）。

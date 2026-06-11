# 短跑 / 實驗產物（非 canonical baselines）

> 完整 WF 指標寫入 `results_dir/`；凍結基線見 [`../baselines/`](../baselines/README.md)。

| 檔案 | 說明 |
|------|------|
| [`m2-smoke-quick-metrics.json`](m2-smoke-quick-metrics.json) | r5.1 · 2025H1 only · **30K** · seed42 · 方向性參考 |

## m2-smoke-quick 摘要（2026-06-12）

| 指標 | 值 | smoke 門檻 |
|------|-----|------------|
| env | r5.1 / eea25956 | r5.1 |
| 2025H1 MDD | **31.87%** | < r4 seed42 **37.7%** |
| max top holding | **18.3%** | < **50%** |
| Sortino | -0.70 | 次目標（30K 欠訓練） |
| Return | -12.44% | 方向性 only |

**解讀**：集中度與 MDD 方向達標，但 30K 不足以 promotion；需 **300K 全期 smoke** 再決定 candidate。

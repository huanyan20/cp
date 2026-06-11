# scripts/

手動執行的診斷、分析與一次性驗證工具（非核心流程、非自動化測試套件）。

> **研究排程**（2026-06-11 v3）：活躍路線圖見 [`docs/RESEARCH_STRATEGY_V3.md`](../docs/RESEARCH_STRATEGY_V3.md)。  
> `sac_gradient_ablation.py`（R7b）已刪。分層 WF 用 repo 根目錄 `walk_forward.py`；編排乾跑用 `research_orchestrator.py`。

> **重要**：一律從 **repo 根目錄** 執行，例如 `python scripts/shap_analysis.py`。
> 這些腳本以 `Path(__file__).resolve().parent.parent` 推導 repo 根並注入 `sys.path`，
> 同時以 cwd 相對路徑讀取 `results_dir/`、`capital_flow_analysis/data/` 等產物。

## 內容

| 腳本 | 用途 |
|------|------|
| `analyze_gap.py` | overnight gap 特徵與 ADR premium 預測力分析 |
| `analyze_model_data.py` | 檢視模型 zip 內部資料結構 |
| `error_analysis.py` | 動能/期間錯誤歸因分析 |
| `friction_analysis.py` | 交易成本與 trade-level 摩擦分析 |
| `sector_analysis.py` | 板塊配置分析 |
| `shap_analysis.py` | PPO 動作的 SHAP 特徵歸因 |
| `model_test_report.py` | 產生模型載入測試報告 |
| `verify_models.py` | 驗證 zip 模型檔完整性 |
| `validation_p4_p5.py` | P4/P5 重構計畫驗證檢查 |
| `test_load_from_zip.py` | 直接從 zip 載入模型測試 |
| `test_single_model.py` | 單一模型載入/推論快速測試 |
| `test_models.py` | 批次測試所有 zip 模型 |
| `test_margin_short.py` | margin/short 環境煙霧測試 |

## 注意

- `scripts/` 已從 `ruff` / `mypy` 掃描範圍排除（見 `pyproject.toml`）。
- 此處的 `test_*.py` 為臨時手動腳本，**不屬於** `tests/` 的自動化測試套件。
- 部分腳本引用較舊的模型檔名（如 `wf_ppo_model_2024H2.zip`），與目前 walk-forward 命名
  （`wf_ppo_enabled_model_2024H2_seed42`）不同，使用前請先確認檔名。

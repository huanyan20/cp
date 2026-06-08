from __future__ import annotations

import argparse
import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    recall_score,
)
from sklearn.model_selection import TimeSeriesSplit

BASE_DIR = Path(__file__).resolve().parents[2]
DATA_PATH = BASE_DIR / "data" / "overnight_gap_features_1d.csv"
REPORT_DIR = BASE_DIR / "reports"

FEATURE_SETS = {
    "Baseline": ["baseline_ret_prev"],
    "ADR_Premium_Chg_Only": ["baseline_ret_prev", "tsm_adr_premium_chg"],
    "ADR_Only": ["baseline_ret_prev", "tsm_adr_premium", "tsm_adr_premium_chg", "TSM_ret"],
    "ADR_SOX": [
        "baseline_ret_prev",
        "tsm_adr_premium",
        "tsm_adr_premium_chg",
        "TSM_ret",
        "sox_ret",
        "sox_nasdaq_spread",
        "semi_hardware_stress_flag",
    ],
    "ADR_SOX_Risk": [
        "baseline_ret_prev",
        "tsm_adr_premium",
        "tsm_adr_premium_chg",
        "TSM_ret",
        "sox_ret",
        "sox_nasdaq_spread",
        "semi_hardware_stress_flag",
        "vix_ret",
        "vix_ret_z",
        "vix_level_z",
        "vix_panic_combo",
        "jpy_strength",
        "dxy_ret",
        "carry_unwind_flag",
    ],
    "Full_Breadth": [
        "baseline_ret_prev",
        "tsm_adr_premium",
        "tsm_adr_premium_chg",
        "TSM_ret",
        "sox_ret",
        "sox_nasdaq_spread",
        "semi_hardware_stress_flag",
        "vix_ret",
        "vix_ret_z",
        "vix_level_z",
        "vix_panic_combo",
        "jpy_strength",
        "dxy_ret",
        "carry_unwind_flag",
        "semi_adr_positive_count",
        "semi_adr_weighted_ret",
        "tsm_only_strength_flag",
    ],
    "Full_Crypto": [
        "baseline_ret_prev",
        "tsm_adr_premium",
        "tsm_adr_premium_chg",
        "TSM_ret",
        "sox_ret",
        "sox_nasdaq_spread",
        "semi_hardware_stress_flag",
        "vix_ret",
        "vix_ret_z",
        "vix_level_z",
        "vix_panic_combo",
        "jpy_strength",
        "dxy_ret",
        "carry_unwind_flag",
        "semi_adr_positive_count",
        "semi_adr_weighted_ret",
        "tsm_only_strength_flag",
        "btc_weekend_gap",
    ],
    "Extended_Macro": [
        "baseline_ret_prev",
        "tsm_adr_premium",
        "tsm_adr_premium_chg",
        "TSM_ret",
        "sox_ret",
        "sox_nasdaq_spread",
        "semi_hardware_stress_flag",
        "vix_ret",
        "vix_ret_z",
        "vix_level_z",
        "vix_panic_combo",
        "jpy_strength",
        "dxy_ret",
        "carry_unwind_flag",
        "semi_adr_positive_count",
        "semi_adr_weighted_ret",
        "tsm_only_strength_flag",
        "btc_weekend_gap",
        "iwm_qqq_spread",
        "gold_copper_ratio_chg",
        "ewt_ret",
    ],
}

TARGET_MAP = {
    "open_gap": "target_2330_open_gap",
    "intraday": "target_2330_intraday",
    "full_day": "target_2330_full_day",
    "gap_fade": "target_gap_fade",
}


def read_feature_data(path: Path = DATA_PATH) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Feature dataset not found: {path}. "
            "Run `python capital_flow_analysis/src/data_pipeline/overnight_gap_features.py --start 2020-01-01 --report` first."
        )
    df = pd.read_csv(path)
    if "tw_trade_date" in df.columns:
        df["tw_trade_date"] = pd.to_datetime(df["tw_trade_date"])
        df = df.sort_values("tw_trade_date").set_index("tw_trade_date")
    else:
        df.index = pd.to_datetime(df.index)
    return df


def coerce_model_frame(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in out.columns:
        if out[col].dtype == "bool":
            out[col] = out[col].astype(float)
        elif out[col].dtype == "object":
            out[col] = pd.to_numeric(out[col], errors="ignore")
    return out


def prepare_dataset(df: pd.DataFrame, target_key: str) -> tuple[pd.DataFrame, str]:
    if target_key not in TARGET_MAP:
        raise ValueError(f"Unsupported target: {target_key}")
    target_col = TARGET_MAP[target_key]
    out = df.copy()
    if "target_gap_fade" not in out:
        out["target_gap_fade"] = (
            (out["target_2330_open_gap"] > 0.01)
            & (out["target_2330_intraday"] < -0.005)
        )
    out["baseline_ret_prev"] = out["target_2330_full_day"].shift(1)
    if "corporate_action_flag" in out:
        out = out[out["corporate_action_flag"] == False].copy()  # noqa: E712
    out = coerce_model_frame(out)
    out = out.dropna(subset=[target_col, "baseline_ret_prev"])
    return out, target_col


def available_features(df: pd.DataFrame, features: list[str]) -> list[str]:
    return [col for col in features if col in df.columns]


def make_time_split(n_rows: int, n_splits: int = 5, test_size: int = 200) -> TimeSeriesSplit:
    if n_rows <= 20:
        raise ValueError("Not enough rows for walk-forward evaluation")
    effective_test_size = min(test_size, max(5, n_rows // (n_splits + 2)))
    effective_splits = min(n_splits, max(2, (n_rows // effective_test_size) - 1))
    return TimeSeriesSplit(n_splits=effective_splits, test_size=effective_test_size)


def fill_train_median(X_train: pd.DataFrame, X_test: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    med = X_train.median(numeric_only=True).fillna(0.0)
    return X_train.fillna(med), X_test.fillna(med)


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    large_idx = np.abs(y_true) >= 0.01
    return {
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "Dir_Acc": float(np.mean(np.sign(y_true) == np.sign(y_pred))),
        "Large_Gap_Acc": (
            float(np.mean(np.sign(y_true[large_idx]) == np.sign(y_pred[large_idx])))
            if np.sum(large_idx) > 0
            else np.nan
        ),
    }


def classification_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> dict[str, float]:
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "Threshold": threshold,
        "Precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "Recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "F1": float(f1_score(y_true, y_pred, zero_division=0)),
        "Positive_Rate": float(np.mean(y_true)),
        "TP": int(tp),
        "FP": int(fp),
        "TN": int(tn),
        "FN": int(fn),
    }


def evaluate_regression(df: pd.DataFrame, target_col: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    splitter = make_time_split(len(df))
    results = []
    importances = []
    full_features = available_features(df, FEATURE_SETS["Extended_Macro"])

    for set_name, features in FEATURE_SETS.items():
        used = available_features(df, features)
        X = df[used]
        y = df[target_col].astype(float)
        fold_metrics = []
        for train_idx, test_idx in splitter.split(X):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
            X_train, X_test = fill_train_median(X_train, X_test)
            model = lgb.LGBMRegressor(
                n_estimators=120,
                learning_rate=0.05,
                max_depth=4,
                num_leaves=15,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                verbosity=-1,
            )
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
            fold_metrics.append(regression_metrics(y_test.values, preds))
            if set_name == "Extended_Macro":
                importances.append(model.feature_importances_)
        avg = {k: np.nanmean([fold[k] for fold in fold_metrics]) for k in fold_metrics[0]}
        avg["Feature Set"] = set_name
        results.append(avg)

    result_df = pd.DataFrame(results)[["Feature Set", "MAE", "RMSE", "Dir_Acc", "Large_Gap_Acc"]]
    importance_df = pd.DataFrame()
    if importances:
        importance_df = pd.DataFrame(
            {
                "Feature": full_features,
                "Importance (Gain/Split)": np.mean(importances, axis=0),
            }
        ).sort_values("Importance (Gain/Split)", ascending=False)
    return result_df, importance_df


def evaluate_gap_fade(df: pd.DataFrame, target_col: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    splitter = make_time_split(len(df))
    results = []
    importances = []
    full_features = available_features(df, FEATURE_SETS["Extended_Macro"])

    for set_name, features in FEATURE_SETS.items():
        used = available_features(df, features)
        X = df[used]
        y = df[target_col].astype(int)
        
        # fold_metrics_by_thresh[threshold] = [ fold_metrics_dict, ... ]
        thresholds = [0.5, 0.6, 0.7]
        fold_metrics_by_thresh = {t: [] for t in thresholds}
        
        for train_idx, test_idx in splitter.split(X):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
            X_train, X_test = fill_train_median(X_train, X_test)
            if y_train.nunique() < 2:
                prob = np.full(len(y_test), float(y_train.mean()))
                feature_importances = np.zeros(len(used))
            else:
                model = lgb.LGBMClassifier(
                    n_estimators=160,
                    learning_rate=0.05,
                    max_depth=3,
                    num_leaves=7,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    class_weight="balanced",
                    random_state=42,
                    verbosity=-1,
                )
                model.fit(X_train, y_train)
                prob = model.predict_proba(X_test)[:, 1]
                feature_importances = model.feature_importances_
                
            for t in thresholds:
                fold_metrics_by_thresh[t].append(classification_metrics(y_test.values, prob, threshold=t))
                
            if set_name == "Extended_Macro":
                importances.append(feature_importances)
                
        for t in thresholds:
            avg = {k: np.nanmean([fold[k] for fold in fold_metrics_by_thresh[t]]) for k in fold_metrics_by_thresh[t][0]}
            avg["Feature Set"] = set_name
            results.append(avg)

    result_df = pd.DataFrame(results)[
        ["Feature Set", "Threshold", "Precision", "Recall", "F1", "Positive_Rate", "TP", "FP", "TN", "FN"]
    ]
    importance_df = pd.DataFrame()
    if importances:
        importance_df = pd.DataFrame(
            {
                "Feature": full_features,
                "Importance (Gain/Split)": np.mean(importances, axis=0),
            }
        ).sort_values("Importance (Gain/Split)", ascending=False)
    return result_df, importance_df


def _report_summary_lines(target_key: str, result_df: pd.DataFrame) -> list[str]:
    if result_df.empty:
        return ["No evaluation rows available."]
    if target_key == "gap_fade":
        best_f1 = result_df.sort_values("F1", ascending=False).iloc[0]
        best_precision = result_df.sort_values("Precision", ascending=False).iloc[0]
        return [
            f"Best F1: `{best_f1['Feature Set']}` at threshold {best_f1['Threshold']:.2f} "
            f"(F1={best_f1['F1']:.4f}, Precision={best_f1['Precision']:.4f}, Recall={best_f1['Recall']:.4f}).",
            f"Highest precision: `{best_precision['Feature Set']}` at threshold {best_precision['Threshold']:.2f} "
            f"(Precision={best_precision['Precision']:.4f}).",
            "Gap fade remains a risk hint, not a direct trading signal, unless precision and recall improve across more data.",
        ]
    best_mae = result_df.sort_values("MAE", ascending=True).iloc[0]
    best_dir = result_df.sort_values("Dir_Acc", ascending=False).iloc[0]
    best_large = result_df.sort_values("Large_Gap_Acc", ascending=False).iloc[0]
    lines = [
        f"Lowest MAE: `{best_mae['Feature Set']}` (MAE={best_mae['MAE']:.4f}, RMSE={best_mae['RMSE']:.4f}).",
        f"Best direction accuracy: `{best_dir['Feature Set']}` (Dir_Acc={best_dir['Dir_Acc']:.4f}).",
        f"Best large-gap accuracy: `{best_large['Feature Set']}` (Large_Gap_Acc={best_large['Large_Gap_Acc']:.4f}).",
    ]
    if target_key == "intraday":
        lines.append("Intraday results should be treated as research context unless they beat baseline consistently.")
    else:
        lines.append("Open-gap results are the strongest current use case for capital-flow features.")
    return lines


def write_report(
    target_key: str,
    result_df: pd.DataFrame,
    importance_df: pd.DataFrame,
    df: pd.DataFrame,
    report_dir: Path = REPORT_DIR,
) -> Path:
    report_dir.mkdir(parents=True, exist_ok=True)
    report_name = "gap_model_evaluation.md" if target_key == "open_gap" else f"gap_model_evaluation_{target_key}.md"
    report_path = report_dir / report_name

    high_premium = df[df["tsm_adr_premium"] > 0.01] if "tsm_adr_premium" in df else pd.DataFrame()
    if not high_premium.empty and "target_gap_fade" in high_premium:
        fade_rate = float(high_premium["target_gap_fade"].mean())
    else:
        fade_rate = np.nan

    title = target_key.replace("_", " ").title()
    with report_path.open("w", encoding="utf-8") as f:
        f.write(f"# Capital Flow Gap Model - {title} Evaluation\n\n")
        f.write("> Method: LightGBM + TimeSeriesSplit walk-forward\n")
        f.write("> Purpose: Evaluate whether overnight capital-flow features improve this target.\n\n")
        f.write("## 1. Summary\n\n")
        for line in _report_summary_lines(target_key, result_df):
            f.write(f"- {line}\n")
        f.write("\n## 2. Ablation Study Results\n\n")
        f.write(result_df.to_markdown(index=False, floatfmt=".4f"))
        f.write("\n\n")
        f.write("## 3. Gap Fade Context\n\n")
        if pd.notna(fade_rate):
            f.write(f"- ADR premium > 1% 時的 gap fade rate: {fade_rate:.2%}\n")
        else:
            f.write("- ADR premium > 1% 時的 gap fade rate: N/A\n")
        if not importance_df.empty:
            f.write("\n## 4. Top Features for RL\n\n")
            f.write(importance_df.head(8).to_markdown(index=False, floatfmt=".2f"))
            f.write("\n")
        f.write("\n## 5. Recommendation\n\n")
        if target_key == "gap_fade":
            f.write("- Use this model as a WARN-style risk hint before treating it as a trading rule.\n")
            f.write("- Prefer threshold calibration over a fixed 0.5 threshold.\n")
        elif target_key == "intraday":
            f.write("- Do not use intraday results for order direction until they beat baseline consistently.\n")
            f.write("- Add Taiwan-side flow or opening microstructure data before expanding macro features.\n")
        else:
            f.write("- Keep ADR premium and ADR returns as the first open-gap baseline.\n")
            f.write("- Avoid expanding the RL observation space until walk-forward metrics improve.\n")
    return report_path


def write_json_report(
    target_key: str,
    result_df: pd.DataFrame,
    importance_df: pd.DataFrame,
    report_dir: Path = REPORT_DIR,
) -> Path:
    report_dir.mkdir(parents=True, exist_ok=True)
    report_name = "gap_model_evaluation.json" if target_key == "open_gap" else f"gap_model_evaluation_{target_key}.json"
    report_path = report_dir / report_name
    
    data = {
        "target": target_key,
        "results": result_df.to_dict(orient="records"),
        "importances": importance_df.to_dict(orient="records") if not importance_df.empty else []
    }
    with report_path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    return report_path


def run_evaluation(target: str, data_path: Path = DATA_PATH) -> tuple[pd.DataFrame, pd.DataFrame, Path]:
    data_path = Path(data_path)
    raw = read_feature_data(data_path)
    df, target_col = prepare_dataset(raw, target)
    report_dir = REPORT_DIR if data_path.resolve() == DATA_PATH.resolve() else data_path.parent
    if target == "gap_fade":
        result_df, importance_df = evaluate_gap_fade(df, target_col)
    else:
        result_df, importance_df = evaluate_regression(df, target_col)
    report_path = write_report(target, result_df, importance_df, df, report_dir=report_dir)
    write_json_report(target, result_df, importance_df, report_dir=report_dir)
    return result_df, importance_df, report_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate overnight gap feature sets.")
    parser.add_argument("--target", choices=sorted(TARGET_MAP), default="open_gap")
    parser.add_argument("--data", default=str(DATA_PATH))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_path = Path(args.data)
    result_df, importance_df, report_path = run_evaluation(args.target, data_path)
    report_name = "gap_model_evaluation.json" if args.target == "open_gap" else f"gap_model_evaluation_{args.target}.json"
    report_dir = REPORT_DIR if data_path.resolve() == DATA_PATH.resolve() else data_path.parent
    json_path = report_dir / report_name
    print("\n=== Walk-Forward Evaluation Results ===")
    print(result_df.to_markdown(index=False, floatfmt=".4f"))
    if not importance_df.empty:
        print("\n=== Top Ranked Features ===")
        print(importance_df.head(8).to_markdown(index=False, floatfmt=".2f"))
    print(f"\nReports saved to {report_path} and {json_path}")


if __name__ == "__main__":
    main()

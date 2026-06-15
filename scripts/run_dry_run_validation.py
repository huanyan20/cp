"""Dry-run validation orchestrator for SL Live."""

import json
import logging
import os
import subprocess
import sys
from pathlib import Path

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from settings import load_settings

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("DryRunValidation")
SETTINGS = load_settings()

def run_command(cmd: list[str]) -> bool:
    logger.info(f"Executing: {' '.join(cmd)}")
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    if result.returncode != 0:
        logger.error(f"Command failed with exit code {result.returncode}")
        logger.error(f"STDOUT: {result.stdout}")
        logger.error(f"STDERR: {result.stderr}")
        return False
    logger.info(result.stdout)
    return True

def main():
    aid = os.getenv("CMONEY_AID", SETTINGS.live.cmoney_aid)
    if not aid:
        logger.error("CMONEY_AID is not set in environment or settings. Cannot run validation.")
        sys.exit(1)

    signal_path = SETTINGS.paths.signal_path
    diff_path = SETTINGS.live.dry_run_diff_path

    # Step 1: Generate live signal
    logger.info("\n" + "="*40 + "\nStep 1: Generate Live Signal (SL h10)\n" + "="*40)
    eval_cmd = [
        sys.executable,
        "scripts/evaluate_sl_live.py",
        "--horizon", "10",
        "--top-k", str(SETTINGS.research.default_topk),
        "--output", str(signal_path),
        "--aid", str(aid)
    ]
    if not run_command(eval_cmd):
        logger.error("Failed to generate live signal.")
        sys.exit(1)

    if not Path(signal_path).exists():
        logger.error(f"Signal file not found at {signal_path}")
        sys.exit(1)

    # Step 2: Run trade guard
    logger.info("\n" + "="*40 + "\nStep 2: Run Pre-trade Guard\n" + "="*40)
    diff_file = Path(diff_path)
    if diff_file.exists():
        diff_file.unlink()
    guard_cmd = [
        sys.executable,
        "rpa_pipeline/trade_guard.py",
        "--signal", str(signal_path),
        "--aid", str(aid),
        "--output", str(diff_path),
        "--no-write-equity"
    ]
    if not run_command(guard_cmd):
        logger.error("Trade guard encountered an error.")
        sys.exit(1)
    
    if not diff_file.exists():
        logger.error(f"Dry run diff file not found at {diff_path}")
        sys.exit(1)

    # Step 3: Parse and summarize results
    logger.info("\n" + "="*40 + "\nStep 3: Validation Summary\n" + "="*40)
    try:
        diff = json.loads(diff_file.read_text(encoding="utf-8"))
        risk = diff.get("risk_checks", {})
        
        passed = risk.get("passed", False)
        reasons = risk.get("reasons", [])
        
        logger.info(f"Risk Checks Passed: {passed}")
        if not passed:
            logger.error("Risk Failures:")
            for r in reasons:
                logger.error(f" - {r}")
        else:
            logger.info("No risk violations detected.")
            
        logger.info(f"Max Single Weight Observed: {risk.get('observed_max_single_weight', 0.0):.4f} (Limit: {risk.get('max_single_weight', 0.0):.4f})")
        logger.info(f"Total Exposure Observed: {risk.get('observed_total_exposure', 0.0):.4f} (Limit: {risk.get('max_total_exposure', 0.0):.4f})")
        logger.info(f"Circuit Breaker MDD: {risk.get('mdd', 0.0)*100:.2f}%")
        logger.info(f"Circuit Breaker Daily Loss: {risk.get('daily_loss', 0.0)*100:.2f}%")
        
        plan = diff.get("plan", {"buys": {}, "sells": {}})
        sells = plan.get("sells", {})
        buys = plan.get("buys", {})
        logger.info(f"Generated Orders: {len(sells)} Sells, {len(buys)} Buys")
        
        # Append to Daily Dry-run Report
        report_path = Path("results_dir/daily_dry_run_report.json")
        from datetime import datetime
        try:
            import yfinance as yf
            def get_price(ticker):
                # Simple helper to fetch latest price for value estimation
                try:
                    hist = yf.Ticker(f"{ticker}.TW" if not ticker.endswith(".TW") else ticker).history(period="5d")
                    if not hist.empty and "Close" in hist:
                        return float(hist["Close"].dropna().iloc[-1])
                except Exception:
                    pass
                return 0.0

            signal_data = json.loads(Path(signal_path).read_text(encoding="utf-8"))
            signal_id = signal_data.get("signal_id", "unknown")
            weights = signal_data.get("target_weights", {})
            top_holdings = {k: round(v, 4) for k, v in sorted(weights.items(), key=lambda x: x[1], reverse=True)[:5]}
            
            target_lots = signal_data.get("target_lots", {})
            nonzero_target_lots_count = sum(1 for qty in target_lots.values() if qty > 0)
            max_single_lot = max(target_lots.values(), default=0) if target_lots else 0

            estimated_trade_value = sum(qty * get_price(t) * 1000 for t, qty in buys.items()) + \
                                    sum(qty * get_price(t) * 1000 for t, qty in sells.items())
            
            max_single_value = max((qty * get_price(t) * 1000 for t, qty in target_lots.items() if qty > 0), default=0.0)

            latest_stress_mdd = 0.0
            stress_path = Path("results_dir/stress_summary.json")
            if stress_path.exists():
                try:
                    stress_data = json.loads(stress_path.read_text(encoding="utf-8"))
                    tests = stress_data.get("tests", {})
                    latest_stress_mdd = max((t_data.get("max_drawdown", 0.0) for t_data in tests.values()), default=0.0)
                except Exception:
                    pass

            report = {
                "date": datetime.now().strftime("%Y-%m-%d"),
                "signal_id": signal_id,
                "candidate_id": signal_data.get("metadata", {}).get("candidate_id"),
                "gate_status": signal_data.get("metadata", {}).get("gate_status"),
                "top_holdings": top_holdings,
                "total_exposure": round(risk.get("observed_total_exposure", 0.0), 4),
                "max_single_weight": round(risk.get("observed_max_single_weight", 0.0), 4),
                "risk_check_passed": passed,
                "rpa_allowed": passed and signal_data.get("metadata", {}).get("gate_status") in {"core", "full"},
                "generated_buys": len(buys),
                "generated_sells": len(sells),
                "macro_guard_level": signal_data.get("metadata", {}).get("macro_guard_level", "OK"),
                "nonzero_target_lots_count": nonzero_target_lots_count,
                "max_single_lot": max_single_lot,
                "max_single_value": round(max_single_value, 2),
                "estimated_trade_value": round(estimated_trade_value, 2),
                "latest_stress_mdd": round(latest_stress_mdd, 4)
            }
            
            history = []
            if report_path.exists():
                try:
                    history = json.loads(report_path.read_text(encoding="utf-8"))
                except json.JSONDecodeError:
                    pass
            history.append(report)
            report_path.parent.mkdir(parents=True, exist_ok=True)
            report_path.write_text(json.dumps(history, indent=4, ensure_ascii=False), encoding="utf-8")
            logger.info(f"Daily dry-run report appended to {report_path}")
        except Exception as e:
            logger.warning(f"Failed to append daily dry-run report: {e}")
        
        if passed:
            logger.info(
                "\nSUCCESS: Dry-run validation passed. This is dry-run only; live/RPA "
                "still requires an approved promotion gate."
            )
        else:
            logger.error("\nFAILURE: Dry-run validation blocked by risk checks. Do not execute this signal.")
            sys.exit(1)
            
    except Exception as e:
        logger.error(f"Failed to parse or summarize dry run diff: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()

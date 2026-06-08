import json
import logging
import os
import smtplib
import subprocess
import sys
import time
from datetime import datetime
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

from dotenv import load_dotenv

from settings import load_settings

load_dotenv()
SETTINGS = load_settings()

# ==========================================
# 1. Logging Setup
# ==========================================
LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)
log_filename = os.path.join(
    LOG_DIR, f"daily_trade_{datetime.now().strftime('%Y%m%d')}.log"
)

# Configure logger
logger = logging.getLogger("DailyTradeRunner")
logger.setLevel(logging.DEBUG)

# File handler
fh = logging.FileHandler(log_filename, encoding="utf-8")
fh.setLevel(logging.DEBUG)
# Console handler
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)

formatter = logging.Formatter("%(asctime)s - [%(levelname)s] - %(message)s")
fh.setFormatter(formatter)
ch.setFormatter(formatter)

logger.addHandler(fh)
logger.addHandler(ch)

LIVE_TTL_SECONDS = SETTINGS.live.signal_ttl_seconds
LIVE_AID = SETTINGS.live.cmoney_aid
DRY_RUN_DIFF_PATH = str(SETTINGS.live.dry_run_diff_path)


# ==========================================
# 2. Notification System (Extensible)
# ==========================================
def send_notification(message, level="INFO"):
    """
    發送通知到 Email。
    請於 .env 檔案中設定以下變數：
    - EMAIL_SENDER (寄件者信箱，例如 Gmail)
    - EMAIL_PASSWORD (寄件者密碼，若是 Gmail 請使用「應用程式密碼」)
    - EMAIL_RECIPIENT (收件者信箱，不設定則預設寄給自己)
    - EMAIL_SMTP_SERVER (預設 smtp.gmail.com)
    - EMAIL_SMTP_PORT (預設 587)
    """
    prefix = f"[{level}]"
    logger.debug(f"[Notification] {prefix} {message}")
    
    sender = os.getenv("EMAIL_SENDER")
    password = os.getenv("EMAIL_PASSWORD")
    recipient = os.getenv("EMAIL_RECIPIENT", sender)
    smtp_server = os.getenv("EMAIL_SMTP_SERVER", "smtp.gmail.com")
    smtp_port = int(os.getenv("EMAIL_SMTP_PORT", "587"))

    if not sender or not password:
        logger.warning("EMAIL_SENDER or EMAIL_PASSWORD not set in .env. Email notification skipped.")
        return

    try:
        msg = MIMEMultipart()
        msg["From"] = sender
        msg["To"] = recipient
        msg["Subject"] = f"【自動交易系統 {level}】排程回報"
        
        # 建立郵件內文
        body = f"時間：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        body += f"層級：{level}\n\n"
        body += f"訊息內容：\n{message}"
        
        msg.attach(MIMEText(body, "plain", "utf-8"))

        # 連線至 SMTP 伺服器並發送
        server = smtplib.SMTP(smtp_server, smtp_port)
        server.starttls()
        server.login(sender, password)
        server.send_message(msg)
        server.quit()
        logger.debug(f"Email sent successfully to {recipient}.")
    except Exception as e:
        logger.error(f"Failed to send email notification: {e}")


# ==========================================
# 3. Process Execution with Retries
# ==========================================
def run_command(cmd, max_retries=1, timeout=300):
    """
    執行系統指令，具備重試機制與超時防護。
    """
    for attempt in range(1, max_retries + 1):
        try:
            logger.info(
                f"Running command (Attempt {attempt}/{max_retries}): {' '.join(cmd)}"
            )
            result = subprocess.run(
                cmd,
                check=True,
                capture_output=True,
                text=True,
                timeout=timeout,
                errors="replace",
            )
            logger.info(f"Command succeeded:\n{result.stdout.strip()}")
            return True, result.stdout
        except subprocess.TimeoutExpired:
            logger.error(f"Command timed out after {timeout}s.")
        except subprocess.CalledProcessError as e:
            logger.error(
                f"Command failed with exit code {e.returncode}.\nSTDOUT: {e.stdout}\nSTDERR: {e.stderr}"
            )
        except Exception as e:
            logger.error(f"Unexpected error executing command: {e}")

        if attempt < max_retries:
            wait_time = 10 * attempt
            logger.warning(f"Retrying in {wait_time} seconds...")
            time.sleep(wait_time)

    return False, "Max retries reached or unrecoverable error."


def _load_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _require_live_execution_context():
    if os.getenv("ENABLE_LIVE_TRADING", "false").strip().lower() != "true":
        raise RuntimeError("ENABLE_LIVE_TRADING must be true for live execution.")
    if not LIVE_AID:
        raise RuntimeError("CMONEY_AID must be set for live execution.")


def _require_signal_guard(signal_path: str = "signal.json", guard_path: str = "capital_flow_analysis/data/preopen_macro_check.json"):
    signal = _load_json(signal_path)
    if str(signal.get("aid")) != str(LIVE_AID):
        raise RuntimeError(f"signal aid mismatch: {signal.get('aid')} != {LIVE_AID}")
    created_at = signal.get("created_at")
    if not created_at:
        raise RuntimeError("signal.json missing created_at")
    created = datetime.fromisoformat(created_at.replace("Z", "+00:00"))
    age_seconds = (datetime.now(created.tzinfo) - created).total_seconds()
    if age_seconds > LIVE_TTL_SECONDS:
        raise RuntimeError(f"signal expired: age {age_seconds:.0f}s > {LIVE_TTL_SECONDS}s")
    guard = _load_json(guard_path)
    if guard.get("level") != "OK":
        raise RuntimeError(f"macro guard is not OK: {guard.get('level')}")


def _require_dry_run_diff(signal_path: str = "signal.json"):
    diff_cmd = [
        sys.executable,
        "trade_guard.py",
        "--signal",
        signal_path,
        "--aid",
        LIVE_AID or "",
        "--output",
        DRY_RUN_DIFF_PATH,
    ]
    ok, output = run_command(diff_cmd, max_retries=1, timeout=300)
    if not ok:
        raise RuntimeError(f"dry-run diff generation failed: {output}")
    if not os.path.exists(DRY_RUN_DIFF_PATH):
        raise RuntimeError("dry-run diff file was not created.")
    diff = _load_json(DRY_RUN_DIFF_PATH)
    risk_checks = diff.get("risk_checks", {})
    if risk_checks and not risk_checks.get("passed", False):
        reasons = "; ".join(risk_checks.get("reasons", []))
        raise RuntimeError(f"dry-run diff risk checks failed: {reasons}")


# ==========================================
# 4. Main Workflow
# ==========================================
def main():
    logger.info("=" * 50)
    logger.info("Daily Automated Trading Workflow Started")
    logger.info("=" * 50)

    send_notification("Daily trade workflow started.", "INFO")
    
    live_trading_enabled = os.getenv("ENABLE_LIVE_TRADING", "false").strip().lower() == "true"
    macro_critical = False
    macro_warn = False

    # ==========================================
    # Step -1: 06:00 Pre-open macro guard
    # ==========================================
    try:
        logger.info("[Preopen Guard] Running macro risk check...")
        guard_cmd = [
            sys.executable,
            "capital_flow_analysis/src/monitoring/preopen_macro_check.py",
        ]
        guard_ok, guard_output = run_command(guard_cmd, max_retries=1, timeout=180)
        guard_path = str(SETTINGS.live.macro_guard_path)
        if not guard_ok:
            macro_critical = True
            send_notification("Preopen macro check failed. Fail-closed: skip pending buys.", "CRITICAL")
        elif os.path.exists(guard_path):
            with open(guard_path, encoding="utf-8") as f:
                guard_status = json.load(f)
            macro_critical = guard_status.get("level") == "CRITICAL"
            macro_warn = guard_status.get("level") == "WARN"
            
            if macro_critical:
                reasons = "; ".join(guard_status.get("critical_reasons", []))
                msg = f"Preopen macro CRITICAL: {reasons}. Pending buys will be skipped."
                logger.critical(msg)
                send_notification(msg, "CRITICAL")
            elif macro_warn:
                reasons = "; ".join(guard_status.get("warn_reasons", []))
                msg = f"Preopen macro WARN: {reasons}. Pending buys will be halved."
                logger.warning(msg)
                send_notification(msg, "WARN")
            else:
                logger.info("[Preopen Guard] OK")
    except Exception as e:
        macro_critical = True
        logger.error(f"[Preopen Guard] Unexpected error: {e}")
        send_notification("Preopen macro check error. Fail-closed: skip pending buys.", "CRITICAL")

    # ==========================================
    # Step 0: 自動登入與執行前一日 Pending BUY 單
    # ==========================================
    try:
        logger.info("[自動登入] 正在背景換取最新 Cookie...")
        login_cmd = [sys.executable, "auto_login.py"]
        login_result = subprocess.run(
            login_cmd, check=True, capture_output=True, text=True
        )
        if "[警告]" in login_result.stdout:
            logger.warning(f"自動登入過程出現警告，請檢查: {login_result.stdout}")
        else:
            logger.info("[V] 自動登入成功，Cookie 已更新")

        if macro_critical:
            logger.critical("[Step 0] Preopen guard is CRITICAL. Skipping Pending BUY orders.")
        else:
            logger.info("[Step 0] 執行前一日 Pending BUY 單...")
            pending_cmd = [sys.executable, "cmoney_rpa.py", "--pending-buys"]
            if macro_warn:
                logger.info("[Step 0] Applying WARN guard: pending buys will be halved.")
                pending_cmd.append("--half-buys")
            if live_trading_enabled:
                pending_cmd.append("--execute")
            
            # 這裡不重試，失敗就繼續，避免卡住後面的賣單
            success_pending, pending_output = run_command(pending_cmd, max_retries=1, timeout=300)
            if not success_pending:
                logger.error(f"[Step 0] Pending BUY 單執行失敗或超時 (略過繼續執行): {pending_output}")
            else:
                logger.info("[Step 0] Pending BUY 處理完成。")
            
    except Exception as e:
        logger.error(f"[Step 0] 自動登入或 Pending BUY 發生未預期錯誤: {e}")

    # ==========================================
    # Step 1: Evaluate Portfolio (Fetch data, Run Model, Generate signal.json)
    # ==========================================
    logger.info("[Step 1] Evaluating Portfolio and Generating Signals...")
    
    # 防呆：執行前刪除舊的 signal.json，避免使用陳舊訊號
    if os.path.exists(str(SETTINGS.paths.signal_path)):
        try:
            os.remove(str(SETTINGS.paths.signal_path))
            logger.info(f"舊的 {SETTINGS.paths.signal_path} 已刪除。")
        except Exception as e:
            logger.warning(f"無法刪除舊的 signal.json: {e}")

    eval_cmd = [
        sys.executable, 
        "evaluate_portfolio.py",
        "--model-path",
        "ppo_portfolio_full_stock_seed42.zip",
        "--overnight-feature-path",
        "capital_flow_analysis/data/overnight_gap_features_1d.csv"
    ]
    
    if macro_warn:
        logger.info("[Step 1] Applying WARN guard: evaluate_portfolio target weights will be halved.")
        eval_cmd.append("--half-buys")

    success, eval_output = run_command(eval_cmd, max_retries=3, timeout=600)

    if not success:
        msg = "Step 1 Failed: evaluate_portfolio.py encountered an error after multiple retries."
        logger.critical(msg)
        send_notification(msg, "ERROR")
        sys.exit(1)

    # Check if signal.json was generated
    if not os.path.exists(str(SETTINGS.paths.signal_path)):
        msg = "Step 1 Failed: signal.json was not found. Model evaluation might have failed silently."
        logger.critical(msg)
        send_notification(msg, "ERROR")
        sys.exit(1)

    logger.info("signal.json successfully generated.")

    try:
        _require_dry_run_diff(str(SETTINGS.paths.signal_path))
        logger.info(f"Dry-run diff written to {DRY_RUN_DIFF_PATH}")
    except Exception as e:
        msg = f"Dry-run diff validation failed: {e}"
        logger.critical(msg)
        send_notification(msg, "ERROR")
        sys.exit(1)

    # ==========================================
    # Step 2: CMoney 大富翁 RPA 下單 (T+1 模式：只賣出)
    # ==========================================
    try:
        _require_live_execution_context()
        _require_signal_guard(str(SETTINGS.paths.signal_path), str(SETTINGS.live.macro_guard_path))
        rpa_cmd = [
            sys.executable,
            "cmoney_rpa.py",
            "--signal",
            str(SETTINGS.paths.signal_path),
            "--sell-only",
        ]
        if live_trading_enabled:
            rpa_cmd.append("--execute")
            logger.warning(
                "[LIVE TRADING ENABLED] (ENABLE_LIVE_TRADING=true) — 真實委託將被送出！"
            )
        else:
            logger.warning(
                "[DRY-RUN 模式] ENABLE_LIVE_TRADING 未設定或非 'true'，本次不會真實下單。"
                " 若要啟用，請在 .env 中設定 ENABLE_LIVE_TRADING=true。"
            )
            
        logger.info("[Step 2] 開始執行 CMoney 大富翁 RPA (Sell-Only)...")
        success, rpa_output = run_command(rpa_cmd, max_retries=1, timeout=300)
    except Exception as e:
        import traceback
        logger.error(f"Exception before or during run_command: {traceback.format_exc()}")
        success = False
        rpa_output = str(e)

    if not success:
        logger.error(f"RPA Output / Error details: {rpa_output}")
        # Check if it was an auth error
        if (
            "Unauthorized" in rpa_output
            or "cookie" in rpa_output.lower()
            or "登入" in rpa_output
        ):
            msg = "Step 2 Failed: CMoney RPA Authentication Error. Check your .env cookie!"
        else:
            msg = "Step 2 Failed: CMoney RPA execution failed."
        logger.critical(msg)
        send_notification(msg, "ERROR")
        sys.exit(1)

    logger.info("=" * 50)
    logger.info("Daily Automated Trading Workflow Completed Successfully!")
    logger.info("=" * 50)
    send_notification(
        "Daily trade workflow completed successfully! Target positions submitted.",
        "INFO",
    )


if __name__ == "__main__":
    main()

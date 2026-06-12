@echo off
cd /d "%~dp0"

echo ==========================================
echo Starting Robust Daily Trading Automation...
echo ==========================================

REM Call Python Runner
set PYTHONPATH=%cd%
env\Scripts\python.exe rpa_pipeline\daily_trade_runner.py

echo.
echo ==========================================
echo Process Finished. Check "logs\" folder for details.
echo ==========================================
pause

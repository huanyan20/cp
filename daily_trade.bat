@echo off
cd /d "%~dp0"

echo ==========================================
echo Starting Robust Daily Trading Automation...
echo ==========================================

REM Call Python Runner
env\Scripts\python.exe daily_trade_runner.py

echo.
echo ==========================================
echo Process Finished. Check "logs\" folder for details.
echo ==========================================
pause

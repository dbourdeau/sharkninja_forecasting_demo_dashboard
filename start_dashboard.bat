@echo off
REM Start Dashboard from Any Location

cd /d "%~dp0"

echo ============================================
echo Call Center Volume Forecasting Dashboard
echo ============================================
echo.
echo Starting dashboard from: %CD%
echo.

REM Check if Python is available
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH
    echo Please install Python 3.8 or higher
    pause
    exit /b 1
)

REM Check if data exists, generate if needed
if not exist "data\combined_data.csv" (
    echo Generating synthetic data...
    python generate_data.py
    echo.
)

echo Starting Streamlit dashboard...
echo.
echo Dashboard will open in your browser at: http://localhost:8501
echo.
echo Press Ctrl+C to stop the server.
echo.

python -m streamlit run dashboard.py

pause


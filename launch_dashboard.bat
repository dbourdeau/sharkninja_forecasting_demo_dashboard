@echo off
cd /d "%~dp0"
echo.
echo ============================================
echo Call Center Volume Forecasting Dashboard
echo ============================================
echo.
echo Current directory: %CD%
echo.
echo Checking for dashboard.py...
if exist "dashboard.py" (
    echo ✓ Found dashboard.py
    echo.
    echo Starting Streamlit dashboard...
    echo The dashboard will open in your browser automatically.
    echo Press Ctrl+C to stop the server.
    echo.
    python -m streamlit run dashboard.py
) else (
    echo ✗ ERROR: dashboard.py not found in current directory
    echo.
    echo Please make sure you're in the correct directory:
    echo %CD%
    pause
    exit /b 1
)


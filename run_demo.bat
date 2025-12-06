@echo off
echo ============================================
echo Call Center Volume Forecasting Demo
echo ============================================
echo.

REM Check if data exists
if not exist "data\combined_data.csv" (
    echo Generating synthetic data...
    python generate_data.py
    echo.
)

echo Starting Streamlit dashboard...
echo.
echo Dashboard will open in your browser automatically.
echo Press Ctrl+C to stop the server.
echo.

python -m streamlit run dashboard.py

pause


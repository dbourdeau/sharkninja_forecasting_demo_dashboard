@echo off
echo ============================================
echo Fixing Prophet Setup
echo ============================================
echo.
echo This will install cmdstanpy and CmdStan backend.
echo This may take 5-10 minutes...
echo.

cd /d "%~dp0"

REM Install cmdstanpy
echo Step 1: Installing cmdstanpy...
python -m pip install cmdstanpy
if errorlevel 1 (
    echo ERROR: Failed to install cmdstanpy
    pause
    exit /b 1
)

echo.
echo Step 2: Installing CmdStan backend...
echo (This downloads ~100MB and may take several minutes)
echo.

python -c "import cmdstanpy; cmdstanpy.install_cmdstan()"
if errorlevel 1 (
    echo WARNING: CmdStan installation had issues
    echo You may need to install it manually
    echo See FIX_PROPHET_ERROR.md for details
) else (
    echo.
    echo ============================================
    echo Setup Complete!
    echo ============================================
    echo.
    echo You can now run the dashboard:
    echo   python -m streamlit run dashboard.py
    echo.
)

pause


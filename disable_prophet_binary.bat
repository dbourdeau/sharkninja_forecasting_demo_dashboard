@echo off
echo ============================================
echo Disabling Prophet Bundled Binary
echo ============================================
echo.
echo This will rename the bundled .bin file so Prophet
echo is forced to compile models using CmdStan instead.
echo.

python -c "import os, shutil; import prophet; p=os.path.dirname(prophet.__file__); b=os.path.join(p, 'stan_model', 'prophet_model.bin'); bk=b+'.backup'; os.path.exists(b) and not os.path.exists(bk) and shutil.move(b, bk) or None; print('✓ Done' if os.path.exists(bk) else '✓ Already disabled or not found')"

echo.
echo Done! Prophet will now compile models instead of using bundled binary.
echo.
pause


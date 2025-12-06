# Fix Prophet Error - Quick Guide

## The Problem
You're getting this error:
```
AttributeError: 'Prophet' object has no attribute 'stan_backend'
```

This happens because Prophet needs the `cmdstanpy` backend, especially on Python 3.12.

## Quick Fix (Choose One Method)

### Method 1: Automatic Fix Script (Recommended)
Run this once:
```bash
python fix_prophet_setup.py
```

This will:
- Install `cmdstanpy`
- Install `CmdStan` (takes a few minutes)

### Method 2: Manual Installation

1. **Install cmdstanpy:**
   ```bash
   pip install cmdstanpy
   ```

2. **Install CmdStan:**
   ```python
   python -c "import cmdstanpy; cmdstanpy.install_cmdstan()"
   ```

   OR manually:
   ```bash
   python -m cmdstanpy.install_cmdstan
   ```

3. **Set environment variable (optional but recommended):**
   ```bash
   set STAN_BACKEND=CMDSTANPY
   ```

### Method 3: Update All Dependencies
```bash
pip install --upgrade -r requirements.txt
pip install cmdstanpy
python -c "import cmdstanpy; cmdstanpy.install_cmdstan()"
```

## After Fixing

1. Close and restart your terminal/PowerShell
2. Navigate to the project folder:
   ```bash
   cd C:\Users\dbour\call_center_forecast_demo
   ```
3. Run the dashboard:
   ```bash
   python -m streamlit run dashboard.py
   ```

## Verify Installation

Test that Prophet works:
```python
python -c "from prophet import Prophet; print('Prophet OK!')"
```

If you get an error, try Method 2 or 3 above.

## Troubleshooting

**If CmdStan installation fails:**
- It might take 5-10 minutes (it's downloading ~100MB)
- Check your internet connection
- Try running as administrator

**If you still get errors:**
1. Uninstall old versions:
   ```bash
   pip uninstall prophet pystan cmdstanpy -y
   ```
2. Reinstall:
   ```bash
   pip install prophet cmdstanpy
   python -c "import cmdstanpy; cmdstanpy.install_cmdstan()"
   ```

## Why This Happens

Prophet uses Stan for statistical modeling. On newer Python versions (3.12+), it needs `cmdstanpy` instead of the older `pystan` backend. The `requirements.txt` has been updated to include this.


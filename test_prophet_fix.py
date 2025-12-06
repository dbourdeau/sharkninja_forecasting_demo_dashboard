"""
Test script to verify Prophet works with cmdstanpy
"""
import os
os.environ['STAN_BACKEND'] = 'CMDSTANPY'

# Import patch first
import prophet_patch

import cmdstanpy
import sys

# Set CmdStan path
default_cmdstan = os.path.join(os.path.expanduser('~'), '.cmdstan', 'cmdstan-2.37.0')
if os.path.exists(default_cmdstan):
    try:
        cmdstanpy.set_cmdstan_path(default_cmdstan)
        print(f"✓ Set CmdStan path to: {default_cmdstan}")
    except Exception as e:
        print(f"✗ Error setting path: {e}")

# Try to import and create Prophet
try:
    from prophet import Prophet
    print("✓ Prophet imported successfully")
    
    # Try to create an instance
    m = Prophet(stan_backend='CMDSTANPY')
    print("✓ Prophet instance created successfully!")
    print("✓ Everything works!")
    
except Exception as e:
    print(f"✗ Error: {e}")
    import traceback
    traceback.print_exc()


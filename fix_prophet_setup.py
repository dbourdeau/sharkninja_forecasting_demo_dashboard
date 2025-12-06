"""
Fix Prophet setup by installing and configuring cmdstanpy backend.
Run this script once to set up Prophet correctly.
"""

import sys
import subprocess

print("=" * 60)
print("Prophet Setup Fix - Installing cmdstanpy backend")
print("=" * 60)
print()

# Step 1: Install cmdstanpy
print("Step 1: Installing cmdstanpy...")
try:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "cmdstanpy==1.2.0"])
    print("✓ cmdstanpy installed successfully")
except Exception as e:
    print(f"✗ Error installing cmdstanpy: {e}")
    sys.exit(1)

print()

# Step 2: Install CmdStan
print("Step 2: Installing CmdStan (this may take a few minutes)...")
try:
    import cmdstanpy
    cmdstanpy.install_cmdstan()
    print("✓ CmdStan installed successfully")
except Exception as e:
    print(f"✗ Error installing CmdStan: {e}")
    print("You may need to install it manually. See DEPLOYMENT.md for details.")
    sys.exit(1)

print()
print("=" * 60)
print("✓ Prophet setup complete!")
print("=" * 60)
print()
print("You can now run the dashboard:")
print("  python -m streamlit run dashboard.py")
print()


"""
Fix Prophet bundled binary issue on Windows
Renames the bundled .bin file so Prophet is forced to compile
"""
import os
import shutil
from path import Path as path_module  # Use pathlib instead

try:
    import prophet
    prophet_path = os.path.dirname(prophet.__file__)
    stan_model_path = os.path.join(prophet_path, 'stan_model')
    binary_path = os.path.join(stan_model_path, 'prophet_model.bin')
    
    if os.path.exists(binary_path):
        # Rename the binary so Prophet can't use it (forces compilation)
        backup_path = binary_path + '.backup'
        if not os.path.exists(backup_path):
            print(f"Renaming bundled binary to force compilation: {binary_path}")
            shutil.move(binary_path, backup_path)
            print("✓ Bundled binary disabled - Prophet will compile models instead")
        else:
            print("✓ Bundled binary already disabled")
    else:
        print("✓ No bundled binary found - Prophet will compile models")
except Exception as e:
    print(f"Could not fix bundled binary: {e}")
    print("Prophet will attempt to compile models anyway")


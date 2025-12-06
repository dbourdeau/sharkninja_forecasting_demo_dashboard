"""
Monkey patch to fix Prophet's CmdStan path issue
This must be imported before Prophet
"""
import os
import cmdstanpy

# Set the correct CmdStan path first
default_cmdstan = os.path.join(os.path.expanduser('~'), '.cmdstan', 'cmdstan-2.37.0')
if os.path.exists(default_cmdstan) and os.path.exists(os.path.join(default_cmdstan, 'makefile')):
    try:
        cmdstanpy.set_cmdstan_path(default_cmdstan)
    except:
        pass

# Monkey patch cmdstanpy.set_cmdstan_path to prevent Prophet from overriding
_original_set_cmdstan = cmdstanpy.set_cmdstan_path
_valid_cmdstan_path = default_cmdstan

def _safe_set_cmdstan(path):
    """Only set CmdStan path if it's valid and has a makefile"""
    path_str = str(path)
    # Check if path is valid before trying to set it
    if os.path.exists(path_str) and os.path.exists(os.path.join(path_str, 'makefile')):
        try:
            return _original_set_cmdstan(path_str)
        except:
            # If setting fails, keep current path
            return None
    # If invalid path, don't change the existing valid path
    # This prevents Prophet from overriding our good path with its bundled broken one
    # Return None silently to avoid errors
    return None

# Replace the function
cmdstanpy.set_cmdstan_path = _safe_set_cmdstan

# Also patch the validation function to be more lenient
try:
    from cmdstanpy.utils.cmdstan import validate_cmdstan_path as _original_validate
    def _safe_validate(path):
        """Validate but don't raise if path is invalid - let set_cmdstan_path handle it"""
        try:
            return _original_validate(path)
        except ValueError as e:
            # If it's about the bundled Prophet path, ignore it
            if 'prophet' in str(path).lower() or 'stan_model' in str(path):
                # Return None to indicate invalid but don't raise
                return None
            raise
    # Replace validation - actually, we can't easily patch this without importing internals
    # Let's just catch the error in set_cmdstan_path
except:
    pass

# Set environment variable
os.environ['CMDSTAN'] = default_cmdstan
os.environ['STAN_BACKEND'] = 'CMDSTANPY'

# Force Prophet to compile models instead of using bundled binaries
# This is important on Windows where bundled binaries may not work
os.environ['PROPHET_FORCE_COMPILE'] = '1'

# Patch Prophet's CmdStanPyBackend to handle crashes and force recompilation
try:
    from prophet.models import CmdStanPyBackend
    import shutil
    
    _original_load_model = CmdStanPyBackend.load_model
    _original_fit = CmdStanPyBackend.fit
    
    def _patched_load_model(self):
        """Patch load_model to disable bundled binary that crashes"""
        # Preemptively disable the bundled binary before Prophet tries to load it
        try:
            import prophet
            prophet_path = os.path.dirname(prophet.__file__)
            bundled_path = os.path.join(prophet_path, 'stan_model', 'prophet_model.bin')
            backup_path = bundled_path + '.backup_crash'
            
            # Disable bundled binary if it exists
            if os.path.exists(bundled_path) and not os.path.exists(backup_path):
                try:
                    shutil.move(bundled_path, backup_path)
                except:
                    pass
        except:
            pass
        
        try:
            model = _original_load_model(self)
            # Check if using bundled binary path
            if model and hasattr(model, '_exe_file') and 'prophet\\stan_model' in str(model._exe_file).lower():
                # This is the bundled binary - disable it
                bundled_path = str(model._exe_file)
                backup_path = bundled_path + '.backup_crash'
                if os.path.exists(bundled_path) and not os.path.exists(backup_path):
                    try:
                        shutil.move(bundled_path, backup_path)
                    except:
                        pass
                # Raise error to trigger Prophet's compilation
                raise FileNotFoundError(f"Bundled binary disabled: {bundled_path}")
            return model
        except (ValueError, FileNotFoundError, OSError) as e:
            error_str = str(e).lower()
            # If binary is missing or disabled, raise to let Prophet handle compilation
            if 'no such file' in error_str or 'prophet_model.bin' in error_str or 'disabled' in error_str:
                # Prophet should handle this by compiling, but if it doesn't, we'll catch it in fit
                raise FileNotFoundError("Prophet model binary not available - compilation required")
            raise
    
    def _patched_fit(self, stan_init, dat, **kwargs):
        """Patch fit to catch crashes and recompile"""
        try:
            return _original_fit(self, stan_init, dat, **kwargs)
        except (RuntimeError, AttributeError) as e:
            error_msg = str(e).lower()
            # Check if it's the crash error (3221225781 or similar) or None model
            is_crash = 'error code' in error_msg or '3221225781' in error_msg or 'error during optimization' in error_msg
            is_none_model = "'nonetype' object has no attribute 'optimize'" in error_msg
            
            if is_crash or is_none_model:
                # Bundled binary crashed or model is None - disable binary and recompile
                bundled_path = None
                if self.model and hasattr(self.model, '_exe_file'):
                    bundled_path = str(self.model._exe_file)
                    if 'prophet\\stan_model' in bundled_path.lower() and os.path.exists(bundled_path):
                        backup_path = bundled_path + '.backup_crash'
                        try:
                            shutil.move(bundled_path, backup_path)
                        except:
                            pass
                
                # Reset model - Prophet will need to compile it
                # The issue is that Prophet doesn't automatically compile when binary is missing
                # We need to trigger Prophet's internal compilation mechanism
                self.model = None
                
                # Try to reinitialize the backend which should trigger compilation
                try:
                    # Force reload which should compile if binary is missing
                    self.model = self.load_model()
                except FileNotFoundError:
                    # Binary is disabled - Prophet needs to compile
                    # Try calling the original load_model but it will fail, then Prophet's fit will handle compilation
                    pass
                
                # If model is still None, Prophet's fit method should handle compilation
                # But if it doesn't, we need to catch it here
                if self.model is None:
                    # Prophet's fit should compile, but let's make sure by trying the fit
                    # which will trigger Prophet's compilation logic
                    try:
                        return _original_fit(self, stan_init, dat, **kwargs)
                    except AttributeError as attr_err:
                        if "'nonetype' object has no attribute 'optimize'" in str(attr_err).lower():
                            raise RuntimeError(
                                "Prophet model compilation failed. The bundled binary crashes on Windows and "
                                "Prophet cannot auto-compile. Please install Prophet from source or use a "
                                "different environment. Alternative: Use Prophet on Linux/WSL or Docker."
                            )
                        raise
                
                # Now try fit again with model
                return _original_fit(self, stan_init, dat, **kwargs)
            raise
    
    CmdStanPyBackend.load_model = _patched_load_model
    CmdStanPyBackend.fit = _patched_fit
except Exception as patch_error:
    # Patching failed - that's okay
    pass


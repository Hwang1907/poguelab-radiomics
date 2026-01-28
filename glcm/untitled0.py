import sys, subprocess, site, importlib

print("Python:", sys.executable)
print("User site:", site.getusersitepackages())
try:
    print("Site pkgs:", site.getsitepackages())
except Exception as e:
    print("Site pkgs: (n/a)", e)

# Install into USER site to avoid permissions/path issues
subprocess.check_call([sys.executable, "-m", "pip", "install", "--user", "-U", "numpy"])

# Now verify import + location
import numpy as np
print("numpy version:", np.__version__)
print("numpy file:", np.__file__)

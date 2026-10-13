import sys
import runpy
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
for p in [str(BASE_DIR), str(BASE_DIR / "app")]:
    if p not in sys.path:
        sys.path.insert(0, p)

dashboard_file = BASE_DIR / "app" / "dashboard.py"
runpy.run_path(str(dashboard_file), run_name="__main__")

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
PY311 = Path(r"C:\Users\chs\.conda\envs\py311\python.exe")


def main() -> int:
    if Path(sys.executable).resolve() != PY311.resolve() and PY311.exists():
        env = os.environ.copy()
        existing = env.get("PYTHONPATH", "")
        env["PYTHONPATH"] = str(ROOT / "src") + (os.pathsep + existing if existing else "")
        return subprocess.call([str(PY311), str(ROOT / "launch_eismaster.py")], cwd=str(ROOT), env=env)

    from eismaster.app import main as app_main

    return app_main()


if __name__ == "__main__":
    raise SystemExit(main())

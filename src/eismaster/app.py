from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

from PySide6.QtWidgets import QApplication

from eismaster.ui.theme import apply_nanobanana_theme


PY311 = Path(r"C:\Users\chs\.conda\envs\py311\python.exe")


def _ensure_py311() -> int | None:
    if sys.version_info[:2] == (3, 11):
        return None
    if not PY311.exists():
        return None

    env = os.environ.copy()
    root = Path(__file__).resolve().parents[2]
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = str(root / "src") + (os.pathsep + existing if existing else "")
    return subprocess.call([str(PY311), "-m", "eismaster"], cwd=str(root), env=env)


def main() -> int:
    redirected = _ensure_py311()
    if redirected is not None:
        return redirected

    from eismaster.ui.main_window import MainWindow

    app = QApplication.instance() or QApplication(sys.argv)
    apply_nanobanana_theme(app)
    window = MainWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())

from __future__ import annotations

import os
import json
import subprocess
import sys
import time
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

    if "--fit-smoke" in sys.argv:
        return _run_fit_smoke(sys.argv)

    from eismaster.ui.main_window import MainWindow

    app = QApplication.instance() or QApplication(sys.argv)
    apply_nanobanana_theme(app)
    window = MainWindow()
    window.show()
    return app.exec()


def _run_fit_smoke(argv: list[str]) -> int:
    try:
        index = argv.index("--fit-smoke")
        input_path = Path(argv[index + 1])
    except (ValueError, IndexError):
        return 2

    report_path = None
    if "--fit-smoke-report" in argv:
        try:
            report_path = Path(argv[argv.index("--fit-smoke-report") + 1])
        except IndexError:
            return 2

    from eismaster.analysis.fitting import fit_spectrum
    from eismaster.io import load_spectrum

    started = time.time()
    payload: dict[str, object]
    try:
        spectrum = load_spectrum(input_path)
        fit = fit_spectrum(spectrum, "zview_segmented_rq_rwo")
        payload = {
            "status": fit.status,
            "message": fit.message,
            "elapsed_s": round(time.time() - started, 3),
            "has_curve": fit.predicted_real_ohm is not None and fit.predicted_imag_ohm is not None,
            "parameters": sorted(fit.parameters),
        }
        code = 0 if fit.status in {"ok", "warn"} and payload["has_curve"] else 1
    except Exception as exc:
        payload = {"status": "error", "message": str(exc), "elapsed_s": round(time.time() - started, 3)}
        code = 1

    if report_path is not None:
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    else:
        print(json.dumps(payload, ensure_ascii=False))
    return code


if __name__ == "__main__":
    raise SystemExit(main())

# EISMaster Optimization Baseline

- Baseline commit: `d72afc93265e13c25f4ec4da8346a7f3d1ce160a`
- Git branch: `main...origin/main`
- Python contract: `>=3.11`
- Formal implementation: `src/eismaster/`
- Comparison implementation: `rewrite/`
- Protected modified files:
  - `EISMaster.spec`
  - `src/eismaster/exporters.py`
  - `src/eismaster/ui/main_window.py`
  - `tests/test_exports.py`

## Real git state

`git status --short --branch`

```text
## main...origin/main
 M EISMaster.spec
 M src/eismaster/exporters.py
 M src/eismaster/ui/main_window.py
 M tests/test_exports.py
?? bilibili_article.md
?? bilibili_article_v2.md
?? dist_rebuild4/
?? docs/images/5.png
?? docs/images/6.png
?? docs/superpowers/
?? rewrite/
?? rust-rewrite-plan/
?? tools/
```

`git diff --numstat`

```text
3	0	EISMaster.spec
223	32	src/eismaster/exporters.py
34	1	src/eismaster/ui/main_window.py
111	2	tests/test_exports.py
```

`git log -5 --oneline --decorate`

```text
d72afc9 (HEAD -> main, origin/main, origin/HEAD) Remove stray export button from fit tab
3a9d041 Polish batch summary layout
7d0354d Restore SciPy release fitting backend
0ce1b3c Fix release fitting dependencies
b652c8e Fix slim release runtime dependencies
```

## Read-only checks

- `python --version` -> `Python 3.7.1`
- `py -3.11 --version` -> `Python 3.11.5`
- `py -3.11 -c "import qfluentwidgets"` -> `ModuleNotFoundError`
- repository-root sample file check for `Ag_EIS_OCV.txt` -> not present at the repo root

## Baseline test command

```powershell
$env:PYTHONPATH='src;.'
py -3.11 -m unittest discover -s tests -q
```

This command is part of the baseline, but it is currently blocked and should not be described as passed. The known blockers are:

- repository-root sample files are missing
- `qfluentwidgets` is not installed in the Python 3.11 interpreter

## Known baseline blockers

- repository-root sample files are missing
- default `python` is not Python 3.11
- `qfluentwidgets` is not installed in the Python 3.11 interpreter

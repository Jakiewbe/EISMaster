from pathlib import Path
import sys

from PyInstaller.utils.hooks import collect_data_files, collect_submodules


ROOT = Path.cwd()
ENV_ROOT = Path(sys.executable).resolve().parent
LIB_BIN = ENV_ROOT / "Library" / "bin"

datas = collect_data_files("eismaster")
for folder_name in ("matlab_bridge", "matlab-DRTtools-local"):
    folder = ROOT / folder_name
    if folder.exists():
        datas.append((str(folder), folder_name))

hiddenimports = []
hiddenimports += collect_submodules("qfluentwidgets")
hiddenimports += collect_submodules("pyqtgraph")

binaries = []
for dll_name in (
    "ffi.dll",
    "ffi-7.dll",
    "ffi-8.dll",
    "libbz2.dll",
    "libcrypto-3-x64.dll",
    "libexpat.dll",
    "liblzma.dll",
    "libssl-3-x64.dll",
    "expat.dll",
):
    dll_path = LIB_BIN / dll_name
    if dll_path.exists():
        binaries.append((str(dll_path), "."))


a = Analysis(
    ["launch_eismaster.py"],
    pathex=[str(ROOT / "src"), str(ROOT)],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="EISMaster",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name="EISMaster",
)

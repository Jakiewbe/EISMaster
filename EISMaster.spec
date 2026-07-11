from pathlib import Path
import sys

from PyInstaller.utils.hooks import collect_data_files, collect_submodules


ROOT = Path.cwd()
ENV_ROOT = Path(sys.executable).resolve().parent
LIB_BIN = ENV_ROOT / "Library" / "bin"

datas = collect_data_files("eismaster")
matlab_bridge = ROOT / "matlab_bridge"
if matlab_bridge.exists():
    datas.append((str(matlab_bridge), "matlab_bridge"))
drttools_dir = ROOT / "matlab-DRTtools-local"
if not drttools_dir.is_dir():
    raise FileNotFoundError(f"Required DRTtools resource is missing: {drttools_dir}")
datas.append((str(drttools_dir), "matlab-DRTtools-local"))

hiddenimports = collect_submodules("qfluentwidgets") + collect_submodules(
    "pyqtgraph",
    filter=lambda name: not (
        name.startswith("pyqtgraph.examples")
        or name.startswith("pyqtgraph.opengl")
    ),
)

binaries = []

# 排除不需要的 Qt DLL（视频、QML、OpenGL软件渲染器等）
qt_excludes = {
    "opengl32sw.dll",       # 软件 OpenGL 渲染器 (20MB)
    "avcodec-61.dll",       # FFmpeg 视频编解码 (13MB)
    "avformat-61.dll",      # FFmpeg 视频格式 (2.5MB)
    "avutil-59.dll",        # FFmpeg 工具 (1MB)
    "swscale-8.dll",        # FFmpeg 缩放 (0.7MB)
    "Qt6Quick.dll",         # QML (6MB)
    "Qt6Qml.dll",           # QML (5MB)
    "Qt6Pdf.dll",           # PDF (4MB)
    "Qt6QmlModels.dll",     # QML 模型
    "Qt6QmlWorkerScript.dll",
    "Qt6VirtualKeyboard.dll",
}

# PySide6 DLL 排除（PyInstaller hooks 自动收集的）
pyside6_dll_excludes = {
    "opengl32sw.dll",
    "Qt6Quick.dll",
    "Qt6Qml.dll",
    "Qt6Pdf.dll",
    "Qt6QmlModels.dll",
}

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
    if dll_name in qt_excludes:
        continue
    dll_path = LIB_BIN / dll_name
    if dll_path.exists():
        binaries.append((str(dll_path), "."))


a = Analysis(
    ["src/eismaster/app.py"],
    pathex=[str(ROOT / "src"), str(ROOT)],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        # 不需要的大型库
        "pyarrow",
        "PIL",
        "Pillow",
        "matplotlib",
        "statsmodels",
        "tkinter",
        "PyQt5",
        "PyQt6",
        "IPython",
        "jupyter",
        "notebook",
        "test",
        "tests",
        "setuptools",
        "pip",
        "wheel",
        "pkg_resources",  # 排除 pkg_resources 以避免 jaraco 缺失错误
        # pandas 不需要的子模块
        "pandas.tests",
        # numpy 不需要的子模块
        "numpy.tests",
        # pyqtgraph 不需要的子模块
        "pyqtgraph.opengl",
        # PySide6 不需要的子模块
        "PySide6.QtQuick",
        "PySide6.QtQml",
        "PySide6.QtPdf",
        "PySide6.Qt3DCore",
        "PySide6.Qt3DRender",
        "PySide6.Qt3DInput",
        "PySide6.Qt3DLogic",
        "PySide6.Qt3DExtras",
        "PySide6.Qt3DAnimation",
        "PySide6.QtBluetooth",
        "PySide6.QtCharts",
        "PySide6.QtDataVisualization",
        "PySide6.QtHelp",
        "PySide6.QtHttpServer",
        "PySide6.QtLocation",
        "PySide6.QtNfc",
        "PySide6.QtPositioning",
        "PySide6.QtRemoteObjects",
        "PySide6.QtScxml",
        "PySide6.QtSensors",
        "PySide6.QtSerialBus",
        "PySide6.QtSerialPort",
        "PySide6.QtSql",
        "PySide6.QtStateMachine",
        "PySide6.QtTextToSpeech",
        "PySide6.QtUiTools",
        "PySide6.QtWebChannel",
        "PySide6.QtWebEngine",
        "PySide6.QtWebEngineCore",
        "PySide6.QtWebEngineWidgets",
        "PySide6.QtWebSockets",
    ],
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

# 过滤掉不需要的 PySide6 DLL
filtered_binaries = []
for dest, src, typ in a.binaries:
    filename = Path(dest).name
    if filename.lower() in {n.lower() for n in pyside6_dll_excludes}:
        continue
    filtered_binaries.append((dest, src, typ))

coll = COLLECT(
    exe,
    filtered_binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name="EISMaster",
)

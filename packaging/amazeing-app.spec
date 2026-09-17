# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller spec for the standalone aMAZEing maze application.

Build from the repository root (see packaging/build_windows.ps1):

    pyinstaller --noconfirm --clean packaging/amazeing-app.spec

Produces dist/amazeing-app/ containing amazeing-app.exe plus its libraries.
The same executable is used for the child processes the app launches
(``amazeing-app.exe --entry auditory ...``), so every command-line tool is
inside the bundle as well.
"""

import os

from PyInstaller.utils.hooks import collect_data_files, collect_submodules

ROOT = os.path.abspath(os.path.join(SPECPATH, ".."))
SRC = os.path.join(ROOT, "src")

hidden = (
    collect_submodules("amazeing")
    + [
        "sounddevice",
        "soundfile",
        "serial",
        "serial.tools.list_ports",
        "yaml",
        "cv2",
        "imageio_ffmpeg",
        "matplotlib.backends.backend_qtagg",
        "matplotlib.backends.backend_agg",
        "scipy.signal",
        "scipy.interpolate",
        "scipy.stats",
    ]
)

datas = (
    collect_data_files("amazeing")          # speaker calibration CSV, tactile CSVs, docs
    + collect_data_files("imageio_ffmpeg")  # bundled ffmpeg for per-trial video segments
)

a = Analysis(
    [os.path.join(SRC, "amazeing", "app", "__main__.py")],
    pathex=[SRC],
    binaries=[],
    datas=datas,
    hiddenimports=hidden,
    hookspath=[],
    runtime_hooks=[],
    # tkinter is NOT excluded: the tactile session and its post-processing open
    # file dialogs with it, so excluding it breaks those in the packaged app.
    excludes=["PyQt5", "PyQt6", "IPython", "notebook", "jupyter", "pytest",
              "numba", "llvmlite", "plotly", "statsmodels", "seaborn", "PIL.ImageQt"],
    noarchive=False,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="amazeing-app",
    debug=False,
    strip=False,
    upx=False,
    console=False,          # no terminal window; child-process output is shown in the app
    icon=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    name="amazeing-app",
)

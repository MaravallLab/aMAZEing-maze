# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller spec for the standalone aMAZEing maze application.

Build from the repository root (see packaging/build_windows.ps1):

    pyinstaller --noconfirm --clean packaging/amazeing-app.spec

Produces dist/amazeing-app/ containing amazeing-app.exe plus its libraries.
The same executable is used for the child processes the app launches
(``amazeing-app.exe --entry auditory ...``), so every command-line tool is
inside the bundle as well.

The thesis research scripts in analysis/auditory are bundled as data, not
compiled in: the Analysis tab runs them by path (``amazeing-app.exe --script
<path> ...``), so the thesis can cite the same files. PyInstaller cannot see
what they import, so their imports are read here and added as hidden imports,
with statsmodels, seaborn and plotly whole. tdt, which reads the photometry
tanks, is bundled, so the photometry scripts run from the packaged app. PyMC
(the Bayesian fits of the value model) is left out; the app says those scripts
need a source install.
"""

import ast
import glob
import os
import sys

from PyInstaller.utils.hooks import collect_data_files, collect_submodules

ROOT = os.path.abspath(os.path.join(SPECPATH, ".."))
SRC = os.path.join(ROOT, "src")
AUDITORY = os.path.join(ROOT, "analysis", "auditory")

sys.path.insert(0, SRC)
from amazeing.app.pipelines import bundle_files  # noqa: E402

# Imported by some research scripts but deliberately not bundled (pymer4 only
# by the earlier batch pipeline, which the app does not offer).
NOT_BUNDLED = {"pymc", "arviz", "pytensor", "kaleido", "IPython", "pymer4"}
# Bundled whole: their submodules are imported lazily.
WHOLE = ("statsmodels", "patsy", "seaborn", "plotly")

SCRIPT_FILES = bundle_files(AUDITORY)


def _script_datas(files):
    out = []
    for f in files:
        rel = os.path.relpath(os.path.dirname(f), AUDITORY)
        dest = "analysis/auditory" if rel == "." else "analysis/auditory/" + rel.replace(os.sep, "/")
        out.append((f, dest))
    return out


def _script_imports(files):
    """Every module the research scripts import, except their own siblings."""
    local = {os.path.splitext(os.path.basename(f))[0] for f in files} | {"model_validation"}
    names = set()
    for f in files:
        if not f.endswith(".py"):
            continue
        with open(f, encoding="utf-8") as fh:
            tree = ast.parse(fh.read(), f)
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                mods = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                mods = [node.module]
            else:
                continue
            for m in mods:
                top = m.split(".")[0]
                if top not in local and top not in NOT_BUNDLED:
                    names.add(m)
    return sorted(names)


def _not_tests(name):
    return ".tests" not in name and ".testing" not in name

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
    # savefig loads the backend of a .pdf, .svg or .eps file only when asked
    + ["matplotlib.backends.backend_pdf", "matplotlib.backends.backend_svg",
       "matplotlib.backends.backend_ps", "matplotlib.backends.backend_pgf"]
    + _script_imports(SCRIPT_FILES)
    + [m for pkg in WHOLE for m in collect_submodules(pkg, filter=_not_tests)]
)

datas = (
    collect_data_files("amazeing")          # speaker calibration CSV, tactile CSVs, docs
    + collect_data_files("imageio_ffmpeg")  # bundled ffmpeg for per-trial video segments
    + collect_data_files("plotly")          # plotly's figure schema, read at import
    + _script_datas(SCRIPT_FILES)           # the research scripts and their R models
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
              "numba", "llvmlite", "PIL.ImageQt", *sorted(NOT_BUNDLED - {"IPython"})],
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

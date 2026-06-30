# Configuration file for the Sphinx documentation builder.
#
# Full reference:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys

# -- Path setup --------------------------------------------------------------
# The auditory experiment code uses *flat* imports (``from config import ...``,
# ``from modules.audio import ...``, ``from grammar_stimuli import ...``) and is
# run with ``src/auditory`` as the working directory rather than installed as a
# package. We therefore put that directory on ``sys.path`` so autodoc can import
# those modules, and add ``analysis/auditory`` so the ``model_validation``
# package is importable. ``src/auditory`` is inserted last so it takes priority
# for the bare ``config`` module name.
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(_REPO_ROOT, "analysis", "auditory"))
sys.path.insert(0, os.path.join(_REPO_ROOT, "src", "auditory"))

# -- Project information -----------------------------------------------------

project = "The aMAZEing maze"
copyright = "2024-2026, Andre Maia Chagas, Alejandra Carriero, Shahd Al Balushi, Moira Eley, Miguel Maravall"
author = "Andre Maia Chagas, Alejandra Carriero, Shahd Al Balushi, Moira Eley, Miguel Maravall"

# -- General configuration ---------------------------------------------------

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "sphinx.ext.duration",
    # ``myst_nb`` provides the MyST Markdown parser *and* Jupyter notebook
    # support. Do NOT also enable ``myst_parser`` -- myst_nb already includes it
    # and registering both raises an error.
    "myst_nb",
    "sphinx_design",
]

templates_path = ["_templates"]
exclude_patterns = []

# -- MyST (Markdown) ---------------------------------------------------------

myst_enable_extensions = [
    "colon_fence",
    "linkify",
    "substitution",
    "html_admonition",
    "html_image",
    "deflist",
]

# -- MyST-NB (notebooks) -----------------------------------------------------
# No notebooks are wired into the toctree yet. When some are added, they render
# using their *stored* outputs -- nothing is executed at build time, so the
# build stays fast and never needs experiment data or hardware.
nb_execution_mode = "off"

# -- autodoc / autosummary ---------------------------------------------------

autosummary_generate = True

autodoc_default_options = {
    "members": True,
    "undoc-members": True,
    "show-inheritance": True,
}
autodoc_member_order = "bysource"
autodoc_typehints = "description"

# Hardware / audio / heavy optional dependencies are not importable in a docs
# build environment (no camera, serial port, audio backend, or PyMC stack), so
# mock them. They are only ever used inside functions, never at import time, so
# mocking is sufficient for autodoc to import every module.
autodoc_mock_imports = [
    "cv2",
    "serial",
    "sounddevice",
    "soundfile",
    "pymc",
    "arviz",
]

# -- napoleon (NumPy-style docstrings) ---------------------------------------

napoleon_numpy_docstring = True
napoleon_google_docstring = True
napoleon_include_init_with_doc = False
napoleon_use_rtype = True

# -- intersphinx -------------------------------------------------------------
# Cross-references to external docs. Inventory fetches require network access;
# if offline these degrade to warnings and the build still succeeds.
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "pandas": ("https://pandas.pydata.org/docs/", None),
}

# -- HTML output -------------------------------------------------------------

html_theme = "shibuya"
html_static_path = ["_static"]
html_title = "The aMAZEing maze"
html_theme_options = {
    "github_url": "https://github.com/MaravallLab/aMAZEing-maze",
}

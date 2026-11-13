# SPDX-FileCopyrightText: 2026 Qoro Quantum
# SPDX-License-Identifier: GPL-3.0-or-later

"""Sphinx configuration for the qiskit-maestro documentation.

Build from the repository root:

    pip install -e ".[docs]"
    sphinx-build -W --keep-going -b html docs docs/_build/html

The API reference imports ``qiskit_maestro``, so the package must be installed.
"""

import pathlib
import shutil
from importlib.metadata import version as installed_version

DOCS = pathlib.Path(__file__).resolve().parent
ROOT = DOCS.parent

project = "qiskit-maestro"
author = "Qoro Quantum"
copyright = "2026, Qoro Quantum"
release = installed_version("qiskit-maestro")
version = release

extensions = [
    "myst_nb",
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
    "sphinx_copybutton",
    "qiskit_sphinx_theme",
]

# Notes for maintainers that live in docs/ but aren't part of the site.
exclude_patterns = [
    "_build",
    "jupyter_execute",
    "maestro-api-notes.md",
    "qiskit-ecosystem-submission.md",
]

# The example notebook lives in examples/ with the scripts, outside the source
# tree, so copy it in. It's shown with the outputs saved in it rather than run
# again: it takes minutes, and its text quotes those outputs.
shutil.copy2(ROOT / "examples" / "vqe_aer_vs_maestro.ipynb", DOCS / "examples")
nb_execution_mode = "off"

myst_enable_extensions = ["colon_fence", "dollarmath"]
myst_heading_anchors = 3

# API reference. Docstrings are Google style. api.rst lists the members to show,
# leaving out the ones every Qiskit backend and job has.
autodoc_default_options = {"member-order": "bysource"}
autodoc_typehints = "description"
autodoc_typehints_description_target = "documented_params"
add_module_names = False
napoleon_google_docstring = True
napoleon_numpy_docstring = False

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable", None),
    "qiskit": ("https://quantum.cloud.ibm.com/docs/api/qiskit", None),
    "qiskit_aer": ("https://qiskit.github.io/qiskit-aer", None),
    "maestro": ("https://qoroquantum.github.io/maestro", None),
}

html_theme = "qiskit-ecosystem"
html_title = f"{project} {release}"
html_favicon = "_static/favicon.ico"
html_theme_options = {"sidebar_qiskit_ecosystem_member": True}
# toc-scroll-room.js keeps the "On this page" highlight on the section clicked.
html_static_path = ["_static"]
html_js_files = ["toc-scroll-room.js"]

# GitHub renders README anchors client-side, so linkcheck can't see them.
linkcheck_anchors_ignore_for_url = [r"https://github\.com/"]

"""Configuration file for the Sphinx documentation builder."""

# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html
import sys
from pathlib import Path

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "autocv"
copyright = "2024, Zach"  # noqa: A001
author = "Zach"
release = "1.0.0"

sys.path.insert(0, str(Path(__file__).parents[1].resolve()))


# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
    "sphinx.ext.viewcode",
    "sphinx_copybutton"
]
html_theme = "furo"

autodoc_member_order = "bysource"
toc_object_entries_show_parents = "hide"

# Links used for cross-referencing stuff in other documentation
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "pillow": ("https://pillow.readthedocs.io/en/stable/", None),
}


templates_path = ["_templates"]

source_suffix = ".rst"

# The master toctree document.
master_doc = "index"

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
exclude_patterns = ["build"]

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output
html_theme_options = {
    "source_repository": "https://github.com/mohanz1/autocv",
    "source_branch": "main",
    "source_directory": "source/",
}

html_static_path = ["_static"]

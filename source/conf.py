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


# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

sys.path.insert(0, str(Path("../").resolve()))

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx_immaterial",
]
html_theme = "sphinx_immaterial"

toc_object_entries_show_parents = "hide"


templates_path = ["_templates"]
exclude_patterns = []

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme_options = {
    # Set the repo location to get a badge with stats
    "repo_url": "https://github.com/mohanz1/autocv",
    "repo_name": "autocv",
    "palette": {"scheme": "slate"},
    "toc_title": "AutoCV",
    "version_dropdown": True,
    "version_info": [
        {"version": "1.0", "title": "1.0", "aliases": ["latest"]},
    ],
}

html_static_path = ["_static"]

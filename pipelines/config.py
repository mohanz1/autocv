from __future__ import annotations

import pathlib

# Packaging
MAIN_PACKAGE = "autocv"

# Directories
ARTIFACT_DIRECTORY = "public"
DOCUMENTATION_DIRECTORY = "docs"

# Linting and test configs
PYPROJECT_TOML = "pyproject.toml"

DOCUMENTATION_OUTPUT_PATH = pathlib.Path(ARTIFACT_DIRECTORY, "docs")


# Reformatting paths
REFORMATTING_FILE_EXTS = (
    ".py",
    ".pyx",
    ".pyi",
    ".cpp",
    ".cxx",
    ".hpp",
    ".hxx",
    ".h",
    ".yml",
    ".yaml",
    ".html",
    ".htm",
    ".js",
    ".json",
    ".toml",
    ".ini",
    ".cfg",
    ".css",
    ".md",
    ".dockerfile",
    "Dockerfile",
    ".editorconfig",
    ".gitattributes",
    ".json",
    ".gitignore",
    ".dockerignore",
    ".txt",
    ".sh",
    ".bat",
    ".ps1",
    ".rb",
    ".pl",
)
PYTHON_REFORMATTING_PATHS = (MAIN_PACKAGE, "pipelines", "noxfile.py")
FULL_REFORMATTING_PATHS = (
    *PYTHON_REFORMATTING_PATHS,
    *(f for f in pathlib.Path.cwd().glob("*") if f.is_file() and f.suffix.endswith(REFORMATTING_FILE_EXTS)),
    ".github",
    "docs",
)

"""Package metadata.

This module stores build- and release-related metadata and is intentionally
dependency-free. Values are re-exported from :mod:`autocv` for convenience.
"""

from __future__ import annotations

from typing import Final

__all__ = (
    "__author__",
    "__ci__",
    "__copyright__",
    "__docs__",
    "__email__",
    "__git_sha1__",
    "__issue_tracker__",
    "__license__",
    "__maintainer__",
    "__url__",
    "__version__",
)

__author__: Final[str] = "devzach"
__maintainer__: Final[str] = "devzach"
__ci__: Final[str] = "https://github.com/autocv/actions"
__copyright__: Final[str] = "2023-present, devzach"
__docs__: Final[str] = "https://mohanz1.github.io/autocv/"
__email__: Final[str] = "dev.zach@gmail.com"
__issue_tracker__: Final[str] = "https://github.com/autocv/issues"
__license__: Final[str] = "MIT"
__url__: Final[str] = "https://github.com/autocv"
__version__: Final[str] = "1.0"
__git_sha1__: Final[str] = "HEAD"

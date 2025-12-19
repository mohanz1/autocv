"""Windows-specific automation bootstrap and metadata exports.

Importing this package configures DPI awareness for the current process and
re-exports the high-level automation entry points alongside project metadata.

Note:
    Only Windows builds are supported; importing on other platforms raises a
    ``RuntimeError``.
"""

from __future__ import annotations

import ctypes
import platform

_WINDOWS_RELEASES_WITH_SHCORE = {"10", "11"}


def _configure_process_dpi_awareness() -> None:
    """Configure process DPI awareness for clearer pixel-accurate capture."""
    if platform.release() in _WINDOWS_RELEASES_WITH_SHCORE:
        ctypes.windll.shcore.SetProcessDpiAwareness(2)
    else:
        ctypes.windll.user32.SetProcessDPIAware()


if platform.system() != "Windows":
    msg = "Only Windows platform is currently supported."
    raise RuntimeError(msg)

_configure_process_dpi_awareness()

from ._about import (  # noqa: E402
    __author__,
    __ci__,
    __copyright__,
    __docs__,
    __email__,
    __git_sha1__,
    __issue_tracker__,
    __license__,
    __maintainer__,
    __url__,
    __version__,
)
from .auto_color_aid import AutoColorAid  # noqa: E402
from .autocv import AutoCV  # noqa: E402

__all__ = (
    "AutoCV",
    "AutoColorAid",
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

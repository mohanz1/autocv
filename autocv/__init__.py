"""Windows-specific automation bootstrap and metadata exports.

Importing this package configures DPI awareness for the current process and
re-exports the high-level automation entry points alongside project metadata.

Note:
    Only Windows builds are supported; importing on other platforms raises a
    ``RuntimeError``.
"""

from __future__ import annotations

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


import ctypes
import platform

if platform.system() != "Windows":
    msg = "Only Windows platform is currently supported."
    raise RuntimeError(msg)

if platform.release() in {"10", "11"}:
    ctypes.windll.shcore.SetProcessDpiAwareness(2)
else:
    ctypes.windll.user32.SetProcessDPIAware()

from .autocv import AutoCV  # noqa: I001
from .auto_color_aid import AutoColorAid
from autocv._about import __author__
from autocv._about import __ci__
from autocv._about import __copyright__
from autocv._about import __docs__
from autocv._about import __email__
from autocv._about import __git_sha1__
from autocv._about import __issue_tracker__
from autocv._about import __license__
from autocv._about import __maintainer__
from autocv._about import __url__
from autocv._about import __version__

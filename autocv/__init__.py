"""The `autocv` package provides a suite of tools to automate and perform computer vision tasks on the Windows platform.

This package includes capabilities for screen capture, image processing, color picking, and simulation of user input,
all designed to facilitate the creation of automation scripts and applications that interact with GUI elements. The
AutoCV class serves as the main interface for these functionalities.

Note:
    The package is specifically designed for the Windows platform and may not function correctly on other operating
        systems.
    DPI awareness is set according to the detected Windows version to ensure consistent behavior across different
        screen resolutions and scalings.

Modules:
    autocv: Contains the AutoCV class, providing high-level methods for automation and computer vision tasks.
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

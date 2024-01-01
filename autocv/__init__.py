"""The `autocv` package provides a suite of tools to automate and perform computer vision tasks on the Windows platform.

This package includes capabilities for screen capture, image processing, color picking, and simulation of user input,
all designed to facilitate the creation of automation scripts and applications that interact with GUI elements. The
AutoCV class serves as the main interface for these functionalities.

Note:
    - The package is specifically designed for the Windows platform and may not function correctly on other operating
    systems.
    - DPI awareness is set according to the detected Windows version to ensure consistent behavior across different
    screen resolutions and scalings.

Modules:
    - autocv: Contains the AutoCV class, providing high-level methods for automation and computer vision tasks.
"""

import ctypes
import platform

__all__ = ("AutoCV",)

if platform.system() != "Windows":
    raise RuntimeError("Only Windows platform is currently supported.")  # noqa: TRY003

if platform.release() in {"10", "11"}:
    ctypes.windll.shcore.SetProcessDpiAwareness(2)
else:
    ctypes.windll.user32.SetProcessDPIAware()

from .autocv import AutoCV

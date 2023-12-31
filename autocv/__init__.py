import ctypes
import platform

__all__ = ("AutoCV",)

if platform.system() != "Windows":
    raise RuntimeError("Only Windows platform is currently supported.")

if platform.release() in ["10", "11"]:
    ctypes.windll.shcore.SetProcessDpiAwareness(2)
else:
    ctypes.windll.user32.SetProcessDPIAware()

from .autocv import AutoCV

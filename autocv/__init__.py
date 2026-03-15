"""Package entrypoint and metadata exports for AutoCV.

Importing :mod:`autocv` keeps heavy runtime dependencies lazy:

- ``AutoCV`` and ``AutoColorAid`` are imported on first access.
- DPI awareness is configured opportunistically on Windows only.
"""

from __future__ import annotations

import ctypes
import platform
from importlib import import_module
from typing import TYPE_CHECKING, Any, Final

if __package__ in {None, ""} and platform.system() != "Windows":
    msg = "Only Windows platform is currently supported."
    raise RuntimeError(msg)

from . import _about

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

_WINDOWS_RELEASES_WITH_SHCORE: Final[set[str]] = {"10", "11"}
_ABOUT_EXPORTS: Final[set[str]] = {
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
}


if TYPE_CHECKING:
    from .auto_color_aid import AutoColorAid
    from .autocv import AutoCV


def _configure_process_dpi_awareness() -> None:
    """Configure process DPI awareness when running on Windows."""
    if platform.system() != "Windows":
        return
    windll: Any | None = getattr(ctypes, "windll", None)
    if windll is None:
        return
    if platform.release() in _WINDOWS_RELEASES_WITH_SHCORE:
        windll.shcore.SetProcessDpiAwareness(2)
        return
    windll.user32.SetProcessDPIAware()


try:
    _configure_process_dpi_awareness()
except (AttributeError, OSError):  # pragma: no cover
    # DPI setup is best-effort only; keep package import robust.
    pass


def __getattr__(name: str) -> object:
    """Lazily resolve heavyweight exports and metadata symbols."""
    if name == "AutoCV":
        return import_module(".autocv", __name__).AutoCV
    if name == "AutoColorAid":
        return import_module(".auto_color_aid", __name__).AutoColorAid
    if name in _ABOUT_EXPORTS:
        return getattr(_about, name)
    msg = f"module {__name__!r} has no attribute {name!r}"
    raise AttributeError(msg)

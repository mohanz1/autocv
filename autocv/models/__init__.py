"""Data structures and domain exceptions used by AutoCV."""

from __future__ import annotations

__all__ = [
    "FilterSettings",
    "InvalidHandleError",
    "InvalidImageError",
    "InvalidLengthError",
]

from .exceptions import InvalidHandleError, InvalidImageError, InvalidLengthError
from .filter_settings import FilterSettings

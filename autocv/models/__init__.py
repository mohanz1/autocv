"""Data structures and domain exceptions used by AutoCV.

This package centralises small, shared models (such as filter parameters) and
project-specific exception types that are raised throughout the codebase.
"""

from __future__ import annotations

__all__ = (
    "FilterSettings",
    "InvalidHandleError",
    "InvalidImageError",
    "InvalidLengthError",
)

from .exceptions import InvalidHandleError, InvalidImageError, InvalidLengthError
from .filter_settings import FilterSettings

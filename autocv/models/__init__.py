"""Dataclasses and exceptions shared across AutoCV."""

from __future__ import annotations

__all__ = (
    "FilterSettings",
    "InvalidHandleError",
    "InvalidImageError",
    "InvalidLengthError",
)

from .exceptions import InvalidHandleError, InvalidImageError, InvalidLengthError
from .filter_settings import FilterSettings

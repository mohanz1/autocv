"""The models package of the AutoCV library.

Provides classes and utilities for representing and manipulating various geometric shapes and entities such as points,
rectangles, circles, colors, and more complex structures. It also includes exceptions specific to the AutoCV domain and
settings configurations for image processing filters.
"""

__all__ = [
    "FilterSettings",
    "InvalidHandleError",
    "InvalidImageError",
    "InvalidLengthError",
]

from .exceptions import (
    InvalidHandleError,
    InvalidImageError,
    InvalidLengthError,
)
from .filter_settings import FilterSettings

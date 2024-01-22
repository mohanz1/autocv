"""The models package of the AutoCV library.

Provides classes and utilities for representing and manipulating various geometric shapes and entities such as points,
rectangles, circles, colors, and more complex structures. It also includes exceptions specific to the AutoCV domain and
settings configurations for image processing filters.
"""

__all__ = [
    "Color",
    "ColorWithPoint",
    "Contour",
    "FilterSettings",
    "InvalidHandleError",
    "InvalidImageError",
    "InvalidLengthError",
    "OrderBy",
    "Point",
    "Rectangle",
    "ShapeList",
    "TextInfo",
]

from .color import Color
from .color_with_point import ColorWithPoint
from .contour import Contour
from .exceptions import (
    InvalidHandleError,
    InvalidImageError,
    InvalidLengthError,
)
from .filter_settings import FilterSettings
from .point import Point
from .rectangle import Rectangle
from .shape_list import OrderBy, ShapeList
from .text_info import TextInfo

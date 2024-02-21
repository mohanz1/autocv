"""This module provides a Rectangle class for representing and manipulating rectangular areas.

The class includes methods for calculating properties like the right and bottom edges, area, center, and overlap with
other rectangles. It also includes class methods for creating Rectangle instances from various forms of data, such as
NumPy arrays, dictionary representations, dimensions, or explicit coordinates.
"""

from __future__ import annotations

__all__ = ("Rectangle",)

import random
from typing import TYPE_CHECKING, NamedTuple

from typing_extensions import Self

from autocv.models.exceptions import InvalidLengthError
from autocv.models.point import Point

if TYPE_CHECKING:
    from collections.abc import Sequence

    import numpy as np
    import numpy.typing as npt


MAX_SIDES = 4


class Rectangle(NamedTuple):
    """A class to represent a rectangle.

    Attributes:
    - left (int): The x-coordinate of the left edge of the rectangle.
    - top (int): The y-coordinate of the top edge of the rectangle.
    - width (int): The width of the rectangle.
    - height (int): The height of the rectangle.
    """

    left: int
    top: int
    width: int
    height: int

    @property
    def right(self: Self) -> int:
        """Calculate the right edge of the rectangle.

        Returns:
            int: The x-coordinate of the right edge of the rectangle.
        """
        return self.left + self.width

    @property
    def bottom(self: Self) -> int:
        """Calculate the bottom edge of the rectangle.

        Returns:
            int: The y-coordinate of the bottom edge of the rectangle.
        """
        return self.top + self.height

    @classmethod
    def from_ndarray(cls: type[Rectangle], data: npt.NDArray[np.uintp]) -> Rectangle:
        """Create a Rectangle object from a NumPy array.

        Args:
            data (npt.NDArray[np.uintp]): A NumPy array containing the rectangle's left, top, width, and height.

        Returns:
            Rectangle: A new instance of Rectangle initialized with the data from the NumPy array.
        """
        return cls(*data)

    def area(self: Self) -> int:
        """Returns the area of the rectangle.

        Returns:
            int: The area of the rectangle.
        """
        return self.width * self.height

    def get_overlap(self: Self, other: tuple[int, int, int, int]) -> Rectangle | None:
        """Get the overlap between this rectangle and another rectangle.

        Args:
            other (Tuple[int, int, int, int]): The rectangle to get the overlap with.

        Returns:
            Optional[autocv.models.Rectangle]: A new rectangle representing the overlap, or None if there is no overlap.
        """
        left = max(self.left, other[0])
        top = max(self.top, other[1])
        right = min(self.right, other[0] + other[2])
        bottom = min(self.bottom, other[1] + other[3])
        if right - left > 0 and bottom - top > 0:
            return Rectangle(left, top, right - left, bottom - top)
        return None

    def center(self: Self) -> Point:
        """Get the center point of the rectangle.

        Returns:
            autocv.models.point: The center point of the rectangle.
        """
        return Point(self.left + self.width // 2, self.top + self.height // 2)

    def random_point(self: Self) -> Point:
        """Get a random point of the rectangle.

        Returns:
            autocv.models.point: The random point of the rectangle.
        """
        return Point(
            random.randint(self.left, self.left + self.width),
            random.randint(self.top, self.top + self.height),
        )

    @classmethod
    def from_row(cls: type[Rectangle], row: dict[str, int]) -> Rectangle:
        """Creates a Rectangle object from a dictionary-like object.

        Args:
            row (Dict[str, int]): A dictionary-like object with the following keys:
                              "left", "top", "width", "height".

        Returns:
            Rectangle: A Rectangle object with the specified attributes.
        """
        return cls(row["left"], row["top"], row["width"], row["height"])

    @classmethod
    def from_dimensions(cls: type[Rectangle], dimensions: Sequence[int]) -> Rectangle:
        """Create a new rectangle from its dimensions.

        Args:
            dimensions (Sequence[int]): A sequence of integers containing the left, top, width, and height of the
                rectangle.

        Returns:
            autocv.models.Rectnangle: A new rectangle.
        """
        if len(dimensions) != MAX_SIDES:
            raise InvalidLengthError(MAX_SIDES, len(dimensions))

        return cls(*dimensions)

    @classmethod
    def from_coordinates(cls: type[Rectangle], coordinates: Sequence[int]) -> Rectangle:
        """Create a new rectangle from its coordinates.

        Args:
            coordinates (Sequence[int]): A sequence of integers containing the x1, y1, x2, and y2 coordinates of the
                rectangle.

        Returns:
            autocv.models.Rectangle: A new rectangle.
        """
        if len(coordinates) != MAX_SIDES:
            raise InvalidLengthError(MAX_SIDES, len(coordinates))

        x1, y1, x2, y2 = coordinates
        return cls(x1, y1, x2 - x1, y2 - y1)

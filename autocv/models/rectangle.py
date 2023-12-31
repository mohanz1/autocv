import random
from collections.abc import Sequence
from typing import NamedTuple, Optional

import numpy as np
import numpy.typing as npt

from autocv.models.exceptions import InvalidLengthError
from autocv.models.point import Point

__all__ = ("Rectangle",)


class Rectangle(NamedTuple):
    """A class to represent a rectangle.

    Attributes
    ----------
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
    def right(self) -> int:
        """Calculate the right edge of the rectangle.

        Returns
        -------
            int: The x-coordinate of the right edge of the rectangle.
        """
        return self.left + self.width

    @property
    def bottom(self) -> int:
        """Calculate the bottom edge of the rectangle.

        Returns
        -------
            int: The y-coordinate of the bottom edge of the rectangle.
        """
        return self.top + self.height

    @classmethod
    def from_ndarray(cls, data: npt.NDArray[np.uintp]) -> "Rectangle":
        return cls(*data)

    def area(self) -> int:
        """Returns the area of the rectangle.

        Returns
        -------
            int: The area of the rectangle.
        """
        return self.width * self.height

    def get_overlap(self, other: tuple[int, int, int, int]) -> Optional["Rectangle"]:
        """Get the overlap between this rectangle and another rectangle.

        Args:
        ----
            other (Tuple[int, int, int, int]): The rectangle to get the overlap with.

        Returns:
        -------
            Optional[autocv.models.Rectangle]: A new rectangle representing the overlap, or None if there is no overlap.
        """
        left = max(self.left, other[0])
        top = max(self.top, other[1])
        right = min(self.right, other[0] + other[2])
        bottom = min(self.bottom, other[1] + other[3])
        if right - left > 0 and bottom - top > 0:
            return Rectangle(left, top, right - left, bottom - top)
        return None

    def center(self) -> Point:
        """Get the center point of the rectangle.

        Returns
        -------
            autocv.models.point: The center point of the rectangle.
        """
        return Point(self.left + self.width // 2, self.top + self.height // 2)

    def random_point(self) -> Point:
        """Get a random point of the rectangle.

        Returns
        -------
            autocv.models.point: The random point of the rectangle.
        """
        return Point(
            random.randint(self.left, self.left + self.width),
            random.randint(self.top, self.top + self.height),
        )

    @classmethod
    def from_row(cls, row: dict[str, int]) -> "Rectangle":
        """Creates a Rectangle object from a dictionary-like object.

        Args:
        ----
            row (Dict[str, int]): A dictionary-like object with the following keys:
                              "left", "top", "width", "height".

        Returns:
        -------
            Rectangle: A Rectangle object with the specified attributes.
        """
        return cls(row["left"], row["top"], row["width"], row["height"])

    @classmethod
    def from_dimensions(cls, dimensions: Sequence[int]) -> "Rectangle":
        """Create a new rectangle from its dimensions.

        Args:
        ----
            dimensions (Sequence[int]): A sequence of integers containing the left, top, width, and height of the rectangle.

        Returns:
        -------
            autocv.models.Rectnangle: A new rectangle.
        """
        if len(dimensions) != 4:
            raise InvalidLengthError(
                f"Expected dimensions of length 4. Instead received dimensions of length {len(dimensions)}.",
            )

        return cls(*dimensions)

    @classmethod
    def from_coordinates(cls, coordinates: Sequence[int]) -> "Rectangle":
        """Create a new rectangle from its coordinates.

        Args:
        ----
            coordinates (Sequence[int]): A sequence of integers containing the x1, y1, x2, and y2 coordinates of the rectangle.

        Returns:
        -------
            autocv.models.Rectangle: A new rectangle.
        """
        if len(coordinates) != 4:
            raise InvalidLengthError(
                f"Expected coordinates of length 4. Instead received coordinates of length {len(coordinates)}.",
            )

        x1, y1, x2, y2 = coordinates
        return cls(x1, y1, x2 - x1, y2 - y1)

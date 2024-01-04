"""This module defines the Point class for representing and manipulating points in 2D space.

The Point class includes methods for subtracting one point from another, finding the center of a point, generating a
random point, creating a point from a numpy array, and calculating the Euclidean distance between two points.
"""

from __future__ import annotations

__all__ = ("Point",)

import math
from typing import NamedTuple, TYPE_CHECKING
from typing_extensions import Self

if TYPE_CHECKING:
    import numpy as np
    import numpy.typing as npt


class Point(NamedTuple):
    """A class representing a point in 2D space.

    Args:
    ----
        x: The x-coordinate of the point.
        y: The y-coordinate of the point.
    """

    x: int
    y: int

    def __sub__(self: Self, other: tuple[int, int]) -> Point:
        """Calculate the vector difference between this point and another point.

        Args:
        ----
            other: The other Point instance.

        Returns:
        -------
            A new Point instance representing the vector difference between the two points.
        """
        return Point(self.x - other[0], self.y - other[1])

    def center(self: Self) -> Point:
        """Calculate the center of the point, which is simply the point itself.

        Returns:
        -------
            autocv.models.Point: The center of the point.
        """
        return self

    def random_point(self: Self) -> Point:
        """Calculate a random point, which is simply the point itself.

        Returns:
        -------
            autocv.models.Point: The center of the point.
        """
        return self

    @classmethod
    def from_ndarray(cls: Point, data: npt.NDArray[np.uintp]) -> Point:
        """Create a Point object from a numpy array of shape (2,) containing the x and y coordinates.

        Args:
        ----
            data (npt.NDArray[np.uintp]): A numpy array of shape (2,) containing the x and y coordinates.

        Returns:
        -------
            autocv.models.Point: A Point object with the specified coordinates.
        """
        return cls(int(data[0]), int(data[1]))

    def distance_to(self: Self, other: tuple[int, int]) -> float:
        """Calculate the Euclidean distance between this point and another point.

        Args:
        ----
            other (Tuple[int, int]): A Point instance or a sequence of two integers representing the x and y coordinates
                of another point.

        Returns:
        -------
            float: The Euclidean distance between the two points.
        """
        return math.dist((self.x, self.y), (other[0], other[1]))

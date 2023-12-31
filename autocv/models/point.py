import math
from typing import NamedTuple

import numpy as np
import numpy.typing as npt

__all__ = ("Point",)


class Point(NamedTuple):
    """A class representing a point in 2D space.

    Args:
    ----
        x: The x-coordinate of the point.
        y: The y-coordinate of the point.
    """

    x: int
    y: int

    def __sub__(self, other: tuple[int, int]) -> "Point":
        """Calculate the vector difference between this point and another point.

        Args:
        ----
            other: The other Point instance.

        Returns:
        -------
            A new Point instance representing the vector difference between the two points.
        """
        return Point(self.x - other[0], self.y - other[1])

    def center(self) -> "Point":
        """Calculate the center of the point, which is simply the point itself.

        Returns
        -------
            autocv.models.Point: The center of the point.
        """
        return self

    def random_point(self) -> "Point":
        """Calculate a random point, which is simply the point itself.

        Returns
        -------
            autocv.models.Point: The center of the point.
        """
        return self

    @classmethod
    def from_ndarray(cls, data: npt.NDArray[np.uintp]) -> "Point":
        """Create a Point object from a numpy array of shape (2,) containing the x and y coordinates.

        Args:
        ----
            data (npt.NDArray[np.uintp]): A numpy array of shape (2,) containing the x and y coordinates.

        Returns:
        -------
            autocv.models.Point: A Point object with the specified coordinates.
        """
        return cls(data[0], data[1])

    def distance_to(self, other: tuple[int, int]) -> float:
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

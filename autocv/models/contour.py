"""Module for representing and manipulating contours in 2D space.

It includes the Contour class which encapsulates methods for computing properties of contours such as area, perimeter,
centroid, and methods for conversion to other shapes.
"""

from __future__ import annotations

__all__ = ("Contour",)

from abc import abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass
from itertools import starmap
from typing import cast, overload

import cv2 as cv
import numpy as np
import numpy.typing as npt
from typing_extensions import Self

from .circle import Circle
from .point import Point
from .rectangle import Rectangle

SLOTS_DATACLASS = {"slots": True} if "slots" in dataclass.__kwdefaults__ else {}


@dataclass(frozen=True, **SLOTS_DATACLASS)
class Contour(Sequence[Sequence[tuple[int, int]]]):
    """Represents a contour in 2D space with various geometric properties and methods."""

    data: npt.NDArray[np.uintp]

    def __len__(self: Self) -> int:
        """Return the number of points in the contour."""
        return len(self.data)

    @overload
    @abstractmethod
    def __getitem__(self: Self, index: int) -> Sequence[tuple[int, int]]: ...

    @overload
    @abstractmethod
    def __getitem__(self: Self, index: slice) -> Sequence[Sequence[tuple[int, int]]]: ...

    def __getitem__(
        self,
        index: int | slice,
    ) -> Sequence[Sequence[tuple[int, int]]] | Sequence[tuple[int, int]]:
        """Retrieve a point or a sequence of points from the contour by index."""
        return cast(
            Sequence[Sequence[tuple[int, int]]] | Sequence[tuple[int, int]],
            self.data[index].tolist(),
        )

    @classmethod
    def from_ndarray(cls: type[Contour], data: npt.NDArray[np.uintp]) -> Contour:
        """Create a new Contour instance from a given numpy ndarray.

        Args:
            data (npt.NDArray[np.uintp]): The ndarray to use for creating the Contour.

        Returns:
            autocv.models.Contour: A new Contour instance created from the given ndarray.
        """
        return cls(data)

    def area(self: Self) -> float:
        """Compute the area of the contour.

        Returns:
            float: A float representing the area of the contour.
        """
        return float(cv.contourArea(self.data))

    def perimeter(self: Self) -> float:
        """Compute the perimeter of the contour.

        Returns:
            float: A float representing the perimeter of the contour.
        """
        return float(cv.arcLength(self.data, closed=True))

    def centroid(self: Self) -> Point:
        """Compute the centroid of the contour.

        Returns:
            autocv.models.Point: A Point representing the (x, y) coordinates of the centroid of the contour.
        """
        m = cv.moments(self.data)
        cx = int(m["m10"] / m["m00"])
        cy = int(m["m01"] / m["m00"])
        return Point(cx, cy)

    def center(self: Self) -> Point:
        """Compute the centroid of the contour.

        Returns:
            autocv.models.Point: A Point representing the (x, y) coordinates of the centroid of the contour.
        """
        return self.centroid()

    def is_point_inside_contour(self: Self, x: int, y: int) -> bool:
        """Check whether a given point lies within the contour.

        Args:
            x (int): The x-coordinate of the point.
            y (int): The y-coordinate of the point.

        Returns:
            bool: True if the point lies inside the contour, False otherwise.
        """
        return bool(cv.pointPolygonTest(self.data, (x, y), measureDist=False) >= 0)

    def random_point(self: Self) -> Point:
        """Get a random point inside the contour.

        Returns:
            autocv.models.Point: A Point object representing the random point.
        """
        (cx, cy), radius = cv.minEnclosingCircle(self.data)
        while True:
            x, y = np.random.normal((cx, cy), radius, size=2)
            if self.is_point_inside_contour(x, y):
                return Point(int(x), int(y))

    def to_points(self: Self) -> Sequence[Point]:
        """Convert the contour to a list of points.

        Returns:
            Sequence[autocv.models.Point]: A list of Point objects representing the points in the contour.
        """
        return tuple(starmap(Point, self.data.squeeze()))

    def get_bounding_rect(self: Self) -> Rectangle:
        """Returns a Rectangle object representing the minimum bounding box that encloses the contour.

        Returns:
            Rectangle: A Rectangle object representing the minimum bounding box that encloses the contour.
        """
        return Rectangle(*cv.boundingRect(self.data))

    def get_bounding_circle(self: Self) -> Circle:
        """Returns a Circle object representing the minimum bounding circle that encloses the contour.

        Returns:
            Circle: A Circle object representing the minimum bounding box that encloses the contour.
        """
        (x, y), radius = cv.minEnclosingCircle(self.data)
        return Circle(int(x), int(y), int(radius))

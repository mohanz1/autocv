"""This module defines the Circle class, which is used for representing circles in a 2D space.

It includes functionality for identifying the center of the circle and generating random points within the circle. The
Circle class is defined with center coordinates and a radius and provides methods to interact with the geometric
properties of the circle.
"""

from __future__ import annotations

__all__ = ("Circle",)

import math
import random
from typing import NamedTuple, Self

from .point import Point


class Circle(NamedTuple):
    """A class to represent a rectangle.

    Attributes:
    ----------
    - x (int): The x-coordinate of the center of the circle.
    - y (int): The y-coordinate of the center of the circle.
    - radius (int): The radius of the circle.
    """

    x: int
    y: int
    radius: int

    def center(self: Self) -> Point:
        """Get the center point of the circle.

        Returns:
        -------
            autocv.models.point: The center point of the circle.
        """
        return Point(self.x, self.y)

    def random_point(self: Self) -> Point:
        """Get a random point of the circle.

        Returns:
        -------
            autocv.models.point: The random point of the circle.
        """
        alpha = 2 * math.pi * random.random()

        # Generate a random radius
        r_rand = self.radius * math.sqrt(random.random())  # sqrt is used to ensure uniform distribution

        # Convert polar coordinates to Cartesian coordinates
        x = r_rand * math.cos(alpha) + self.x
        y = r_rand * math.sin(alpha) + self.y

        return Point(int(x), int(y))

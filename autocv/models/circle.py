import math
import random
from typing import NamedTuple

from .point import Point

__all__ = ("Circle",)


class Circle(NamedTuple):
    """A class to represent a rectangle.

    Attributes
    ----------
    - x (int): The x-coordinate of the center of the circle.
    - y (int): The y-coordinate of the center of the circle.
    - radius (int): The radius of the circle.
    """

    x: int
    y: int
    radius: int

    def center(self) -> Point:
        """Get the center point of the circle.

        Returns
        -------
            autocv.models.point: The center point of the circle.
        """
        return Point(self.x, self.y)

    def random_point(self) -> Point:
        """Get a random point of the circle.

        Returns
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

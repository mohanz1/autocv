"""This module defines the ColorWithPoint class.

This is a data structure to store a color and its corresponding point in 2D space. It includes methods to construct this
class from various types of sequences and numpy arrays.
"""

from __future__ import annotations

__all__ = ("Color",)

from typing import NamedTuple

import numpy as np
import numpy.typing as npt
from typing_extensions import Self


class Color(NamedTuple):
    """A class representing a color as a combination of red, green, and blue components.

    Attributes:
        r (int): An integer representing the red component.
        g (int): An integer representing the green component.
        b (int): An integer representing the blue component.
    """

    r: int
    g: int
    b: int

    def __array__(self: Self, dtype: npt.DTypeLike = np.int16) -> npt.NDArray[np.int16]:  # noqa: PLW3201
        """Return a numpy array of the red, green, and blue components.

        Args:
            dtype (np.int16): The desired data-type for the array.  If not given, then the type will
            be determined as the minimum type required to hold the objects in the
            sequence.

        Returns:
            npt.NDArray[np.int16]: A 1D numpy array of shape (3,) representing the red, green, and blue components.
        """
        return np.array([self.r, self.g, self.b], dtype=dtype)

    def is_color_within_color_and_tolerance(self: Self, color: tuple[int, int, int], tolerance: int = 0) -> bool:
        """Check if this color is within the specified color and tolerance.

        Args:
            color (tuple[int, int, int]): The color to compare against.
            tolerance (int): The maximum difference allowed for each channel. Defaults to 0.

        Returns:
            bool: True if this color is within the specified color and tolerance, False otherwise.
        """
        # Convert this color to a numpy array
        current_color = np.array(self)

        # Convert the target color to a numpy array and set the lower and upper bounds for each channel
        color_array = np.array(color, dtype=np.int16)
        lower_bounds = np.clip(color_array - tolerance, 0, 255)
        upper_bounds = np.clip(color_array + tolerance, 0, 255)

        # Check if this color is within the specified color and tolerance
        return bool(np.all((current_color >= lower_bounds) & (current_color <= upper_bounds)))

    def invert(self: Self, color: Color | tuple[int, int, int] | None = None) -> Color:
        """Return the inverse of the color.

        Args:
            color (Color | tuple[int, int, int] | None): An optional color or sequence of integers representing the
                color to be inverted. If not provided, the color instance on which the method is called is used.

        Returns:
            Color: A new Color instance representing the inverted color.
        """
        color = color or self
        return Color(255 - color[0], 255 - color[1], 255 - color[2])

    @staticmethod
    def to_decimal(color: Color | tuple[int, int, int]) -> int:
        """Return the decimal representation of the color.

        Args:
            color (Color | tuple[int, int, int]): A color or sequence of integers representing the color to be
                converted.

        Returns:
            int: An integer representing the decimal value of the color.
        """
        if isinstance(color, Color):
            color = (color.r, color.g, color.b)
        return (color[0] << 16) + (color[1] << 8) + color[2]

    @staticmethod
    def to_hex(color: Color | tuple[int, int, int]) -> str:
        """Return the hexadecimal representation of the color.

        Args:
            color (Color | tuple[int, int, int]): A color or sequence of integers representing the color to be
                converted.

        Returns:
            str: A string representing the hexadecimal value of the color.
        """
        if isinstance(color, Color):
            color = (color.r, color.g, color.b)
        return f"#{color[0]:02x}{color[1]:02x}{color[2]:02x}"

"""This module defines the ColorWithPoint class.

This is a data structure to store a color and its corresponding point in 2D space. It includes methods to construct this
class from various types of sequences and numpy arrays.
"""

from __future__ import annotations

__all__ = ("ColorWithPoint",)

from dataclasses import astuple, dataclass
from typing import Any, TYPE_CHECKING
from typing_extensions import Self

from .color import Color
from .exceptions import InvalidLengthError
from .point import Point

if TYPE_CHECKING:
    import numpy as np
    import numpy.typing as npt
    from collections.abc import Iterator, Sequence

COLOR_WITH_POINT_LENGTH = 5
SLOTS_DATACLASS = {"slots": True} if "slots" in dataclass.__kwdefaults__ else {}


@dataclass(frozen=True, **SLOTS_DATACLASS)
class ColorWithPoint:
    """Represents a color and its associated point in 2D space."""

    color: Color
    point: Point

    def __iter__(self: Self) -> Iterator[Any]:
        """Returns an iterator over the fields of the class in the order they were defined."""
        yield from astuple(self)

    @classmethod
    def from_sequence(cls: ColorWithPoint, sequence: Sequence[int]) -> ColorWithPoint:
        """Constructs a ColorWithPoint instance from a sequence of integers.

        Args:
        ----
            sequence (Sequence[int]): A sequence of 5 integers in the order x, y, red, green, blue.

        Raises:
        ------
            InvalidLength: If the length of the sequence is not 5.

        Returns:
        -------
            ColorWithPoint: A ColorWithPoint instance.
        """
        if len(sequence) != COLOR_WITH_POINT_LENGTH:
            raise InvalidLengthError(COLOR_WITH_POINT_LENGTH, len(sequence))

        r, g, b, x, y = sequence
        return cls(Color(r, g, b), Point(x, y))

    @classmethod
    def from_ndarray_sequence(
        cls: ColorWithPoint, data: npt.NDArray[np.uint8], sequence: Sequence[int]
    ) -> ColorWithPoint:
        """Constructs a ColorWithPoint instance from a numpy ndarray and a sequence of integers.

        Args:
        ----
            data (npt.NDArray[np.uint8]): A numpy ndarray of 3 integers representing the red, green, and blue values.
            sequence (Sequence[int]): A sequence of 2 integers representing the x and y coordinates.

        Returns:
        -------
            A ColorWithPoint instance.
        """
        return cls(Color(*data), Point(sequence[0], sequence[1]))

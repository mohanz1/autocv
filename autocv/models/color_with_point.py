from collections.abc import Iterator, Sequence
from dataclasses import astuple, dataclass
from typing import Any

import numpy as np
import numpy.typing as npt

from .color import Color
from .exceptions import InvalidLengthError
from .point import Point

__all__ = ("ColorWithPoint",)

SLOTS_DATACLASS = dict(slots=True) if "slots" in dataclass.__kwdefaults__ else {}


@dataclass(frozen=True, **SLOTS_DATACLASS)
class ColorWithPoint:
    color: Color
    point: Point

    def __iter__(self) -> Iterator[Any]:
        """Returns an iterator over the fields of the class in the order they were defined."""
        yield from astuple(self)

    @classmethod
    def from_sequence(cls, sequence: Sequence[int]) -> "ColorWithPoint":
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
        if len(sequence) != 5:
            raise InvalidLengthError(f"Expected Sequence of length 5. Instead received sequence of length {len(sequence)}.")

        r, g, b, x, y = sequence
        return cls(Color(r, g, b), Point(x, y))

    @classmethod
    def from_ndarray_sequence(cls, data: npt.NDArray[np.uint8], sequence: Sequence[int]) -> "ColorWithPoint":
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
